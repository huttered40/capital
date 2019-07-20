/* Author: Edward Hutter */

#include "CholeskyQR2.h"
#include "../QRvalidate/QRvalidate.h"

using namespace std;

int main(int argc, char** argv){
  using MatrixTypeS = Matrix<DATATYPE,INTTYPE,Square,Cyclic>;
  using MatrixTypeR = Matrix<DATATYPE,INTTYPE,Rectangular,Cyclic>;

#ifdef PROFILE
  TAU_PROFILE_SET_CONTEXT(0)
#endif /*PROFILE*/

  // argv[1] - Matrix size x where x represents 2^x.
  // So in future, we might want t way to test non power of 2 dimension matrices

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  util::InitialGEMM<DATATYPE>();

  // size -- total number of processors in the tunable grid
  INTTYPE globalMatrixDimensionM = atoi(argv[1]);
  INTTYPE globalMatrixDimensionN = atoi(argv[2]);

  size_t baseCaseMultiplier = atoi(argv[3]);
  size_t inverseCutOffMultiplier = atoi(argv[4]);
  size_t panelDimensionMultiplier = atoi(argv[5]);
 
  // Use the grid that the user specifies in the command line
  size_t dimensionD = atoi(argv[6]);
  size_t dimensionC = atoi(argv[7]);
  size_t sliceSize = dimensionD*dimensionC;
  #if defined(BLUEWATERS) || defined(STAMPEDE2)
  size_t helper = dimensionC*dimensionC;
  size_t pCoordY = rank/helper;
  size_t pCoordX = (rank%helper)/dimensionC;
  #else
  size_t pCoordX = rank%dimensionC;
  size_t pCoordY = (rank%sliceSize)/dimensionC;
  #endif

  size_t numIterations=atoi(argv[8]);
  string fileStr = argv[9];
  string fileStrTotal=fileStr;
  #ifdef PROFILE
  fileStrTotal += "_timer.txt";
  #endif
  #ifdef CRITTER
  fileStrTotal += "_critter.txt";
  #endif
  #ifdef PERFORMANCE
  string fileStrNumericsTotal=fileStr;
  fileStrTotal += "_perf.txt";
  fileStrNumericsTotal += "_numerics.txt";
  ofstream fptrNumericsTotal;
  #endif
  ofstream fptrTotal;
  if (rank == 0){
    fptrTotal.open(fileStrTotal.c_str());
    #ifdef PERFORMANCE
    fptrNumericsTotal.open(fileStrNumericsTotal.c_str());
    #endif
  }

  // Note: matA and matR are rectangular, but the pieces owned by the individual processors may be square (so also rectangular)
  MatrixTypeR matA(globalMatrixDimensionN,globalMatrixDimensionM, dimensionC, dimensionD);
  MatrixTypeS matR(globalMatrixDimensionN,globalMatrixDimensionN, dimensionC, dimensionC);

  #ifdef PERFORMANCE
  DATATYPE totalError1 = 0;
  DATATYPE totalError2 = 0;
  double totalTime = 0;
  #endif

  #ifdef CRITTER
  std::vector<size_t> Inputs{globalMatrixDimensionM,globalMatrixDimensionN,dimensionD,dimensionC,baseCaseMultiplier,inverseCutOffMultiplier,panelDimensionMultiplier};
  std::vector<const char*> InputNames{"m","n","d","c","bcm","icm","pdm"};
  #endif

  size_t numFuncs = 0;				// For figuring out how many functions are being profiled (smart way to find average over all iterations)
  size_t i;
  for (i=0; i<numIterations; i++){
    // reset the matrix before timer starts
    #if defined(BLUEWATERS) || defined(STAMPEDE2)
    matA.DistributeRandom(pCoordX, pCoordY, dimensionC, dimensionD, rank/dimensionC);
    #else
    matA.DistributeRandom(pCoordX, pCoordY, dimensionC, dimensionD, (rank%sliceSize));
    #endif
    MPI_Barrier(MPI_COMM_WORLD);	// make sure each process starts together
    #ifdef CRITTER
    Critter::reset();
    #endif
    TAU_FSTART(Total);
    #ifdef PERFORMANCE
    volatile double startTime=MPI_Wtime();
    #endif
    auto commInfoTunable = util::buildTunableTopology(MPI_COMM_WORLD, dimensionD, dimensionC);
    CholeskyQR2::FactorTunable(matA, matR, dimensionD, dimensionC, MPI_COMM_WORLD, commInfoTunable, inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier);
    util::destroyTunableTopology(commInfoTunable);
    #ifdef PERFORMANCE
    double iterTimeLocal = MPI_Wtime() - startTime;
    double iterTimeGlobal = 0;
    MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      fptrTotal << size << "\t" << i << "\t" << globalMatrixDimensionM << "\t" << globalMatrixDimensionN << "\t" << iterTimeGlobal << endl;
      totalTime += iterTimeGlobal;
    }
    #endif
    TAU_FSTOP_FILE(Total, fptrTotal, i, numFuncs);
    #ifdef CRITTER
    Critter::print(fptrTotal, "QR", size, Inputs.size(), &Inputs[0], &InputNames[0]);
    #endif

    #ifdef PERFORMANCE
    MatrixTypeR saveA = matA;
    #if defined(BLUEWATERS) || defined(STAMPEDE2)
    saveA.DistributeRandom(pCoordX, pCoordY, dimensionC, dimensionD, rank/dimensionC);
    #else
    saveA.DistributeRandom(pCoordX, pCoordY, dimensionC, dimensionD, (rank%sliceSize));
    #endif
    commInfoTunable = util::buildTunableTopology(MPI_COMM_WORLD, dimensionD, dimensionC);
    pair<DATATYPE,DATATYPE> error = QRvalidate::validateParallelTunable(saveA, matA, matR, dimensionD, dimensionC, MPI_COMM_WORLD, commInfoTunable);
    util::destroyTunableTopology(commInfoTunable);
    double residualErrorGlobal,orthogonalityErrorGlobal;
    MPI_Reduce(&error.first, &residualErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&error.second, &orthogonalityErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0){
      fptrNumericsTotal << size << "\t" << i << "\t" << residualErrorGlobal << "\t" << orthogonalityErrorGlobal << endl;
      totalError1 += residualErrorGlobal;
      totalError2 += orthogonalityErrorGlobal;
    }
    #endif
  }
  if (rank == 0){
    fptrTotal.close();
    #ifdef PERFORMANCE
    fptrNumericsTotal.close();
    #endif
  }
  MPI_Finalize();
  return 0;
}
