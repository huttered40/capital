/* Author: Edward Hutter */

// System includes
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <utility>

// Local includes
#include "./../../../Util/shared.h"
#include "CholeskyQR2.h"
#include "./../QRvalidate/QRvalidate.h"
#include "../../../Timer/Timer.h"

using namespace std;

// Idea: We calculate 3D Summa as usual, and then we pass it into the MMvalidate solo class

int main(int argc, char** argv)
{
  using MatrixTypeS = Matrix<DATATYPE,INTTYPE,MatrixStructureSquare,MatrixDistributerCyclic>;
  using MatrixTypeR = Matrix<DATATYPE,INTTYPE,MatrixStructureRectangle,MatrixDistributerCyclic>;
  using MatrixTypeLT = Matrix<DATATYPE,INTTYPE,MatrixStructureLowerTriangular,MatrixDistributerCyclic>;
  using MatrixTypeUT = Matrix<DATATYPE,INTTYPE,MatrixStructureSquare,MatrixDistributerCyclic>;

#ifdef PROFILE
  TAU_PROFILE_SET_CONTEXT(0)
#endif /*PROFILE*/

  // argv[1] - Matrix size x where x represents 2^x.
  // So in future, we might want t way to test non power of 2 dimension matrices

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // size -- total number of processors in the tunable grid
  INTTYPE globalMatrixDimensionM = atoi(argv[1]);
  INTTYPE globalMatrixDimensionN = atoi(argv[2]);

  int baseCaseMultiplier = atoi(argv[3]);
  int inverseCutOffMultiplier = atoi(argv[4]);
  int panelDimensionMultiplier = atoi(argv[5]);
 
  // Use the grid that the user specifies in the command line
  int dimensionD = atoi(argv[6]);
  int dimensionC = atoi(argv[7]);
  int sliceSize = dimensionD*dimensionC;
  int pCoordX = rank%dimensionC;
  int pCoordY = (rank%sliceSize)/dimensionC;
  int pCoordZ = rank/sliceSize;

  INTTYPE localMatrixDimensionM = globalMatrixDimensionM/dimensionD;
  INTTYPE localMatrixDimensionN = globalMatrixDimensionN/dimensionC;

  int numIterations=atoi(argv[8]);
  string fileStr = argv[9];
  string fileStrTotal=fileStr;
  string fileStrAvg=fileStr;
  string fileStrNumericsTotal=fileStr + "_numerics.txt";
  string fileStrNumericsAvg=fileStr + "_numerics_avg.txt";
  #ifdef PROFILE
  fileStrTotal += "_timer.txt";
  fileStrAvg += "_timer_avg.txt";
  #endif
  #ifdef CRITTER
  fileStrTotal += "_critter.txt";
  fileStrAvg += "_critter_avg.txt";
  #endif
  #ifdef PERFORMANCE
  fileStrTotal += "_perf.txt";
  fileStrAvg += "_perf_avg.txt";
  #endif
  FILE* fptrTotal = fopen(fileStrTotal.c_str(),"w");
  FILE* fptrAvg = fopen(fileStrAvg.c_str(),"w");
  FILE* fptrNumericsTotal = fopen(fileStrNumericsTotal.c_str(),"w");
  FILE* fptrNumericsAvg = fopen(fileStrNumericsAvg.c_str(),"w");

  // Note: matA and matR are rectangular, but the pieces owned by the individual processors may be square (so also rectangular)
  MatrixTypeR matA(globalMatrixDimensionN,globalMatrixDimensionM, dimensionC, dimensionD);
  MatrixTypeS matR(globalMatrixDimensionN,globalMatrixDimensionN, dimensionC, dimensionC);

  // Loop for getting a good range of results.
  DATATYPE totalError1 = 0;
  DATATYPE totalError2 = 0;
  double totalTime = 0;
  int numFuncs = 0;				// For figuring out how many functions are being profiled (smart way to find average over all iterations)
  for (int i=0; i<numIterations; i++)
  {
    double saveTime;
    // reset the matrix before timer starts
    matA.DistributeRandom(pCoordX, pCoordY, dimensionC, dimensionD, (rank%sliceSize));
    #ifdef CRITTER
    Critter_Clear();
    #endif
    TAU_FSTART(Total);
    #ifdef PERFORMANCE
    double startTime=MPI_Wtime();
    #endif
    auto commInfoTunable = util<DATATYPE,INTTYPE>::buildTunableTopology(
      MPI_COMM_WORLD, dimensionD, dimensionC);
    CholeskyQR2<DATATYPE,INTTYPE,cblasEngine>::FactorTunable(
      matA, matR, dimensionD, dimensionC, MPI_COMM_WORLD, commInfoTunable, inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier);
    util<DATATYPE,INTTYPE>::destroyTunableTopology(commInfoTunable);
    #ifdef PERFORMANCE
    if (rank == 0) {
      double totalTimeLocal=MPI_Wtime() - startTime;
      fprintf(fptrTotal, "%d\t %g\n", i, totalTimeLocal);
      totalTime += totalTimeLocal;
      cout << "\nPERFORMANCE\nTotal time: " << totalTimeLocal << endl;
    }
    #endif
    TAU_FSTOP_FILE(Total, fptrTotal, i, numFuncs);
    #ifdef CRITTER
    Critter_Print(fptrTotal, i, fptrAvg, numIterations);
    #endif

    if (rank == 0) { std::cout << "\nNUMERICS\n"; }
    MatrixTypeR saveA = matA;
    saveA.DistributeRandom(pCoordX, pCoordY, dimensionC, dimensionD, (rank%sliceSize));
    commInfoTunable = util<DATATYPE,INTTYPE>::buildTunableTopology(
      MPI_COMM_WORLD, dimensionD, dimensionC);
    pair<DATATYPE,DATATYPE> error = QRvalidate<DATATYPE,INTTYPE>::validateParallelTunable(
      saveA, matA, matR, dimensionD, dimensionC, MPI_COMM_WORLD, commInfoTunable);
    util<DATATYPE,INTTYPE>::destroyTunableTopology(commInfoTunable);
    if (rank == 0)
    {
      fprintf(fptrNumericsTotal, "%d\t %g\t %g\n", i, error.first, error.second);
      totalError1 += error.first;
      totalError2 += error.second;
    }
  }
  if (rank == 0)
  {
    fprintf(fptrNumericsAvg, "%g\t %g\n", totalError1/numIterations, totalError2/numIterations);
    #ifdef PERFORMANCE
    fprintf(fptrAvg, "%g\n", totalTime/numIterations);
    #endif
  }
  fclose(fptrTotal);
  #ifdef PERFORMANCE
  fclose(fptrAvg);
  #endif
  #ifdef CRITTER
  fclose(fptrAvg);
  #endif
  #ifdef PROFILE
  util<DATATYPE,INTTYPE>::processAveragesFromFile(fptrAvg, fileStrTotal, numFuncs, numIterations, rank);
  fclose(fptrAvg);
  #endif

  MPI_Finalize();
  return 0;
}
