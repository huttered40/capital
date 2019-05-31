/* Author: Edward Hutter */

#include "CFR3D.h"
#include "../CFvalidate/CFvalidate.h"

using namespace std;

template<typename MatrixAType, typename MatrixTType>
static pair<typename MatrixAType::ScalarType,double>
       runTestCF(MatrixAType& matA, MatrixTType& matT, char dir, size_t inverseCutOffMultiplier, size_t blockSizeMultiplier, size_t panelDimensionMultiplier,
                 size_t pCoordX, size_t pCoordY, size_t pGridDimensionSize, ofstream& fptrTotal,
                 #ifdef PERFORMANCE
                 ofstream& fptrNumericsTotal,
                 #endif
                 size_t iterNum, size_t numIter, size_t rank, size_t size, size_t& numFuncs){
  using T = typename MatrixAType::ScalarType;
  double iterTimeGlobal=-1;
  T iterErrorGlobal;		// define this out here so that compilation doesn't fail with Critter/Analysis runs
  // Reset matrixA
  // Note: I think this call below is still ok given the new topology mapping on Blue Waters/Stampede2
  matA.DistributeSymmetric(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize+pCoordY, true);
  MPI_Barrier(MPI_COMM_WORLD);		// make sure each process starts together
  #ifdef CRITTER
  Critter_Clear();
  #endif
  TAU_FSTART(Total);
  #ifdef PERFORMANCE
  double startTime=MPI_Wtime();
  #endif
  auto commInfo3D = util::build3DTopology(MPI_COMM_WORLD);
  CFR3D::Factor(matA, matT, inverseCutOffMultiplier, blockSizeMultiplier, panelDimensionMultiplier, dir, MPI_COMM_WORLD, commInfo3D);
  util::destroy3DTopology(commInfo3D);
  #ifdef PERFORMANCE
  double iterTimeLocal=MPI_Wtime();
  iterTimeLocal -= startTime;
  MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0) { fptrTotal << size << "\t" << iterNum << "\t" << matA.getNumRowsGlobal() << "\t" << iterTimeGlobal << endl; }
  #endif
  TAU_FSTOP_FILE(Total, fptrTotal, iterNum, numFuncs);
  #ifdef CRITTER
  Critter_Print(fptrTotal, iterNum);
  #endif

  #ifdef PERFORMANCE
/* Sequential validation is no longer in the codepath. For use, create a new branch and comment in this code.
  if (methodKey2 == 0)
  {
    Matrix<T,U,StructureA,Distribution> saveA = matA;
    matA.DistributeSymmetric(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize+pCoordY, true);
    matA.DistributeSymmetric(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize+pCoordY, true);
    CFvalidate<T,U>::validateLocal(saveA, matA, dir, MPI_COMM_WORLD);
  }
*/
  MatrixAType saveA = matA;
  // Note: I think this call below is still ok given the new topology mapping on Blue Waters/Stampede2
  saveA.DistributeSymmetric(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize+pCoordY, true);
  commInfo3D = util::build3DTopology(MPI_COMM_WORLD);
  T iterErrorLocal;
  iterErrorLocal = CFvalidate::validateParallel(saveA, matA, dir, MPI_COMM_WORLD, commInfo3D);
  MPI_Reduce(&iterErrorLocal, &iterErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  util::destroy3DTopology(commInfo3D);
  #endif 
  return make_pair(iterErrorGlobal, iterTimeGlobal);
}

int main(int argc, char** argv){
  using MatrixTypeA = Matrix<DATATYPE,INTTYPE,Square,Cyclic>;
  using MatrixTypeL = Matrix<DATATYPE,INTTYPE,Square,Cyclic>;
  using MatrixTypeR = Matrix<DATATYPE,INTTYPE,Square,Cyclic>;

  #ifdef PROFILE
  TAU_PROFILE_SET_CONTEXT(0)
  #endif /*PROFILE*/

  // argv[1] - Matrix size x where x represents 2^x.
  // So in future, we might want t way to test non power of 2 dimension matrices

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // size -- total number of processors in the 3D grid
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  util::InitialGEMM<DATATYPE>();

  size_t pGridDimensionSize = std::nearbyint(std::pow(size,1./3.));
  size_t helper = pGridDimensionSize;
  helper *= helper;
  #if defined(BLUEWATERS) || defined(STAMPEDE2)
  size_t pCoordZ = rank%pGridDimensionSize;
  size_t pCoordY = rank/helper;
  size_t pCoordX = (rank%helper)/pGridDimensionSize;
  #else
  size_t pCoordX = rank%pGridDimensionSize;
  size_t pCoordY = (rank%helper)/pGridDimensionSize;
  size_t pCoordZ = rank/helper;
  #endif

  /*
    methodKey1 -> 0) Lower
		  1) Upper
  */
//  size_t methodKey1 = atoi(argv[1]);
//  char dir = (methodKey1==0 ? 'L' : 'U');
  char dir = 'U';
  INTTYPE globalMatrixSize = atoi(argv[1]);
  size_t blockSizeMultiplier = atoi(argv[2]);
  size_t inverseCutOffMultiplier = atoi(argv[3]); // multiplies baseCase dimension by sucessive 2
  size_t panelDimensionMultiplier = atoi(argv[4]);
  size_t numIterations = atoi(argv[5]);

  string fileStr = argv[6];
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

  MatrixTypeA matA(globalMatrixSize,globalMatrixSize, pGridDimensionSize, pGridDimensionSize);
  MatrixTypeA matT(globalMatrixSize,globalMatrixSize, pGridDimensionSize, pGridDimensionSize);

  #ifdef PERFORMANCE
  DATATYPE totalError = 0;
  double totalTime = 0;
  #endif
  size_t numFuncs=0;
  for (size_t i=0; i<numIterations; i++){
    pair<DATATYPE,double> info = runTestCF(matA, matT, dir, inverseCutOffMultiplier, blockSizeMultiplier, panelDimensionMultiplier, pCoordX, pCoordY, pGridDimensionSize,
      fptrTotal,
      #ifdef PERFORMANCE
      fptrNumericsTotal,
      #endif
      i, numIterations, rank, size, numFuncs);
    
    #ifdef PERFORMANCE
    if (rank == 0){
      fptrNumericsTotal << size << "\t" << i << "\t" << info.first << endl;
      totalError += info.first;
      totalTime += info.second;
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
