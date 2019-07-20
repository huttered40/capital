/* Author: Edward Hutter */

#include "MM3D.h"
#include "../MMvalidate/MMvalidate.h"

using namespace std;

// Idea: We calculate 3D Summa as usual, and then we pass it into the MMvalidate solo class

template<typename MatrixAType, typename MatrixBType, typename MatrixCType>
static double runTestGemm(MatrixAType& matA, MatrixBType& matB, MatrixCType& matC, blasEngineArgumentPackage_gemm<typename MatrixAType::ScalarType>& blasArgs,
                          size_t methodKey3, size_t pCoordX, size_t pCoordY, size_t pGridDimensionSize,
			  ofstream& fptrTotal, size_t iterNum, size_t numIter, size_t rank, size_t size, size_t& numFuncs){
  double iterTimeGlobal;
  // Note: I think these calls below are still ok given the new topology mapping on Blue Waters/Stampede2
  matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize + pCoordY);
  matB.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, (pCoordX*pGridDimensionSize + pCoordY)*(-1));
  matC.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, (pCoordX*pGridDimensionSize + pCoordY)*(-1));
  MPI_Barrier(MPI_COMM_WORLD);		// make sure each process starts together
  #ifdef CRITTER
  Critter::reset();
  #endif
  TAU_FSTART(Total);
  #ifdef PERFORMANCE
  double startTime=MPI_Wtime();
  #endif
  auto commInfo3D = util::build3DTopology(MPI_COMM_WORLD);
  MM3D::Multiply(matA, matB, matC, MPI_COMM_WORLD, commInfo3D, blasArgs, methodKey3);
  util::destroy3DTopology(commInfo3D);
  #ifdef PERFORMANCE
  double iterTimeLocal=MPI_Wtime();
  iterTimeLocal -= startTime;
  MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0) { fptrTotal << size << "\t" << iterNum << "\t" << iterTimeGlobal << endl; }
  #endif
  TAU_FSTOP_FILE(Total, fptrTotal, iterNum, numFuncs);
  #ifdef CRITTER
  Critter::print(fptrTotal, "MatrixMultiplication", size, pGridDimensionSize, pGridDimensionSize);
  #endif
  return iterTimeGlobal;
}

template<typename MatrixAType, typename MatrixBType>
static double runTestTrmm(MatrixAType& matA, MatrixBType& matB, blasEngineArgumentPackage_trmm<typename MatrixAType::ScalarType>& blasArgs,
                          size_t methodKey3, size_t pCoordX, size_t pCoordY, size_t pGridDimensionSize,
			  ofstream& fptrTotal, size_t iterNum, size_t numIter, size_t rank, size_t size, size_t& numFuncs){
  double iterTimeGlobal;
  matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize + pCoordY);
  matB.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, (pCoordX*pGridDimensionSize + pCoordY)*(-1));
  MPI_Barrier(MPI_COMM_WORLD);		// make sure each process starts together
  #ifdef CRITTER
  Critter::clear();
  #endif
  TAU_FSTART(Total);
  #ifdef PERFORMANCE
  double startTime=MPI_Wtime();
  #endif
  auto commInfo3D = util::build3DTopology(
    MPI_COMM_WORLD);
  MM3D::Multiply(matA, matB, MPI_COMM_WORLD, commInfo3D, blasArgs, methodKey3);
  util::destroy3DTopology(commInfo3D);
  #ifdef PERFORMANCE
  double iterTimeLocal=MPI_Wtime();
  iterTimeLocal -= startTime;
  MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0) { fptrTotal << size << "\t" << iterNum << "\t" << iterTimeGlobal << endl; }
  #endif
  TAU_FSTOP_FILE(Total, fptrTotal, iterNum, numFuncs);
  #ifdef CRITTER
  Critter::print(fptrTotal, "MatrixMultiplication");
  #endif
  return iterTimeGlobal;
}


int main(int argc, char** argv){
  using MatrixTypeR = Matrix<DATATYPE,INTTYPE,Rectangular,Cyclic>;
  using MatrixTypeLT = Matrix<DATATYPE,INTTYPE,LowerTriangular,Cyclic>;
  using MatrixTypeUT = Matrix<DATATYPE,INTTYPE,UpperTriangular,Cyclic>;

  #ifdef PROFILE
  TAU_PROFILE_SET_CONTEXT(0)
  #endif /*TIMER*/

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // size -- total number of processors in the 3D grid
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  util::InitialGEMM<DATATYPE>();

  /*
    Choices for methodKey1: 0) Gemm
			    1) TRMM
  */
  size_t methodKey1 = atoi(argv[1]);
  /*
    Choices for methodKey3: 0) Broadcast + Allreduce
			    1) Allgather + Allreduce
  */
  size_t methodKey3 = atoi(argv[2]);

  size_t pGridDimensionSize = std::nearbyint(pow(size,1./3.));
  size_t helper = pGridDimensionSize;
  helper *= helper;
  #if defined(BLUEWATERS) || defined(STAMPEDE2)
  size_t pCoordY = rank/helper;
  size_t pCoordX = (rank%helper)/pGridDimensionSize;
  #else
  size_t pCoordX = rank%pGridDimensionSize;
  size_t pCoordY = (rank%helper)/pGridDimensionSize;
  #endif

  INTTYPE globalMatrixSizeM = atoi(argv[3]);
  INTTYPE globalMatrixSizeN = atoi(argv[4]);

  if (methodKey1 == 0){
    // GEMM
    INTTYPE globalMatrixSizeK = atoi(argv[5]);
    int numIterations = atoi(argv[6]);
    string fileStr = argv[7];
    string fileStrTotal=fileStr;
    #ifdef PROFILE
    fileStrTotal += "_timer.txt";
    #endif
    #ifdef CRITTER
    fileStrTotal += "_critter.txt";
    #endif
    #ifdef PERFORMANCE
    fileStrTotal += "_perf.txt";
    #endif
    ofstream fptrTotal;
    if (rank == 0){
      fptrTotal.open(fileStrTotal.c_str());
    }

    MatrixTypeR matA(globalMatrixSizeK,globalMatrixSizeM,pGridDimensionSize,pGridDimensionSize);
    MatrixTypeR matB(globalMatrixSizeN,globalMatrixSizeK,pGridDimensionSize,pGridDimensionSize);
    MatrixTypeR matC(globalMatrixSizeN,globalMatrixSizeM,pGridDimensionSize,pGridDimensionSize);
    blasEngineArgumentPackage_gemm<DATATYPE> blasArgs(blasEngineOrder::AblasColumnMajor, blasEngineTranspose::AblasNoTrans, blasEngineTranspose::AblasNoTrans, 1., 0.);
  
    // Loop for getting a good range of results.
    double totalTime = 0;
    size_t numFuncs = 0;
    for (size_t i=0; i<numIterations; i++){
      double iterTime = runTestGemm(matA, matB, matC, blasArgs, methodKey3, pCoordX, pCoordY, pGridDimensionSize, fptrTotal, i, numIterations, rank, size, numFuncs);
      totalTime += iterTime;
    }
    MMvalidate::validateLocal(matA,matB,matC,MPI_COMM_WORLD,blasArgs);
    if (rank == 0){
      fptrTotal.close();
    }
  }
  else if (methodKey1 == 1){
    // TRMM
    // First, we need to collect some special arguments.
    /*
      Choices for matrixUpLo: 0) Lower-triangular
			      1) Upper-triangular
    */
    size_t matrixUpLo = atoi(argv[5]);
    /*
      Choices for triangleSide: 0) Triangle * Rectangular (matrixA * matrixB)
			        1) Rectangular * Triangle (matrixB * matrixA)
    */
    size_t triangleSide = atoi(argv[6]);
    size_t numIterations = atoi(argv[7]);
    string fileStr = argv[8];
    string fileStrTotal=fileStr;
    #ifdef PROFILE
    fileStrTotal += "_timer.txt";
    #endif
    #ifdef CRITTER
    fileStrTotal += "_critter.txt";
    #endif
    #ifdef PERFORMANCE
    fileStrTotal += "_perf.txt";
    #endif
    ofstream fptrTotal;
    if (rank == 0){
      fptrTotal.open(fileStrTotal.c_str());
    }

    // I guess I will go through all cases. Ugh!
    double totalTime = 0;
    size_t numFuncs = 0;
    if ((matrixUpLo == 0) && (triangleSide == 0)){
      MatrixTypeLT matA(globalMatrixSizeM,globalMatrixSizeM, pGridDimensionSize,pGridDimensionSize);
      MatrixTypeR matB(globalMatrixSizeN,globalMatrixSizeM, pGridDimensionSize,pGridDimensionSize);
      blasEngineArgumentPackage_trmm<DATATYPE> blasArgs(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasLeft, blasEngineUpLo::AblasLower,
        blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
 
      // Loop for getting a good range of results.
      for (size_t i=0; i<numIterations; i++){
        double iterTime = runTestTrmm(matA, matB, blasArgs, methodKey3, pCoordX, pCoordY, pGridDimensionSize, fptrTotal, i, numIterations, rank, size, numFuncs);
        totalTime += iterTime;
      }
    }
    else if ((matrixUpLo == 0) && (triangleSide == 1)){
      MatrixTypeR matB(globalMatrixSizeN,globalMatrixSizeM, pGridDimensionSize,pGridDimensionSize);
      MatrixTypeLT matA(globalMatrixSizeN,globalMatrixSizeN, pGridDimensionSize,pGridDimensionSize);
      blasEngineArgumentPackage_trmm<DATATYPE> blasArgs(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasRight, blasEngineUpLo::AblasLower,
        blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);

      // Loop for getting a good range of results.
      for (size_t i=0; i<numIterations; i++){
        double iterTime = runTestTrmm(matA, matB, blasArgs, methodKey3, pCoordX, pCoordY, pGridDimensionSize, fptrTotal, i, numIterations, rank, size, numFuncs);
        totalTime += iterTime;
      }
    }
    else if ((matrixUpLo == 1) && (triangleSide == 0)){
      MatrixTypeUT matA(globalMatrixSizeM,globalMatrixSizeM, pGridDimensionSize,pGridDimensionSize);
      MatrixTypeR matB(globalMatrixSizeN,globalMatrixSizeM, pGridDimensionSize,pGridDimensionSize);
      blasEngineArgumentPackage_trmm<DATATYPE> blasArgs(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasLeft, blasEngineUpLo::AblasUpper,
        blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
  
      // Loop for getting a good range of results.
      for (size_t i=0; i<numIterations; i++){
        double iterTime = runTestTrmm(matA, matB, blasArgs, methodKey3, pCoordX, pCoordY, pGridDimensionSize, fptrTotal, i, numIterations, rank, size, numFuncs);
        totalTime += iterTime;
      }
    }
    else if ((matrixUpLo == 1) && (triangleSide == 1)){
      MatrixTypeR matB(globalMatrixSizeN,globalMatrixSizeM, pGridDimensionSize,pGridDimensionSize);
      MatrixTypeUT matA(globalMatrixSizeN,globalMatrixSizeN, pGridDimensionSize,pGridDimensionSize);
      blasEngineArgumentPackage_trmm<DATATYPE> blasArgs(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasRight, blasEngineUpLo::AblasUpper,
        blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);

      // Loop for getting a good range of results.
      for (size_t i=0; i<numIterations; i++){
        double iterTime = runTestTrmm(matA, matB, blasArgs, methodKey3, pCoordX, pCoordY, pGridDimensionSize, fptrTotal, i, numIterations, rank, size, numFuncs);
        totalTime += iterTime;
      }
    }
    if (rank == 0)
    {
      fptrTotal.close();
    }
  }

  MPI_Finalize();
  return 0;
}
