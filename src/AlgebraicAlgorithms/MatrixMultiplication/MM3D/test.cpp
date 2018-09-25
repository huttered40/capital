/* Author: Edward Hutter */

// System includes
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <string>

// Local includes
#include "./../../../Util/shared.h"
#include "MM3D.h"
#include "../../../Timer/CTFtimer.h"
#include "../MMvalidate/MMvalidate.h"

using namespace std;

// Idea: We calculate 3D Summa as usual, and then we pass it into the MMvalidate solo class

template<
		typename T, typename U,
		template<typename,typename, template<typename,typename,int> class> class StructureA,
  		template<typename,typename, template<typename,typename,int> class> class StructureB,
  		template<typename,typename, template<typename,typename,int> class> class StructureC = MatrixStructureSquare,
  		template<typename,typename,int> class Distribution
	>
static double runTestGemm(
                        Matrix<T,U,StructureA,Distribution>& matA,
                        Matrix<T,U,StructureB,Distribution>& matB,
                        Matrix<T,U,StructureC,Distribution>& matC,
			blasEngineArgumentPackage_gemm<T>& blasArgs,
			int methodKey3, int pCoordX, int pCoordY, int pGridDimensionSize,
			ofstream& fptrTotal, int iterNum, int numIter, int rank, int size, int& numFuncs
)
{
  double iterTimeGlobal;
  matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize + pCoordY);
  matB.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, (pCoordX*pGridDimensionSize + pCoordY)*(-1));
  matC.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, (pCoordX*pGridDimensionSize + pCoordY)*(-1));
  MPI_Barrier(MPI_COMM_WORLD);		// make sure each process starts together
  #ifdef CRITTER
  Critter_Clear();
  #endif
  TAU_FSTART(Total);
  #ifdef PERFORMANCE
  volatile double startTime=MPI_Wtime();
  #endif
  auto commInfo3D = util<T,U>::build3DTopology(MPI_COMM_WORLD);
  MM3D<T,U,cblasEngine>::Multiply(
    matA, matB, matC, MPI_COMM_WORLD, commInfo3D, blasArgs, methodKey3);
  util<T,U>::destroy3DTopology(commInfo3D);
  #ifdef PERFORMANCE
  volatile double iterTimeLocal=MPI_Wtime();
  iterTimeLocal -= startTime;
  MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0) { fptrTotal << size << "\t" << iterNum << "\t" << iterTimeGlobal << endl; }
  #endif
  TAU_FSTOP_FILE(Total, fptrTotal, iterNum, numFuncs);
  #ifdef CRITTER
  Critter_Print(fptrTotal, iterNum);
  #endif
  return iterTimeGlobal;
}

template<
		typename T, typename U,
		template<typename,typename, template<typename,typename,int> class> class StructureA,
  		template<typename,typename, template<typename,typename,int> class> class StructureB,
  		template<typename,typename,int> class Distribution
	>
static double runTestTrmm(
                        Matrix<T,U,StructureA,Distribution>& matA,
                        Matrix<T,U,StructureB,Distribution>& matB,
			blasEngineArgumentPackage_trmm<T>& blasArgs,
			int methodKey3, int pCoordX, int pCoordY, int pGridDimensionSize,
			ofstream& fptrTotal, int iterNum, int numIter, int rank, int size, int& numFuncs
)
{
  double iterTimeGlobal;
  matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize + pCoordY);
  matB.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, (pCoordX*pGridDimensionSize + pCoordY)*(-1));
  MPI_Barrier(MPI_COMM_WORLD);		// make sure each process starts together
  #ifdef CRITTER
  Critter_Clear();
  #endif
  TAU_FSTART(Total);
  #ifdef PERFORMANCE
  double startTime=MPI_Wtime();
  #endif
  auto commInfo3D = util<T,U>::build3DTopology(
    MPI_COMM_WORLD);
  MM3D<T,U,cblasEngine>::Multiply(
    matA, matB, MPI_COMM_WORLD, commInfo3D, blasArgs, methodKey3);
  util<T,U>::destroy3DTopology(commInfo3D);
  #ifdef PERFORMANCE
  volatile double iterTimeLocal=MPI_Wtime();
  iterTimeLocal -= startTime;
  MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0) { fptrTotal << size << "\t" << iterNum << "\t" << iterTimeGlobal << endl; }
  #endif
  TAU_FSTOP_FILE(Total, fptrTotal, iterNum, numFuncs);
  #ifdef CRITTER
  Critter_Print(fptrTotal, iterNum);
  #endif
  return iterTimeGlobal;
}


int main(int argc, char** argv)
{
  using MatrixTypeS = Matrix<DATATYPE,INTTYPE,MatrixStructureSquare,MatrixDistributerCyclic>;
  using MatrixTypeR = Matrix<DATATYPE,INTTYPE,MatrixStructureRectangle,MatrixDistributerCyclic>;
  using MatrixTypeLT = Matrix<DATATYPE,INTTYPE,MatrixStructureLowerTriangular,MatrixDistributerCyclic>;
  using MatrixTypeUT = Matrix<DATATYPE,INTTYPE,MatrixStructureUpperTriangular,MatrixDistributerCyclic>;

  #ifdef PROFILE
  TAU_PROFILE_SET_CONTEXT(0)
  #endif /*TIMER*/

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // size -- total number of processors in the 3D grid
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  util<DATATYPE,INTTYPE>::InitialGEMM();

  /*
    Choices for methodKey1: 0) Gemm
			    1) TRMM
  */
  int methodKey1 = atoi(argv[1]);
  /*
    Choices for methodKey3: 0) Broadcast + Allreduce
			    1) Allgather + Allreduce
  */
  int methodKey3 = atoi(argv[2]);

  int pGridDimensionSize = std::nearbyint(pow(size,1./3.));
  int helper = pGridDimensionSize;
  helper *= helper;
  #ifdef BLUEWATERS
  int pCoordZ = rank%pGridDimensionSize;
  int pCoordY = rank/helper;
  int pCoordX = (rank%helper)/pGridDimensionSize;
  #else
  int pCoordX = rank%pGridDimensionSize;
  int pCoordY = (rank%helper)/pGridDimensionSize;
  int pCoordZ = rank/helper;
  #endif

  INTTYPE globalMatrixSizeM = atoi(argv[3]);
  INTTYPE localMatrixSizeM = globalMatrixSizeM/pGridDimensionSize;
  INTTYPE globalMatrixSizeN = atoi(argv[4]);
  INTTYPE localMatrixSizeN = globalMatrixSizeN/pGridDimensionSize;

  if (methodKey1 == 0)
  {
    // GEMM
    INTTYPE globalMatrixSizeK = atoi(argv[5]);
    INTTYPE localMatrixSizeK = globalMatrixSizeK/pGridDimensionSize;
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
    if (rank == 0)
    {
      fptrTotal.open(fileStrTotal.c_str());
    }

    MatrixTypeR matA(globalMatrixSizeK,globalMatrixSizeM,pGridDimensionSize,pGridDimensionSize);
    MatrixTypeR matB(globalMatrixSizeN,globalMatrixSizeK,pGridDimensionSize,pGridDimensionSize);
    MatrixTypeR matC(globalMatrixSizeN,globalMatrixSizeM,pGridDimensionSize,pGridDimensionSize);
    blasEngineArgumentPackage_gemm<DATATYPE> blasArgs(blasEngineOrder::AblasColumnMajor, blasEngineTranspose::AblasNoTrans, blasEngineTranspose::AblasNoTrans, 1., 0.);
  
    // Loop for getting a good range of results.
    double totalTime = 0;
    int numFuncs = 0;
    for (int i=0; i<numIterations; i++)
    {
      double iterTime = runTestGemm(matA, matB, matC, blasArgs, methodKey3, pCoordX, pCoordY, pGridDimensionSize, fptrTotal, i, numIterations, rank, size, numFuncs);
      totalTime += iterTime;
    }
    if (rank == 0)
    {
      fptrTotal.close();
    }
  }
  else if (methodKey1 == 1)
  {
    // TRMM
    // First, we need to collect some special arguments.
    /*
      Choices for matrixUpLo: 0) Lower-triangular
			      1) Upper-triangular
    */
    int matrixUpLo = atoi(argv[5]);
    /*
      Choices for triangleSide: 0) Triangle * Rectangle (matrixA * matrixB)
			        1) Rectangle * Triangle (matrixB * matrixA)
    */
    int triangleSide = atoi(argv[6]);
    int numIterations = atoi(argv[7]);
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
    if (rank == 0)
    {
      fptrTotal.open(fileStrTotal.c_str());
    }

    // I guess I will go through all cases. Ugh!
    double totalTime = 0;
    int numFuncs = 0;
    if ((matrixUpLo == 0) && (triangleSide == 0))
    {
      MatrixTypeLT matA(globalMatrixSizeM,globalMatrixSizeM, pGridDimensionSize,pGridDimensionSize);
      MatrixTypeR matB(globalMatrixSizeN,globalMatrixSizeM, pGridDimensionSize,pGridDimensionSize);
      blasEngineArgumentPackage_trmm<DATATYPE> blasArgs(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasLeft, blasEngineUpLo::AblasLower,
        blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
 
      // Loop for getting a good range of results.
      for (int i=0; i<numIterations; i++)
      {
        double iterTime = runTestTrmm(matA, matB, blasArgs, methodKey3, pCoordX, pCoordY, pGridDimensionSize, fptrTotal, i, numIterations, rank, size, numFuncs);
        totalTime += iterTime;
      }
    }
    else if ((matrixUpLo == 0) && (triangleSide == 1))
    {
      MatrixTypeR matB(globalMatrixSizeN,globalMatrixSizeM, pGridDimensionSize,pGridDimensionSize);
      MatrixTypeLT matA(globalMatrixSizeN,globalMatrixSizeN, pGridDimensionSize,pGridDimensionSize);
      blasEngineArgumentPackage_trmm<DATATYPE> blasArgs(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasRight, blasEngineUpLo::AblasLower,
        blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);

      // Loop for getting a good range of results.
      for (int i=0; i<numIterations; i++)
      {
        double iterTime = runTestTrmm(matA, matB, blasArgs, methodKey3, pCoordX, pCoordY, pGridDimensionSize, fptrTotal, i, numIterations, rank, size, numFuncs);
        totalTime += iterTime;
      }
    }
    else if ((matrixUpLo == 1) && (triangleSide == 0))
    {
      MatrixTypeUT matA(globalMatrixSizeM,globalMatrixSizeM, pGridDimensionSize,pGridDimensionSize);
      MatrixTypeR matB(globalMatrixSizeN,globalMatrixSizeM, pGridDimensionSize,pGridDimensionSize);
      blasEngineArgumentPackage_trmm<DATATYPE> blasArgs(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasLeft, blasEngineUpLo::AblasUpper,
        blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
  
      // Loop for getting a good range of results.
      for (int i=0; i<numIterations; i++)
      {
        double iterTime = runTestTrmm(matA, matB, blasArgs, methodKey3, pCoordX, pCoordY, pGridDimensionSize, fptrTotal, i, numIterations, rank, size, numFuncs);
        totalTime += iterTime;
      }
    }
    else if ((matrixUpLo == 1) && (triangleSide == 1))
    {
      MatrixTypeR matB(globalMatrixSizeN,globalMatrixSizeM, pGridDimensionSize,pGridDimensionSize);
      MatrixTypeUT matA(globalMatrixSizeN,globalMatrixSizeN, pGridDimensionSize,pGridDimensionSize);
      blasEngineArgumentPackage_trmm<DATATYPE> blasArgs(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasRight, blasEngineUpLo::AblasUpper,
        blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);

      // Loop for getting a good range of results.
      for (int i=0; i<numIterations; i++)
      {
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
