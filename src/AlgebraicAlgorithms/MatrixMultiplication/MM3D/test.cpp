/* Author: Edward Hutter */

// System includes
#include <iostream>
#include <cstdlib>
#include <cmath>

// Local includes
#include "MM3D.h"
#include "../../../Timer/Timer.h"
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
static void runTestGemm(
                        Matrix<T,U,StructureA,Distribution>& matA,
                        Matrix<T,U,StructureB,Distribution>& matB,
                        Matrix<T,U,StructureC,Distribution>& matC,
			blasEngineArgumentPackage_gemm<T>& blasArgs,
			int methodKey2,
			int methodKey3,
			int pCoordX, int pCoordY, int pGridDimensionSize
)
{
#ifdef CRITTER
  Critter_Clear();
#endif
  TAU_FSTART(Total);
  auto commInfo3D = util<double,int>::build3DTopology(MPI_COMM_WORLD);
  MM3D<double,int,cblasEngine>::Multiply(
    matA, matB, matC, MPI_COMM_WORLD, commInfo3D, blasArgs, methodKey3);
  util<double,int>::destroy3DTopology(commInfo3D);
  TAU_FSTOP(Total);
#ifdef CRITTER
  Critter_Print();
#endif
  if (methodKey2 == 0)
  {
    // Sequential validation after 1 iteration, since numIterations == 1
    // Lets make sure matrixA and matrixB are set correctly by re-setting their values
    matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize + pCoordY);
    matB.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, (pCoordX*pGridDimensionSize + pCoordY)*(-1));
    MMvalidate<double,int,cblasEngine>::validateLocal(
      matA, matB, matC, MPI_COMM_WORLD, blasArgs);
  }
}

template<
		typename T, typename U,
		template<typename,typename, template<typename,typename,int> class> class StructureA,
  		template<typename,typename, template<typename,typename,int> class> class StructureB,
  		template<typename,typename,int> class Distribution
	>
static void runTestTrmm(
                        Matrix<T,U,StructureA,Distribution>& matA,
                        Matrix<T,U,StructureB,Distribution>& matB,
			blasEngineArgumentPackage_trmm<T>& blasArgs,
			int methodKey2,
			int methodKey3,
			int pCoordX, int pCoordY, int pGridDimensionSize
)
{
  Matrix<T,U,StructureB,Distribution> matBcopy = matB;
#ifdef CRITTER
  Critter_Clear();
#endif
  TAU_FSTART(Total);
  auto commInfo3D = util<double,int>::build3DTopology(
    MPI_COMM_WORLD);
  MM3D<double,int,cblasEngine>::Multiply(
    matA, matB, MPI_COMM_WORLD, commInfo3D, blasArgs, methodKey3);
  util<double,int>::destroy3DTopology(commInfo3D);
  TAU_FSTOP(Total);
#ifdef CRITTER
  Critter_Print();
#endif
  if (methodKey2 == 0)
  {
    // Sequential validation after 1 iteration, since numIterations == 1
    MMvalidate<double,int,cblasEngine>::validateLocal(
      matA, matBcopy, matB, MPI_COMM_WORLD, blasArgs);
  }
}


int main(int argc, char** argv)
{
  using MatrixTypeS = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;
  using MatrixTypeR = Matrix<double,int,MatrixStructureRectangle,MatrixDistributerCyclic>;
  using MatrixTypeLT = Matrix<double,int,MatrixStructureLowerTriangular,MatrixDistributerCyclic>;
  using MatrixTypeUT = Matrix<double,int,MatrixStructureUpperTriangular,MatrixDistributerCyclic>;

#ifdef PROFILE
  TAU_PROFILE_SET_CONTEXT(0)
#endif /*TIMER*/

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // size -- total number of processors in the 3D grid
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /*
    Choices for methodKey1: 0) Gemm
			    1) TRMM
  */
  int methodKey1 = atoi(argv[1]);
  /*
    Choices for methodKey2: 0) Sequential validation
			    1) Performance testing
  */
  int methodKey2 = atoi(argv[2]);
  /*
    Choices for methodKey3: 0) Broadcast + Allreduce
			    1) Allgather + Allreduce
  */
  int methodKey3 = atoi(argv[3]);

  int pGridDimensionSize = std::nearbyint(pow(size,1./3.));
  int helper = pGridDimensionSize;
  helper *= helper;
  int pCoordX = rank%pGridDimensionSize;
  int pCoordY = (rank%helper)/pGridDimensionSize;
  int pCoordZ = rank/helper;

  uint64_t globalMatrixSizeM = atoi(argv[4]);
  uint64_t localMatrixSizeM = globalMatrixSizeM/pGridDimensionSize;
  uint64_t globalMatrixSizeN = atoi(argv[5]);
  uint64_t localMatrixSizeN = globalMatrixSizeN/pGridDimensionSize;

  pTimer myTimer;
  if (methodKey1 == 0)
  {
    // GEMM
    uint64_t globalMatrixSizeK = atoi(argv[6]);
    uint64_t localMatrixSizeK = globalMatrixSizeK/pGridDimensionSize;
    int numIterations = atoi(argv[7]);

    MatrixTypeR matA(globalMatrixSizeK,globalMatrixSizeM,pGridDimensionSize,pGridDimensionSize);
    MatrixTypeR matB(globalMatrixSizeN,globalMatrixSizeK,pGridDimensionSize,pGridDimensionSize);
    MatrixTypeR matC(globalMatrixSizeN,globalMatrixSizeM,pGridDimensionSize,pGridDimensionSize);
    blasEngineArgumentPackage_gemm<double> blasArgs(blasEngineOrder::AblasColumnMajor, blasEngineTranspose::AblasNoTrans, blasEngineTranspose::AblasNoTrans, 1., 0.);
  
    // Loop for getting a good range of results.
    for (int i=0; i<numIterations; i++)
    {
      // Don't use rank. Need to use the rank relative to the slice its on, since each slice will start off with the same matrix
      matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize + pCoordY);
      matB.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, (pCoordX*pGridDimensionSize + pCoordY)*(-1));
      matC.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, (pCoordX*pGridDimensionSize + pCoordY)*(-1));
      runTestGemm(matA, matB, matC, blasArgs, methodKey2, methodKey3, pCoordX, pCoordY, pGridDimensionSize);
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
    int matrixUpLo = atoi(argv[6]);
    /*
      Choices for triangleSide: 0) Triangle * Rectangle (matrixA * matrixB)
			        1) Rectangle * Triangle (matrixB * matrixA)
    */
    int triangleSide = atoi(argv[7]);
    int numIterations = atoi(argv[8]);

    // I guess I will go through all cases. Ugh!
    if ((matrixUpLo == 0) && (triangleSide == 0))
    {
      MatrixTypeLT matA(globalMatrixSizeM,globalMatrixSizeM, pGridDimensionSize,pGridDimensionSize);
      MatrixTypeR matB(globalMatrixSizeN,globalMatrixSizeM, pGridDimensionSize,pGridDimensionSize);
      blasEngineArgumentPackage_trmm<double> blasArgs(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasLeft, blasEngineUpLo::AblasLower,
        blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
 
      // Loop for getting a good range of results.
      for (int i=0; i<numIterations; i++)
      {
        matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize + pCoordY);
        matB.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, (pCoordX*pGridDimensionSize + pCoordY)*(-1));
        runTestTrmm(matA, matB, blasArgs, methodKey2, methodKey3, pCoordX, pCoordY, pGridDimensionSize);
      }
    }
    else if ((matrixUpLo == 0) && (triangleSide == 1))
    {
      MatrixTypeR matB(globalMatrixSizeN,globalMatrixSizeM, pGridDimensionSize,pGridDimensionSize);
      MatrixTypeLT matA(globalMatrixSizeN,globalMatrixSizeN, pGridDimensionSize,pGridDimensionSize);
      blasEngineArgumentPackage_trmm<double> blasArgs(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasRight, blasEngineUpLo::AblasLower,
        blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);

      // Loop for getting a good range of results.
      for (int i=0; i<numIterations; i++)
      {
        matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize + pCoordY);
        matB.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, (pCoordX*pGridDimensionSize + pCoordY)*(-1));
        runTestTrmm(matA, matB, blasArgs, methodKey2, methodKey3, pCoordX, pCoordY, pGridDimensionSize);
      }
    }
    else if ((matrixUpLo == 1) && (triangleSide == 0))
    {
      MatrixTypeUT matA(globalMatrixSizeM,globalMatrixSizeM, pGridDimensionSize,pGridDimensionSize);
      MatrixTypeR matB(globalMatrixSizeN,globalMatrixSizeM, pGridDimensionSize,pGridDimensionSize);
      blasEngineArgumentPackage_trmm<double> blasArgs(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasLeft, blasEngineUpLo::AblasUpper,
        blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
  
      // Loop for getting a good range of results.
      for (int i=0; i<numIterations; i++)
      {
        matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize + pCoordY);
        matB.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, (pCoordX*pGridDimensionSize + pCoordY)*(-1));
        runTestTrmm(matA, matB, blasArgs, methodKey2, methodKey3, pCoordX, pCoordY, pGridDimensionSize);
      }
    }
    else if ((matrixUpLo == 1) && (triangleSide == 1))
    {
      MatrixTypeR matB(globalMatrixSizeN,globalMatrixSizeM, pGridDimensionSize,pGridDimensionSize);
      MatrixTypeUT matA(globalMatrixSizeN,globalMatrixSizeN, pGridDimensionSize,pGridDimensionSize);
      blasEngineArgumentPackage_trmm<double> blasArgs(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasRight, blasEngineUpLo::AblasUpper,
        blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);

      // Loop for getting a good range of results.
      for (int i=0; i<numIterations; i++)
      {
        matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize + pCoordY);
        matB.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, (pCoordX*pGridDimensionSize + pCoordY)*(-1));
        runTestTrmm(matA, matB, blasArgs, methodKey2, methodKey3, pCoordX, pCoordY, pGridDimensionSize);
      }
    }
  }

  MPI_Finalize();
  return 0;
}
