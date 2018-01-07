/* Author: Edward Hutter */

// System includes
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

// Local includes
#include "../../../Timer/Timer.h"
#include "MM3D.h"

using namespace std;

// Idea: We calculate 3D Summa as usual, and then we pass it into the MMvalidate solo class

int main(int argc, char** argv)
{
  using MatrixTypeS = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;
  using MatrixTypeR = Matrix<double,int,MatrixStructureRectangle,MatrixDistributerCyclic>;
  using MatrixTypeLT = Matrix<double,int,MatrixStructureLowerTriangular,MatrixDistributerCyclic>;
  using MatrixTypeUT = Matrix<double,int,MatrixStructureUpperTriangular,MatrixDistributerCyclic>;

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // size -- total number of processors in the 3D grid
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /*
    Choices for methodKey1: 0) Gemm
			    1) TRMM
                            2) SYRK
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

  /*
    Choices for dimensionKey: 0) Non-power of 2
			      1) Power of 2
  */
  int dimensionKey = atoi(argv[4]);
  uint64_t globalMatrixSizeM = (dimensionKey == 0 ? atoi(argv[5]) : (1<<(atoi(argv[5]))));
  uint64_t localMatrixSizeM = globalMatrixSizeM/pGridDimensionSize;
  uint64_t globalMatrixSizeN = (dimensionKey == 0 ? atoi(argv[6]) : (1<<(atoi(argv[6]))));
  uint64_t localMatrixSizeN = globalMatrixSizeN/pGridDimensionSize;

  pTimer myTimer;
  if (methodKey1 == 0)
  {
    // GEMM
    uint64_t globalMatrixSizeK = (dimensionKey == 0 ? atoi(argv[7]) : (1<<(atoi(argv[7]))));
    uint64_t localMatrixSizeK = globalMatrixSizeK/pGridDimensionSize;

    MatrixTypeR matA(localMatrixSizeK,localMatrixSizeM,globalMatrixSizeK,globalMatrixSizeM);
    MatrixTypeR matB(localMatrixSizeN,localMatrixSizeK,globalMatrixSizeN,globalMatrixSizeK);
    MatrixTypeR matC(localMatrixSizeN,localMatrixSizeM,globalMatrixSizeN,globalMatrixSizeM);

    // Don't use rank. Need to use the rank relative to the slice its on, since each slice will start off with the same matrix
    matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize + pCoordY);
    matB.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, (pCoordX*pGridDimensionSize + pCoordY)*(-1));

    blasEngineArgumentPackage_gemm<double> blasArgs;
    blasArgs.order = blasEngineOrder::AblasColumnMajor;
    blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
    blasArgs.transposeB = blasEngineTranspose::AblasNoTrans;
    blasArgs.alpha = 1.;
    blasArgs.beta = 0;
  
    int numIterations = (methodKey2 == 0 ? 1 : atoi(argv[8]));
    // Loop for getting a good range of results.
    for (int i=0; i<numIterations; i++)
    {
      myTimer.setStartTime();
      MM3D<double,int,cblasEngine>::
        Multiply(matA, matB, matC, localMatrixSizeM, localMatrixSizeN, localMatrixSizeK, MPI_COMM_WORLD, blasArgs, methodKey3);
      myTimer.setEndTime();
      myTimer.printParallelTime(1e-8, MPI_COMM_WORLD, "MM3D GEMM iteration", i);
      MPI_Barrier(MPI_COMM_WORLD);
    }
    if (methodKey2 == 0)
    {
      // Sequential validation after 1 iteration, since numIterations == 1
      MMvalidate<double,int,cblasEngine>::validateLocal(matA, matB, matC, localMatrixSizeM, localMatrixSizeN, localMatrixSizeK,
        globalMatrixSizeM, globalMatrixSizeN, globalMatrixSizeK, MPI_COMM_WORLD, blasArgs);
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
    int matrixUpLo = atoi(argv[7]);
    /*
      Choices for triangleSide: 0) Triangle * Rectangle (matrixA * matrixB)
			        1) Rectangle * Triangle (matrixB * matrixA)
    */
    int triangleSide = atoi(argv[8]);

    blasEngineArgumentPackage_trmm<double> blasArgs;
    blasArgs.order = blasEngineOrder::AblasColumnMajor;
    blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
    blasArgs.diag =blasEngineDiag::AblasNonUnit;
    blasArgs.alpha = 1.;

    // I guess I will go through all cases. Ugh!
    if ((matrixUpLo == 0) && (triangleSide == 0))
    {
      MatrixTypeLT matA(localMatrixSizeM,localMatrixSizeM,globalMatrixSizeM,globalMatrixSizeM);
      MatrixTypeR matB(localMatrixSizeN,localMatrixSizeM,globalMatrixSizeN,globalMatrixSizeM);

      matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize + pCoordY);
      matB.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, (pCoordX*pGridDimensionSize + pCoordY)*(-1));

      blasArgs.side = blasEngineSide::AblasLeft;
      blasArgs.uplo = blasEngineUpLo::AblasLower;
 
      // Make a copy of matrixB before it gets overwritten by MM3D. This won't hurt performance numbers of anything
      MatrixTypeR matBcopy = matB;
 
      int numIterations = (methodKey2 == 0 ? 1 : atoi(argv[9]));
      // Loop for getting a good range of results.
      for (int i=0; i<numIterations; i++)
      {
        myTimer.setStartTime();
        MM3D<double,int,cblasEngine>::
          Multiply(matA, matB, localMatrixSizeM, localMatrixSizeN, MPI_COMM_WORLD, blasArgs, methodKey3);
        myTimer.setEndTime();
        myTimer.printParallelTime(1e-8, MPI_COMM_WORLD, "MM3D TRMM iteration", i);
        MPI_Barrier(MPI_COMM_WORLD);
      }
      if (methodKey2 == 0)
      {
        // Sequential validation after 1 iteration, since numIterations == 1
        MMvalidate<double,int,cblasEngine>::validateLocal(matA, matBcopy, matB, localMatrixSizeM, localMatrixSizeN, globalMatrixSizeM, globalMatrixSizeN, MPI_COMM_WORLD, blasArgs);
      }
    }
    else if ((matrixUpLo == 0) && (triangleSide == 1))
    {
      MatrixTypeR matB(localMatrixSizeN,localMatrixSizeM,globalMatrixSizeN,globalMatrixSizeM);
      MatrixTypeLT matA(localMatrixSizeN,localMatrixSizeN,globalMatrixSizeN,globalMatrixSizeN);

      matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize + pCoordY);
      matB.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, (pCoordX*pGridDimensionSize + pCoordY)*(-1));

      blasArgs.side = blasEngineSide::AblasRight;
      blasArgs.uplo = blasEngineUpLo::AblasLower;

      // Make a copy of matrixB before it gets overwritten by MM3D. This won't hurt performance numbers of anything
      MatrixTypeR matBcopy = matB;
  
      int numIterations = (methodKey2 == 0 ? 1 : atoi(argv[9]));
      // Loop for getting a good range of results.
      for (int i=0; i<numIterations; i++)
      {
        myTimer.setStartTime();
        MM3D<double,int,cblasEngine>::
          Multiply(matA, matB, localMatrixSizeM, localMatrixSizeN, MPI_COMM_WORLD, blasArgs, methodKey3);
        myTimer.setEndTime();
        myTimer.printParallelTime(1e-8, MPI_COMM_WORLD, "MM3D TRMM iteration", i);
        MPI_Barrier(MPI_COMM_WORLD);
      }
      if (methodKey2 == 0)
      {
        // Sequential validation after 1 iteration, since numIterations == 1
        MMvalidate<double,int,cblasEngine>::validateLocal(matA, matBcopy, matB, localMatrixSizeM, localMatrixSizeN, globalMatrixSizeM, globalMatrixSizeN, MPI_COMM_WORLD, blasArgs);
      }
    }
    else if ((matrixUpLo == 1) && (triangleSide == 0))
    {
      MatrixTypeUT matA(localMatrixSizeM,localMatrixSizeM,globalMatrixSizeM,globalMatrixSizeM);
      MatrixTypeR matB(localMatrixSizeN,localMatrixSizeM,globalMatrixSizeN,globalMatrixSizeM);

      matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize + pCoordY);
      matB.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, (pCoordX*pGridDimensionSize + pCoordY)*(-1));

      blasArgs.side = blasEngineSide::AblasLeft;
      blasArgs.uplo = blasEngineUpLo::AblasUpper;
  
      // Make a copy of matrixB before it gets overwritten by MM3D. This won't hurt performance numbers of anything
      MatrixTypeR matBcopy = matB;

      int numIterations = (methodKey2 == 0 ? 1 : atoi(argv[9]));
      // Loop for getting a good range of results.
      for (int i=0; i<numIterations; i++)
      {
        myTimer.setStartTime();
        MM3D<double,int,cblasEngine>::
          Multiply(matA, matB, localMatrixSizeM, localMatrixSizeN, MPI_COMM_WORLD, blasArgs, methodKey3);
        myTimer.setEndTime();
        myTimer.printParallelTime(1e-8, MPI_COMM_WORLD, "MM3D TRMM iteration", i);
        MPI_Barrier(MPI_COMM_WORLD);
      }
      if (methodKey2 == 0)
      {
        // Sequential validation after 1 iteration, since numIterations == 1
        MMvalidate<double,int,cblasEngine>::validateLocal(matA, matBcopy, matB, localMatrixSizeM, localMatrixSizeN, globalMatrixSizeM, globalMatrixSizeN, MPI_COMM_WORLD, blasArgs);
      }

    }
    else if ((matrixUpLo == 1) && (triangleSide == 1))
    {
      MatrixTypeR matB(localMatrixSizeN,localMatrixSizeM,globalMatrixSizeN,globalMatrixSizeM);
      MatrixTypeUT matA(localMatrixSizeN,localMatrixSizeN,globalMatrixSizeN,globalMatrixSizeN);

      matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize + pCoordY);
      matB.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, (pCoordX*pGridDimensionSize + pCoordY)*(-1));

      blasArgs.side = blasEngineSide::AblasRight;
      blasArgs.uplo = blasEngineUpLo::AblasUpper;

      // Make a copy of matrixB before it gets overwritten by MM3D. This won't hurt performance numbers of anything
      MatrixTypeR matBcopy = matB;
  
      int numIterations = (methodKey2 == 0 ? 1 : atoi(argv[9]));
      // Loop for getting a good range of results.
      for (int i=0; i<numIterations; i++)
      {
        myTimer.setStartTime();
        MM3D<double,int,cblasEngine>::
          Multiply(matA, matB, localMatrixSizeM, localMatrixSizeN, MPI_COMM_WORLD, blasArgs, methodKey3);
        myTimer.setEndTime();
        myTimer.printParallelTime(1e-8, MPI_COMM_WORLD, "MM3D TRMM iteration", i);
        MPI_Barrier(MPI_COMM_WORLD);
      }
      if (methodKey2 == 0)
      {
        // Sequential validation after 1 iteration, since numIterations == 1
        MMvalidate<double,int,cblasEngine>::validateLocal(matA, matBcopy, matB, localMatrixSizeM, localMatrixSizeN, globalMatrixSizeM, globalMatrixSizeN, MPI_COMM_WORLD, blasArgs);
      }

    }
    else
    {
      cout << "Bad input for TRMM\n";
      MPI_Abort(MPI_COMM_WORLD,0);
    }
  }
  else
  {
/*
    // SYRK
    //
      Choices for matrixAtranspose: 0) NoTrans
			      	    1) Trans
    //
    int matrixAtranspose = atoi(argv[6]);
    blasEngineArgumentPackage_syrk<double> blasArgs;
    blasArgs.order = blasEngineOrder::AblasColumnMajor;
    blasArgs.uplo = blasEngineUpLo::AblasUpper;			// Lets only use the Upper for testing
    blasArgs.alpha = 1.;
    blasArgs.beta = 0;
    MatrixTypeR matC(localMatrixSizeM,localMatrixSizeM,globalMatrixSizeM,globalMatrixSizeM);

    if (matrixAtranspose == 0)
    {
      MatrixTypeR matA(localMatrixSizeN,localMatrixSizeM,globalMatrixSizeN,globalMatrixSizeM);
      matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize + pCoordY);
      blasArgs.transposeA = blasEngineTranspose::AblasTrans;
    }
    else
    {
      MatrixTypeR matA(localMatrixSizeM,localMatrixSizeN,globalMatrixSizeM,globalMatrixSizeN);
      matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize + pCoordY);
      blasArgs.transposeA = blasEngineTranspose::AblasTrans;
    }

    pTimer myTimer;
    MM3D<double,int,cblasEngine>::
      Multiply(matA, matC, localMatrixSizeN, localMatrixSizeK, MPI_COMM_WORLD, blasArgs);
    MPI_Barrier(MPI_COMM_WORLD);
    MMvalidate<double,int,cblasEngine>::validateLocal(matC, localMatrixSizeN, localMatrixSizeK,
      globalMatrixSizeN, globalMatrixSizeK, MPI_COMM_WORLD, blasArgs);
*/
  }

  MPI_Finalize();
  return 0;
}
