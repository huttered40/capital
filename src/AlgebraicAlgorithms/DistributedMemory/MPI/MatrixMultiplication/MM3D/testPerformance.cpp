/* Author: Edward Hutter */

// System includes
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

// Local includes
#include "../../../../../Timer/Timer.h"
#include "MM3D.h"

using namespace std;

int main(int argc, char** argv)
{
  using MatrixTypeS = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;
  using MatrixTypeR = Matrix<double,int,MatrixStructureRectangle,MatrixDistributerCyclic>;

  // argv[1] - Matrix size x where x represents 2^x.
  // So in future, we might want t way to test non power of 2 dimension matrices

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // size -- total number of processors in the 3D grid

  int pGridDimensionSize = std::nearbyint(std::pow(size,1./3.));
  uint64_t globalMatrixSizeM = (1<<(atoi(argv[1])));
  uint64_t localMatrixSizeM = globalMatrixSizeM/pGridDimensionSize;
  uint64_t globalMatrixSizeN = (1<<(atoi(argv[2])));
  uint64_t localMatrixSizeN = globalMatrixSizeN/pGridDimensionSize;
  uint64_t globalMatrixSizeK = (1<<(atoi(argv[3])));
  uint64_t localMatrixSizeK = globalMatrixSizeK/pGridDimensionSize;
  
  MatrixTypeR matA(localMatrixSizeK,localMatrixSizeM,globalMatrixSizeK,globalMatrixSizeM);
  MatrixTypeR matB(localMatrixSizeN,localMatrixSizeK,globalMatrixSizeN,globalMatrixSizeK);
  MatrixTypeR matC(localMatrixSizeN,localMatrixSizeM,globalMatrixSizeN,globalMatrixSizeM);

  int helper = pGridDimensionSize;
  helper *= helper;
  int pCoordX = rank%pGridDimensionSize;
  int pCoordY = (rank%helper)/pGridDimensionSize;
  int pCoordZ = rank/helper;

  matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize + pCoordY);
  matB.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, (pCoordX*pGridDimensionSize + pCoordY)*(-1));

  blasEngineArgumentPackage_gemm<double> blasArgs;
  blasArgs.order = blasEngineOrder::AblasColumnMajor;
  blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
  blasArgs.transposeB = blasEngineTranspose::AblasNoTrans;
  blasArgs.alpha = 1.;
  blasArgs.beta = 0.;

  pTimer myTimer;

  // Loop for getting a good range of results.
  for (int i=0; i<10; i++)
  {
    myTimer.setStartTime();
    MM3D<double,int,MatrixStructureRectangle,MatrixStructureRectangle,MatrixStructureRectangle,cblasEngine>::
      Multiply(matA, matB, matC, localMatrixSizeM, localMatrixSizeN, localMatrixSizeK, MPI_COMM_WORLD, blasArgs);
    myTimer.setEndTime();
    myTimer.printParallelTime(1e-8, MPI_COMM_WORLD, "MM3D iteration", i);
  }

  MPI_Finalize();

  return 0;
}
