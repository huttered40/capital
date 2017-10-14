/* Author: Edward Hutter */

// System includes
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

// Local includes
#include "../../../../../Timer/Timer.h"
#include "SquareMM3D.h"

using namespace std;

int main(int argc, char** argv)
{
  using MatrixTypeA = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;
  using MatrixTypeB = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;
  using MatrixTypeC = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;

  // argv[1] - Matrix size x where x represents 2^x.
  // So in future, we might want t way to test non power of 2 dimension matrices

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // size -- total number of processors in the 3D grid

  int pGridDimensionSize = std::nearbyint(std::pow(size,1./3.));
  uint64_t globalMatrixSize = (1<<(atoi(argv[1])));
  uint64_t localMatrixSize = globalMatrixSize/pGridDimensionSize;
  
  cout << "global matrix size - " << globalMatrixSize << ", local Matrix size - " << localMatrixSize;
  cout << ", rank - " << rank << ", size - " << size << ", one dimension of the 3D grid's size - " << pGridDimensionSize << endl;

  MatrixTypeA matA(localMatrixSize,localMatrixSize,globalMatrixSize,globalMatrixSize);
  MatrixTypeB matB(localMatrixSize,localMatrixSize,globalMatrixSize,globalMatrixSize);
  MatrixTypeC matC(localMatrixSize,localMatrixSize,globalMatrixSize,globalMatrixSize);

  int helper = pGridDimensionSize;
  helper *= helper;
  int pCoordX = rank%pGridDimensionSize;
  int pCoordY = (rank%helper)/pGridDimensionSize;
  int pCoordZ = rank/helper;

  matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize + pCoordY);
  matB.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, (pCoordX*pGridDimensionSize + pCoordY)*(-1));

  blasEngineArgumentPackage_gemm<double> blasArgs;
  blasArgs.order = blasEngineOrder::AblasRowMajor;
  blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
  blasArgs.transposeB = blasEngineTranspose::AblasNoTrans;
  blasArgs.alpha = 1.;
  blasArgs.beta = 0.;

  pTimer myTimer;

  // Loop for getting a good range of results.
  for (int i=0; i<10; i++)
  {
    myTimer.setStartTime();
    SquareMM3D<double,int,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare, cblasEngine>::
      Multiply(matA, matB, matC, localMatrixSize, localMatrixSize, localMatrixSize, MPI_COMM_WORLD, blasArgs);
    myTimer.setEndTime();
    myTimer.printParallelTime(1e-8, MPI_COMM_WORLD, "MM3D iteration", i);
  }

  MPI_Finalize();

  return 0;
}
