/* Author: Edward Hutter */

// System includes
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

// Local includes
#include "CholeskyQR2.h"

using namespace std;

// Idea: We calculate 3D Summa as usual, and then we pass it into the MMvalidate solo class

int main(int argc, char** argv)
{
  using MatrixTypeS = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;
  using MatrixTypeR = Matrix<double,int,MatrixStructureRectangle,MatrixDistributerCyclic>;
  using MatrixTypeLT = Matrix<double,int,MatrixStructureLowerTriangular,MatrixDistributerCyclic>;
  using MatrixTypeUT = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;

  // argv[1] - Matrix size x where x represents 2^x.
  // So in future, we might want t way to test non power of 2 dimension matrices

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // size -- total number of processors in the 3D grid

  int pGridDimensionSize = ceil(pow(size,1./3.));
  uint64_t globalMatrixSize = (1<<(atoi(argv[1])));
  uint64_t localMatrixSize = globalMatrixSize/pGridDimensionSize;
  
  MatrixTypeS matA(localMatrixSize,localMatrixSize,globalMatrixSize,globalMatrixSize);
  MatrixTypeS matQ(localMatrixSize,localMatrixSize,globalMatrixSize,globalMatrixSize);
  MatrixTypeS matR(localMatrixSize,localMatrixSize,globalMatrixSize,globalMatrixSize);

  int helper = pGridDimensionSize;
  helper *= helper;
  int pCoordX = rank%pGridDimensionSize;
  int pCoordY = (rank%helper)/pGridDimensionSize;
  int pCoordZ = rank/helper;

  matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize);

  CholeskyQR2<double,int,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare,cblasEngine>::
    FactorTunable(matA, matQ, matR, localMatrixSize, localMatrixSize, MPI_COMM_WORLD);

  MPI_Finalize();

  return 0;
}
