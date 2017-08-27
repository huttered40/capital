/* Author: Edward Hutter */

// System includes
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

// Local includes
#include "CFR3D.h"

using namespace std;

int main(int argc, char** argv)
{
  using MatrixTypeA = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;
  using MatrixTypeL = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;

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
  
  cout << "global matrix size - " << globalMatrixSize << ", local Matrix size - " << localMatrixSize;
  cout << ", rank - " << rank << ", size - " << size << ", one dimension of the 3D grid's size - " << pGridDimensionSize << endl;

  MatrixTypeA matA(localMatrixSize,localMatrixSize,globalMatrixSize,globalMatrixSize);
  MatrixTypeL matL(localMatrixSize,localMatrixSize,globalMatrixSize,globalMatrixSize);
  MatrixTypeL matLI(localMatrixSize,localMatrixSize,globalMatrixSize,globalMatrixSize);

  int helper = pGridDimensionSize;
  helper *= helper;
  int pCoordX = rank%pGridDimensionSize;
  int pCoordY = (rank%helper)/pGridDimensionSize;
  int pCoordZ = rank/helper;

  matA.DistributeSymmetric(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, true);

  cout << "Processor " << rank << " has dimensions - (" << pCoordX << "," << pCoordY << "," << pCoordZ << ")\n";

  CFR3D<double,int,MatrixStructureSquare,MatrixStructureSquare>::
    Factor(matA, matL, matLI, localMatrixSize, MPI_COMM_WORLD);

  if (rank == 0)
  //matL.print();

  MPI_Finalize();

  return 0;
}
