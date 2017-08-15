/* Author: Edward Hutter */

// System includes
#include <iostream>
#include <mpi.h>

// Local includes
#include "Summa3D.h"

using namespace std;

int main(int argc, char** argv)
{
  using MatrixTypeA = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;
  using MatrixTypeB = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;
  using MatrixTypeC = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MatrixTypeA matA(4,4,4,4);
  MatrixTypeB matB(4,4,4,4);
  MatrixTypeC matC(4,4,4,4);

  Summa3D<double,int,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare>::Multiply(matA, matB, matC, 4, 4, 4, MPI_COMM_WORLD);
  return 0;
}
