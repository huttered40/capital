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

  // argv[1] - Rectangular matrix dimension y (numRows)
  // argv[2] - Rectangular matrix dimension x (numColumns)

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // size -- total number of processors in the 3D grid

  uint64_t globalMatrixDimensionY = (1<<(atoi(argv[1])));
  uint64_t globalMatrixDimensionX = (1<<(atoi(argv[2])));
  uint64_t localMatrixDimensionY = globalMatrixDimensionY/size;
  uint64_t localMatrixDimensionX = globalMatrixDimensionX;
  
  MatrixTypeR matA(localMatrixDimensionX,localMatrixDimensionY,globalMatrixDimensionX,globalMatrixDimensionY);
  MatrixTypeR matQ(localMatrixDimensionX,localMatrixDimensionY,globalMatrixDimensionX,globalMatrixDimensionY);
  MatrixTypeS matR(localMatrixDimensionX,localMatrixDimensionX,globalMatrixDimensionX,globalMatrixDimensionX);

  matA.DistributeRandom(0, rank, 1, size);

  cout << "Rank " << rank << " has local dimensionX - " << localMatrixDimensionX << ", localDimensionY - " << localMatrixDimensionY << endl;

  CholeskyQR2<double,int,MatrixStructureRectangle,MatrixStructureRectangle,MatrixStructureSquare,cblasEngine>::
    Factor1D(matA, matQ, matR, globalMatrixDimensionX, globalMatrixDimensionY, MPI_COMM_WORLD);

//  double error = MMvalidate<double,int,cblasEngine>::validateLocal(matC, localMatrixSize, localMatrixSize, localMatrixSize,
//    globalMatrixSize, globalMatrixSize, globalMatrixSize, MPI_COMM_WORLD, blasArgs);

//  std::cout << "Error for procesor " << rank << " = " << error << std::endl;

  MPI_Finalize();

  return 0;
}
