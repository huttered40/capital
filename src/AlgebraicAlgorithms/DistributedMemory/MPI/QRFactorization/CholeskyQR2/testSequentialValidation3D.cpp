/* Author: Edward Hutter */

// System includes
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

// Local includes
#include "CholeskyQR2.h"
#include "./../QRvalidate/QRvalidate.h"

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
  int pGridDimensionSize = ceil(pow(size,1./3.));
  // size -- total number of processors in the 3D grid

  int helper = pGridDimensionSize;
  helper *= helper;
  int pCoordX = rank%pGridDimensionSize;
  int pCoordY = (rank%helper)/pGridDimensionSize;
  int pCoordZ = rank/helper;

  uint64_t globalMatrixDimensionY = (1<<(atoi(argv[1])));
  uint64_t globalMatrixDimensionX = (1<<(atoi(argv[2])));
  uint64_t localMatrixDimensionY = globalMatrixDimensionY/pGridDimensionSize;
  uint64_t localMatrixDimensionX = globalMatrixDimensionX/pGridDimensionSize;

  // New protocol: CholeskyQR_3D only works properly with square matrix A. Rectangular matrices must use CholeskyQR_Tunable
  MatrixTypeS matA(localMatrixDimensionX,localMatrixDimensionY,globalMatrixDimensionX,globalMatrixDimensionY);
  MatrixTypeS matQ(localMatrixDimensionX,localMatrixDimensionY,globalMatrixDimensionX,globalMatrixDimensionY);
  MatrixTypeS matR(localMatrixDimensionX,localMatrixDimensionX,globalMatrixDimensionX,globalMatrixDimensionX);

  matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize+pCoordY);

  cout << "Rank " << rank << " has local dimensionX - " << localMatrixDimensionX << ", localDimensionY - " << localMatrixDimensionY << endl;

  CholeskyQR2<double,int,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare,cblasEngine>::
    Factor3D(matA, matQ, matR, globalMatrixDimensionX, globalMatrixDimensionY, MPI_COMM_WORLD);

  if (rank == 1)
  {
    //matQ.print();
  }

  QRvalidate<double,int>::validateLocal3D(matA, matQ, matR, globalMatrixDimensionX, globalMatrixDimensionY, MPI_COMM_WORLD);

  MPI_Finalize();

  return 0;
}
