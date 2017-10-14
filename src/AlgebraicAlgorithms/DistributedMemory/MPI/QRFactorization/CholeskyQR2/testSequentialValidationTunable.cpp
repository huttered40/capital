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

  // size -- total number of processors in the tunable grid
  int globalMatrixDimensionY = (1<<(atoi(argv[1])));
  int globalMatrixDimensionX = (1<<(atoi(argv[2])));

  int dimensionC,dimensionD;
  if (argc == 4)
  {
    // Use the grid that the user specifies in the command line
    dimensionD = (1<<(atoi(argv[3])));
    dimensionC = (1<<(atoi(argv[4])));
  }
  else
  {
    // Find the optimal grid based on the number of processors, size, and the dimensions of matrix A
    dimensionD = std::nearbyint(std::pow((size*globalMatrixDimensionY*globalMatrixDimensionY)/(globalMatrixDimensionX*globalMatrixDimensionX), 1./3));
    dimensionC = std::nearbyint(std::pow((size*globalMatrixDimensionX)/globalMatrixDimensionY, 1./3));
  }

  cout << "dimensionD - " << dimensionD << " and dimensionC - " << dimensionC << std::endl;

  int sliceSize = dimensionD*dimensionC;
  int pCoordX = rank%dimensionC;
  int pCoordY = (rank%sliceSize)/dimensionC;
  int pCoordZ = rank/sliceSize;

  int localMatrixDimensionY = globalMatrixDimensionY/dimensionD;
  int localMatrixDimensionX = globalMatrixDimensionX/dimensionC;

  // Note: matA and matR are rectangular, but the pieces owned by the individual processors may be square (so also rectangular)
  MatrixTypeR matA(localMatrixDimensionX,localMatrixDimensionY,globalMatrixDimensionX,globalMatrixDimensionY);
  MatrixTypeR matQ(localMatrixDimensionX,localMatrixDimensionY,globalMatrixDimensionX,globalMatrixDimensionY);
  MatrixTypeS matR(localMatrixDimensionX,localMatrixDimensionX,globalMatrixDimensionX,globalMatrixDimensionX);

  matA.DistributeRandom(pCoordX, pCoordY, dimensionC, dimensionD, (rank%sliceSize));

//  CholeskyQR2<double,int,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare,cblasEngine>::
//    FactorTunable(matA, matQ, matR, localMatrixSize, localMatrixSize, MPI_COMM_WORLD);

  MPI_Finalize();

  return 0;
}
