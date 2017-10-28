/* Author: Edward Hutter */

// System includes
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

// Local includes
#include "MM3D.h"

using namespace std;

// Idea: We calculate 3D Summa as usual, and then we pass it into the MMvalidate solo class

int main(int argc, char** argv)
{
  using MatrixTypeS = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;
  using MatrixTypeLT = Matrix<double,int,MatrixStructureLowerTriangular,MatrixDistributerCyclic>;
  using MatrixTypeUT = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;

  // argv[1] - Matrix size x where x represents 2^x.
  // So in future, we might want t way to test non power of 2 dimension matrices

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // size -- total number of processors in the 3D grid

  int pGridDimensionSize = std::nearbyint(std::pow(size,1./3.));
  uint64_t globalMatrixSizeN = (1<<(atoi(argv[1])));
  uint64_t localMatrixSizeN = globalMatrixSizeN/pGridDimensionSize;
  uint64_t globalMatrixSizeK = (1<<(atoi(argv[2])));
  uint64_t localMatrixSizeK = globalMatrixSizeK/pGridDimensionSize;
  
  MatrixTypeS matA(localMatrixSizeK,localMatrixSizeN,globalMatrixSizeN,globalMatrixSizeN);
  MatrixTypeS matC(localMatrixSizeN,localMatrixSizeN,globalMatrixSizeN,globalMatrixSizeN);

  int helper = pGridDimensionSize;
  helper *= helper;
  int pCoordX = rank%pGridDimensionSize;
  int pCoordY = (rank%helper)/pGridDimensionSize;
  int pCoordZ = rank/helper;

  matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize + pCoordY);

  blasEngineArgumentPackage_syrk<double> blasArgs;
  blasArgs.order = blasEngineOrder::AblasColumnMajor;
  blasArgs.uplo = blasEngineUpLo::AblasUpper;
  blasArgs.transposeA = blasEngineTranspose::AblasTrans;
  blasArgs.alpha = 1.;
  blasArgs.beta = 0;

  MM3D<double,int,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare, cblasEngine>::
    Multiply(matA, matC, localMatrixSizeN, localMatrixSizeK, MPI_COMM_WORLD, blasArgs);

  MPI_Barrier(MPI_COMM_WORLD);

  MMvalidate<double,int,cblasEngine>::validateLocal(matC, localMatrixSizeN, localMatrixSizeK,
    globalMatrixSizeN, globalMatrixSizeK, MPI_COMM_WORLD, blasArgs);

  MPI_Finalize();

  return 0;
}
