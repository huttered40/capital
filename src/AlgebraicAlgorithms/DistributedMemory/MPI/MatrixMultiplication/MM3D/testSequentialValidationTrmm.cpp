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
  uint64_t globalMatrixSizeM = (1<<(atoi(argv[1])));
  uint64_t localMatrixSizeM = globalMatrixSizeM/pGridDimensionSize;
  uint64_t globalMatrixSizeN = (1<<(atoi(argv[2])));
  uint64_t localMatrixSizeN = globalMatrixSizeN/pGridDimensionSize;
  
  MatrixTypeLT matA(localMatrixSizeN,localMatrixSizeN,globalMatrixSizeN,globalMatrixSizeN);
  MatrixTypeS matB(localMatrixSizeN,localMatrixSizeM,globalMatrixSizeN,globalMatrixSizeM);

  int helper = pGridDimensionSize;
  helper *= helper;
  int pCoordX = rank%pGridDimensionSize;
  int pCoordY = (rank%helper)/pGridDimensionSize;
  int pCoordZ = rank/helper;

  matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize + pCoordY);
  matB.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, (pCoordX*pGridDimensionSize + pCoordY)*(-1));

  blasEngineArgumentPackage_trmm<double> blasArgs;
  blasArgs.order = blasEngineOrder::AblasColumnMajor;
  blasArgs.side = blasEngineSide::AblasRight;
  blasArgs.uplo = blasEngineUpLo::AblasLower;
  blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
  blasArgs.diag =blasEngineDiag::AblasNonUnit;
  blasArgs.alpha = 1.;
  MM3D<double,int,MatrixStructureLowerTriangular,MatrixStructureSquare,MatrixStructureSquare, cblasEngine>::
    Multiply(matA, matB, localMatrixSizeM, localMatrixSizeN, MPI_COMM_WORLD, blasArgs);

  MPI_Barrier(MPI_COMM_WORLD);

  MMvalidate<double,int,cblasEngine>::validateLocal(matB, localMatrixSizeM, localMatrixSizeN,
    globalMatrixSizeM, globalMatrixSizeN, MPI_COMM_WORLD, blasArgs);

  MPI_Finalize();

  return 0;
}