/* Author: Edward Hutter */

// System includes
#include <iostream>
#include <cstdlib>
#include <utility>
#include <cmath>
#include <string>
#include <mpi.h>

// Local includes
#include "CFR3D.h"
#include "../CFvalidate/CFvalidate.h"
#include "../../../Timer/Timer.h"

using namespace std;

int main(int argc, char** argv)
{
  using MatrixTypeA = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;
  using MatrixTypeL = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;
  using MatrixTypeR = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;

  // argv[1] - Matrix size x where x represents 2^x.
  // So in future, we might want t way to test non power of 2 dimension matrices

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // size -- total number of processors in the 3D grid
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  int pGridDimensionSize = std::nearbyint(std::pow(size,1./3.));
  int helper = pGridDimensionSize;
  helper *= helper;
  int pCoordX = rank%pGridDimensionSize;
  int pCoordY = (rank%helper)/pGridDimensionSize;
  int pCoordZ = rank/helper;

  /*
    methodKey1 -> 0) Lower
		              1) Upper
  */
  int methodKey1 = atoi(argv[1]);
  /*
    methodKey2 -> 0) Sequential Validation
		              1) Performance
  */
  int methodKey2 = atoi(argv[2]);
  /*
    methodKey3 -> 0) Non power of 2 dimenson
		              1) Power of 2 dimension
  */
  int methodKey3 = atoi(argv[3]);

  uint64_t globalMatrixSize = (methodKey3 ? (1<<(atoi(argv[4]))) : atoi(argv[4]));
  int blockSizeMultiplier = atoi(argv[5]);

  pTimer myTimer;
  int numIterations = 1;

  if (methodKey1 == 0)
  {
    MatrixTypeA matA(globalMatrixSize,globalMatrixSize, pGridDimensionSize, pGridDimensionSize);
    MatrixTypeR matL(globalMatrixSize,globalMatrixSize, pGridDimensionSize, pGridDimensionSize);
    MatrixTypeR matLI(globalMatrixSize,globalMatrixSize, pGridDimensionSize, pGridDimensionSize);

    matA.DistributeSymmetric(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize+pCoordY, true);
    if (methodKey2 == 1) {numIterations = atoi(argv[6]);}
    for (int i=0; i<numIterations; i++)
    {
      myTimer.setStartTime();
      CFR3D<double,int,cblasEngine>::Factor(matA, matL, matLI, 'L', blockSizeMultiplier, MPI_COMM_WORLD);
      myTimer.setEndTime();
      myTimer.printParallelTime(1e-8, MPI_COMM_WORLD, "CFR3D Lower", i);
    }
    CFvalidate<double,int>::validateCF_Local(matL, matLI, 'L', MPI_COMM_WORLD);
  }
  else
  {
    MatrixTypeA matA(globalMatrixSize,globalMatrixSize, pGridDimensionSize, pGridDimensionSize);
    MatrixTypeR matR(globalMatrixSize,globalMatrixSize, pGridDimensionSize, pGridDimensionSize);
    MatrixTypeR matRI(globalMatrixSize,globalMatrixSize, pGridDimensionSize, pGridDimensionSize);

    matA.DistributeSymmetric(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize+pCoordY, true);
    if (methodKey2 == 1) {numIterations = atoi(argv[6]);}
    for (int i=0; i<numIterations; i++)
    {
      myTimer.setStartTime();
      CFR3D<double,int,cblasEngine>::Factor(matA, matR, matRI, 'U', blockSizeMultiplier, MPI_COMM_WORLD);
      myTimer.setEndTime();
      myTimer.printParallelTime(1e-8, MPI_COMM_WORLD, "CFR3D Upper", i);
    }

    CFvalidate<double,int>::validateCF_Local(matR, matRI, 'U', MPI_COMM_WORLD);
  }  

  MPI_Finalize();
  return 0;
}
