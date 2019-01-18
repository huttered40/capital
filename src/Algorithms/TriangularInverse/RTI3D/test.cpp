/* Author: Edward Hutter */

// System includes
#include <iostream>
#include <cstdlib>
#include <utility>
#include <cmath>
#include <string>
#include <mpi.h>

// Local includ#include "CFR3D.h"
#include "RTI3D.h"
#include "../TIvalidate/TIvalidate.h"
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

  pTimer myTimer;
  int numIterations = 1;

  if (methodKey1 == 0)
  {
    MatrixTypeL matL(globalMatrixSize,globalMatrixSize, pGridDimensionSize, pGridDimensionSize);
    MatrixTypeL matLI(globalMatrixSize,globalMatrixSize, pGridDimensionSize, pGridDimensionSize);
    matL.DistributeSymmetric(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize+pCoordY, true);

    if (methodKey2 == 1) {numIterations = atoi(argv[5]);}
    for (int i=0; i<numIterations; i++)
    {
      myTimer.setStartTime();
      RTI3D<double,int,cblasEngine>::Invert(matL, matLI, 'L', MPI_COMM_WORLD);
      myTimer.setEndTime();
      myTimer.printParallelTime(1e-8, MPI_COMM_WORLD, "RTI3D Lower", i);
    }
    if (methodKey2 == 0)
    {
      TIvalidate<double,int>::validateTI_Local(matLI, 'L', MPI_COMM_WORLD);
    }
  }
  else
  {
    MatrixTypeR matR(globalMatrixSize,globalMatrixSize, pGridDimensionSize, pGridDimensionSize);
    MatrixTypeL matRI(globalMatrixSize,globalMatrixSize, pGridDimensionSize, pGridDimensionSize);
    matR.DistributeSymmetric(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize+pCoordY, true);

    if (methodKey2 == 1) {numIterations = atoi(argv[5]);}
    for (int i=0; i<numIterations; i++)
    {
      myTimer.setStartTime();
      RTI3D<double,int,cblasEngine>::Invert(matR, matRI, 'U', MPI_COMM_WORLD);
      myTimer.setEndTime();
      myTimer.printParallelTime(1e-8, MPI_COMM_WORLD, "RTI3D Upper", i);
    }

    if (methodKey2 == 0)
    {
      TIvalidate<double,int>::validateTI_Local(matRI, 'U', MPI_COMM_WORLD);
    }
  }  

  MPI_Finalize();
  return 0;
}
