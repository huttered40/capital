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
//#include "../CFvalidate/CFvalidate.h"
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

  uint64_t globalMatrixSize = (1<<(atoi(argv[3])));
  uint64_t localMatrixSize = globalMatrixSize/pGridDimensionSize;

  pTimer myTimer;
  int numIterations = 1;

  if (methodKey1 == 0)
  {
    MatrixTypeL matL(localMatrixSize,localMatrixSize,globalMatrixSize,globalMatrixSize);
    MatrixTypeL matLI(localMatrixSize,localMatrixSize,globalMatrixSize,globalMatrixSize);

    matL.DistributeSymmetric(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize+pCoordY, true);
    if (methodKey2 == 1) {numIterations = atoi(argv[4]);}
    for (int i=0; i<numIterations; i++)
    {
      myTimer.setStartTime();
      RTI3D<double,int,cblasEngine>::Invert(matL, matLI, localMatrixSize, 'L', MPI_COMM_WORLD);
      myTimer.setEndTime();
      myTimer.printParallelTime(1e-8, MPI_COMM_WORLD, "RTI3D Lower", i);
    }

    //TIvalidate<double,int>::validateTI_Local(matL, matLI, localMatrixSize, globalMatrixSize, 'L', MPI_COMM_WORLD);
  }
  else
  {
    MatrixTypeR matR(localMatrixSize,localMatrixSize,globalMatrixSize,globalMatrixSize);
    MatrixTypeR matRI(localMatrixSize,localMatrixSize,globalMatrixSize,globalMatrixSize);

    matR.DistributeSymmetric(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize+pCoordY, true);
    if (methodKey2 == 1) {numIterations = atoi(argv[5]);}
    for (int i=0; i<numIterations; i++)
    {
      myTimer.setStartTime();
      RTI3D<double,int,cblasEngine>::Invert(matR, matRI, localMatrixSize, 'U', MPI_COMM_WORLD);
      myTimer.setEndTime();
      myTimer.printParallelTime(1e-8, MPI_COMM_WORLD, "RTI3D Upper", i);
    }

    //TIvalidate<double,int>::validateTI_Local(matR, matRI, localMatrixSize, globalMatrixSize, 'U', MPI_COMM_WORLD);
  }  

  MPI_Finalize();
  return 0;
}
