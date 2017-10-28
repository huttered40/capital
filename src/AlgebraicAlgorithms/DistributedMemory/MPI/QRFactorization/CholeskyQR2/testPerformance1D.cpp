/* Author: Edward Hutter */

// System includes
#include <iostream>
#include <cstdlib>
#include <utility>
#include <cmath>
#include <mpi.h>

// Local includes
#include "CholeskyQR2.h"
#include "./../QRvalidate/QRvalidate.h"
#include "../../../../../Timer/Timer.h"

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
  // size -- total number of processors in the 1D grid

  int numIterations = 10;
  int globalMatrixDimensionM = (1<<(atoi(argv[1])));
  int globalMatrixDimensionN = (1<<(atoi(argv[2])));
  int localMatrixDimensionM = globalMatrixDimensionM/size;
  int localMatrixDimensionN = globalMatrixDimensionN;

  if (argc == 4)
  {
    numIterations = atoi(argv[3]);
  }

  MatrixTypeR matA(localMatrixDimensionN,localMatrixDimensionM,globalMatrixDimensionN,globalMatrixDimensionM);
  MatrixTypeR matQ(localMatrixDimensionN,localMatrixDimensionM,globalMatrixDimensionN,globalMatrixDimensionM);
  MatrixTypeS matR(localMatrixDimensionN,localMatrixDimensionN,globalMatrixDimensionN,globalMatrixDimensionN);

  matA.DistributeRandom(0, rank, 1, size, rank);

  cout << "Rank " << rank << " has local dimensionN - " << localMatrixDimensionN << ", localDimensionM - " << localMatrixDimensionM << endl;

  pTimer myTimer;
  // Loop for getting a good range of results.
  for (int i=0; i<numIterations; i++)
  {
    myTimer.setStartTime();
    CholeskyQR2<double,int,MatrixStructureRectangle,cblasEngine>::
      Factor1D(matA, matQ, matR, globalMatrixDimensionM, globalMatrixDimensionN, MPI_COMM_WORLD);
    myTimer.setEndTime();
    myTimer.printParallelTime(1e-8, MPI_COMM_WORLD, "1D-CQR2 iteration", i);
  }

  MPI_Finalize();
  return 0;
}