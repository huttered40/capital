/* Author: Edward Hutter */

// System includes
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <mpi.h>

// Local includes
#include "CholeskyQR2.h"
#include "./../QRvalidate/QRvalidate.h"
#include "../../../Timer/Timer.h"

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
  int numIterations = 10;
  int exponentM = atoi(argv[1]);
  int exponentN = atoi(argv[2]);
  int globalMatrixDimensionM = (1<<exponentM);
  int globalMatrixDimensionN = (1<<exponentN);

  int dimensionC,dimensionD;
  if (argc >= 5)
  {
    // Use the grid that the user specifies in the command line
    int exponentD = atoi(argv[3]);
    int exponentC = atoi(argv[4]);

/*
    // Do an exponent check, but first we need the log-2 of numPEs(size)
    int exponentNumPEs = std::nearbyint(std::log2(size));

    int sumExponentsD = (exponentNumPEs+exponentM+exponentM - exponentN - exponentN);
    int sumExponentsC = (exponentNumPEs+exponentN - exponentM);

    if ((sumExponentsD%3 != 0) || (sumExponentsC%3==0))
    {
      MPI_Abort(MPI_COMM_WORLD); 
    }
*/
    dimensionD = /*(1<<exponentD);*/exponentD;
    dimensionC = /*(1<<exponentC);*/exponentC;
  }
  else
  {
    // Do an exponent check, but first we need the log-2 of numPEs(size)
    int exponentNumPEs = std::nearbyint(std::log2(size));

    int sumExponentsD = (exponentNumPEs+exponentM+exponentM - exponentN - exponentN);
    int sumExponentsC = (exponentNumPEs+exponentN - exponentM);

    int exponentD;
    int exponentC;
    if (sumExponentsD%3 == 0) {exponentD = sumExponentsD/3;}
    else { exponentD = (sumExponentsD - (sumExponentsD%3))/3;}

    if (sumExponentsC <= 0) {exponentC = 0;}
    else if (sumExponentsC%3==0) {exponentC = sumExponentsC/3;}
    else {exponentC = (sumExponentsC - (sumExponentsC%3))/3;}

    // Find the optimal grid based on the number of processors, size, and the dimensions of matrix A
    dimensionD = std::min(size, (1<<exponentD));
    dimensionC = std::max(1,(1<<exponentC));

    // Extra error check so that we use all of the processors
    if (dimensionD*dimensionC*dimensionC < size)
    {
      dimensionD <<= (exponentNumPEs - exponentD - 2*exponentC);
    }
  }

  if (argc == 6)
  {
    numIterations = atoi(argv[5]);
  }

  if (rank==0)
  {
    cout << "dimensionD - " << dimensionD << " and dimensionC - " << dimensionC << std::endl;
  }

  int sliceSize = dimensionD*dimensionC;
  int pCoordX = rank%dimensionC;
  int pCoordY = (rank%sliceSize)/dimensionC;
  int pCoordZ = rank/sliceSize;

  int localMatrixDimensionM = globalMatrixDimensionM/dimensionD;
  int localMatrixDimensionN = globalMatrixDimensionN/dimensionC;

  // Note: matA and matR are rectangular, but the pieces owned by the individual processors may be square (so also rectangular)
  MatrixTypeR matA(localMatrixDimensionN,localMatrixDimensionM,globalMatrixDimensionN,globalMatrixDimensionM);
  MatrixTypeR matQ(localMatrixDimensionN,localMatrixDimensionM,globalMatrixDimensionN,globalMatrixDimensionM);
  MatrixTypeS matR(localMatrixDimensionN,localMatrixDimensionN,globalMatrixDimensionN,globalMatrixDimensionN);

  matA.DistributeRandom(pCoordX, pCoordY, dimensionC, dimensionD, (rank%sliceSize));

  pTimer myTimer;
  // Loop for getting a good range of results.
  for (int i=0; i<numIterations; i++)
  {
    myTimer.setStartTime();
    CholeskyQR2<double,int,MatrixStructureRectangle,cblasEngine>::
      FactorTunable(matA, matQ, matR, globalMatrixDimensionM, globalMatrixDimensionN, dimensionD, dimensionC, MPI_COMM_WORLD);
    myTimer.setEndTime();
    myTimer.printParallelTime(1e-8, MPI_COMM_WORLD, "Tunable CQR2 iteration", i);
  }

  MPI_Finalize();

  return 0;
}
