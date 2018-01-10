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
  int numIterations = 1;
  pTimer myTimer;

  /*
    methodKey1 -> 0) 1D
                  1) 3D
                  2) Tunable
  */
  int methodKey1 = atoi(argv[1]);
  /*
    methodKey1 -> 0) Sequential validaton
                  1) Performance
  */
  int methodKey2 = atoi(argv[2]);
  if (methodKey2 == 1)
  {
    numIterations = atoi(argv[3]);
  }

  if (methodKey1 == 0)
  {
    // 1D
    int globalMatrixDimensionM = (1<<(atoi(argv[4])));
    int globalMatrixDimensionN = (1<<(atoi(argv[5])));
    int localMatrixDimensionM = globalMatrixDimensionM/size;
    int localMatrixDimensionN = globalMatrixDimensionN;
    MatrixTypeR matA(localMatrixDimensionN,localMatrixDimensionM,globalMatrixDimensionN,globalMatrixDimensionM);
    MatrixTypeR matQ(localMatrixDimensionN,localMatrixDimensionM,globalMatrixDimensionN,globalMatrixDimensionM);
    MatrixTypeS matR(localMatrixDimensionN,localMatrixDimensionN,globalMatrixDimensionN,globalMatrixDimensionN);

    matA.DistributeRandom(0, rank, 1, size, rank);

    cout << "Rank " << rank << " has local dimensionN - " << localMatrixDimensionN << ", localDimensionM - " << localMatrixDimensionM << endl;

    // Loop for getting a good range of results.
    for (int i=0; i<numIterations; i++)
    {
      myTimer.setStartTime();
      CholeskyQR2<double,int,cblasEngine>::
        Factor1D(matA, matQ, matR, globalMatrixDimensionM, globalMatrixDimensionN, MPI_COMM_WORLD);
      myTimer.setEndTime();
      myTimer.printParallelTime(1e-8, MPI_COMM_WORLD, "1D-CQR2 iteration", i);
    }
    if (methodKey2 == 0)
    {
      QRvalidate<double,int>::validateLocal1D(matA, matQ, matR, globalMatrixDimensionM, globalMatrixDimensionN, MPI_COMM_WORLD);
    }
  }
  else if (methodKey1 == 1)
  {
    // 3D
    int pGridDimensionSize = std::nearbyint(std::pow(size,1./3.));
    // size -- total number of processors in the 3D grid
    int helper = pGridDimensionSize;
    helper *= helper;
    int pCoordX = rank%pGridDimensionSize;
    int pCoordY = (rank%helper)/pGridDimensionSize;
    int pCoordZ = rank/helper;

    int globalMatrixDimensionM = (1<<(atoi(argv[4])));
    int globalMatrixDimensionN = (1<<(atoi(argv[5])));
    int localMatrixDimensionM = globalMatrixDimensionM/pGridDimensionSize;
    int localMatrixDimensionN = globalMatrixDimensionN/pGridDimensionSize;

    // New protocol: CholeskyQR_3D only works properly with square matrix A. Rectangular matrices must use CholeskyQR_Tunable
    MatrixTypeR matA(localMatrixDimensionN,localMatrixDimensionM,globalMatrixDimensionN,globalMatrixDimensionM);
    MatrixTypeR matQ(localMatrixDimensionN,localMatrixDimensionM,globalMatrixDimensionN,globalMatrixDimensionM);
    MatrixTypeS matR(localMatrixDimensionN,localMatrixDimensionN,globalMatrixDimensionN,globalMatrixDimensionN);

    matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize+pCoordY);

    cout << "Rank " << rank << " has local localDimensionN - " << localMatrixDimensionN << ", localDimensionM - " << localMatrixDimensionM << endl;

    // Loop for getting a good range of results.
    for (int i=0; i<numIterations; i++)
    {
      myTimer.setStartTime();
      CholeskyQR2<double,int,cblasEngine>::
        Factor3D(matA, matQ, matR, globalMatrixDimensionM, globalMatrixDimensionN, MPI_COMM_WORLD);
      myTimer.setEndTime();
      myTimer.printParallelTime(1e-8, MPI_COMM_WORLD, "3D-CQR2 iteration", i);
    }
    if (methodKey2 == 0)
    {
      QRvalidate<double,int>::validateLocal3D(matA, matQ, matR, globalMatrixDimensionM, globalMatrixDimensionN, MPI_COMM_WORLD);
    }
  }
  else if (methodKey1 == 2)
  {
    // Tunable
    /*
      methodKey2 -> 0) User-defined grid
                    1) Optimal grid
    */
    int methodKey3 = atoi(argv[4]);

    // size -- total number of processors in the tunable grid
    int numIterations = 1;
    int exponentM = atoi(argv[5]);
    int exponentN = atoi(argv[6]);
    int globalMatrixDimensionM = (1<<exponentM);
    int globalMatrixDimensionN = (1<<exponentN);
 
    int dimensionC,dimensionD;
    if (methodKey3 == 0)
    {
      // Use the grid that the user specifies in the command line
      int exponentD = atoi(argv[7]);
      int exponentC = atoi(argv[8]);

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

    // Loop for getting a good range of results.
    for (int i=0; i<numIterations; i++)
    {
      myTimer.setStartTime();
      CholeskyQR2<double,int,cblasEngine>::
        FactorTunable(matA, matQ, matR, globalMatrixDimensionM, globalMatrixDimensionN, dimensionD, dimensionC, MPI_COMM_WORLD);
      myTimer.setEndTime();
      myTimer.printParallelTime(1e-8, MPI_COMM_WORLD, "Tunable CQR2 iteration", i);
    }
    if (methodKey2 == 0)
    {
      QRvalidate<double,int>::validateLocalTunable(matA, matQ, matR, globalMatrixDimensionM, globalMatrixDimensionN, dimensionD, dimensionC, MPI_COMM_WORLD);
    }
  }
  else
  {
    MPI_Abort(MPI_COMM_WORLD,-1);
    return 0;
  }

  MPI_Finalize();
  return 0;
}
