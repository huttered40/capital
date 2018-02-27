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

#ifdef PROFILE
  TAU_PROFILE_SET_CONTEXT(0)
#endif /*PROFILE*/

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
    methodKey2 -> 0) Sequential validaton
                  1) Performance
                  2) Parallel validation
  */
  int methodKey2 = atoi(argv[2]);
  /*
    methodKey3 -> 0) Non powers of 2
                  1) Powers of 2
  */
  int methodKey3 = atoi(argv[3]);
  if (methodKey2 == 1)
  {
    numIterations = atoi(argv[4]);
  }

  if (methodKey1 == 0)
  {
    // 1D
    int globalMatrixDimensionM = (methodKey3 ? (1<<(atoi(argv[5]))) : atoi(argv[5]));
    int globalMatrixDimensionN = (methodKey3 ? (1<<(atoi(argv[6]))) : atoi(argv[6]));
/*
    int localMatrixDimensionM = globalMatrixDimensionM/size;
    int localMatrixDimensionN = globalMatrixDimensionN;
*/
    MatrixTypeR matA(globalMatrixDimensionN,globalMatrixDimensionM, 1, size);
    MatrixTypeR matQ(globalMatrixDimensionN,globalMatrixDimensionM, 1, size);
    MatrixTypeS matR(globalMatrixDimensionN,globalMatrixDimensionN, 1, 1);

    matA.DistributeRandom(0, rank, 1, size, rank);

    //cout << "Rank " << rank << " has local dimensionN - " << localMatrixDimensionN << ", localDimensionM - " << localMatrixDimensionM << endl;
    // Perform "cold run"
    CholeskyQR2<double,int,cblasEngine>::Factor1D(
      matA, matQ, matR, MPI_COMM_WORLD);
    myTimer.clear();

    // Loop for getting a good range of results.
    for (int i=0; i<numIterations; i++)
    {
      size_t index1 = myTimer.setStartTime("CholeskyQR2::Factor1D");
      CholeskyQR2<double,int,cblasEngine>::Factor1D(
        matA, matQ, matR, MPI_COMM_WORLD);
      myTimer.setEndTime("CholeskyQR2::Factor1D", index1);
      myTimer.finalize(MPI_COMM_WORLD);
      myTimer.clear();
      //myTimer.printParallelTime(1e-8, MPI_COMM_WORLD, "1D-CQR2 iteration", i);
    }
    if (methodKey2 == 0)
    {
      QRvalidate<double,int>::validateLocal1D(matA, matQ, matR, MPI_COMM_WORLD);
    }
    else if (methodKey2 == 2)
    {
      // No parallel validation yet for 1D. Might not want it or need it
    }
    else
    {
      //myTimer.printRunStats(MPI_COMM_WORLD, "1D-CQR2");
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

    int globalMatrixDimensionM = (methodKey3 ? (1<<(atoi(argv[5]))) : atoi(argv[5]));
    int globalMatrixDimensionN = (methodKey3 ? (1<<(atoi(argv[6]))) : atoi(argv[6]));

    int MMid = atoi(argv[7]);
    int TSid = atoi(argv[8]);
    int INVid = atoi(argv[9]);
    int inverseCutOffMultiplier = atoi(argv[10]);
    int baseCaseMultiplier = atoi(argv[11]);

    // New protocol: CholeskyQR_3D only works properly with square matrix A. Rectangular matrices must use CholeskyQR_Tunable
    MatrixTypeR matA(globalMatrixDimensionN,globalMatrixDimensionM,pGridDimensionSize,pGridDimensionSize);
    MatrixTypeR matQ(globalMatrixDimensionN,globalMatrixDimensionM,pGridDimensionSize,pGridDimensionSize);
    MatrixTypeS matR(globalMatrixDimensionN,globalMatrixDimensionN,pGridDimensionSize,pGridDimensionSize);

    matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize+pCoordY);

    // Perform "cold run"
    std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int> commInfo3D = setUpCommunicators(
      MPI_COMM_WORLD);
    CholeskyQR2<double,int,cblasEngine>::Factor3D(
      matA, matQ, matR, MPI_COMM_WORLD, commInfo3D, MMid, TSid, INVid, inverseCutOffMultiplier, baseCaseMultiplier);
    myTimer.clear();
    MPI_Comm_free(&std::get<0>(commInfo3D));
    MPI_Comm_free(&std::get<1>(commInfo3D));
    MPI_Comm_free(&std::get<2>(commInfo3D));
    MPI_Comm_free(&std::get<3>(commInfo3D));

    // Loop for getting a good range of results.
    for (int i=0; i<numIterations; i++)
    {
      //matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize+pCoordY);
      size_t index1 = myTimer.setStartTime("CholeskyQR2::Factor3D");
      std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int> commInfo3D = setUpCommunicators(
        MPI_COMM_WORLD);
      CholeskyQR2<double,int,cblasEngine>::Factor3D(
        matA, matQ, matR, MPI_COMM_WORLD, commInfo3D, MMid, TSid, INVid, inverseCutOffMultiplier, baseCaseMultiplier);
      myTimer.setEndTime("CholeskyQR2::Factor3D", index1);
      myTimer.finalize(MPI_COMM_WORLD);
      myTimer.clear();
      MPI_Comm_free(&std::get<0>(commInfo3D));
      MPI_Comm_free(&std::get<1>(commInfo3D));
      MPI_Comm_free(&std::get<2>(commInfo3D));
      MPI_Comm_free(&std::get<3>(commInfo3D));
      //myTimer.printParallelTime(1e-8, MPI_COMM_WORLD, "3D-CQR2 iteration", i);
    }
    if (methodKey2 == 0)
    {
      // No sequential validation right now. Not really necessary anymore.
    }
    if (methodKey2 == 2)
    {
      // matrix A was corrupted in CQR2, so reset it.
      //matA.DistributeRandom(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize+pCoordY);
      std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int> commInfo3D = setUpCommunicators(
        MPI_COMM_WORLD);
      QRvalidate<double,int>::validateParallel3D(
        matA, matQ, matR, MPI_COMM_WORLD, commInfo3D);
      MPI_Comm_free(&std::get<0>(commInfo3D));
      MPI_Comm_free(&std::get<1>(commInfo3D));
      MPI_Comm_free(&std::get<2>(commInfo3D));
      MPI_Comm_free(&std::get<3>(commInfo3D));
    }
    else
    {
      //myTimer.printRunStats(MPI_COMM_WORLD, "3D-CQR2");
    }
  }
  else if (methodKey1 == 2)
  {
    // Tunable
    /*
      methodKey4 -> 0) User-defined grid
                    1) Optimal grid
    */
    int methodKey4 = atoi(argv[5]);

    // size -- total number of processors in the tunable grid
    int numIterations = 1;
    int exponentM = atoi(argv[6]);
    int exponentN = atoi(argv[7]);
    int globalMatrixDimensionM = (methodKey3 ? (1<<exponentM) : exponentM);
    int globalMatrixDimensionN = (methodKey3 ? (1<<exponentN) : exponentN);

    int MMid = atoi(argv[8]);
    int TSid = atoi(argv[9]);
    int INVid = atoi(argv[10]);
    int inverseCutOffMultiplier = atoi(argv[11]);
    int baseCaseMultiplier = atoi(argv[12]);
 
    int dimensionC,dimensionD;
    if (methodKey4 == 0)
    {
      // Use the grid that the user specifies in the command line
      int exponentD = atoi(argv[13]);
      int exponentC = atoi(argv[14]);

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
    MatrixTypeR matA(globalMatrixDimensionN,globalMatrixDimensionM, dimensionC, dimensionD);
    MatrixTypeR matQ(globalMatrixDimensionN,globalMatrixDimensionM, dimensionC, dimensionD);
    MatrixTypeS matR(globalMatrixDimensionN,globalMatrixDimensionN, dimensionC, dimensionC);

    matA.DistributeRandom(pCoordX, pCoordY, dimensionC, dimensionD, (rank%sliceSize));

    // Perform "cold run"
    std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm> commInfoTunable = getTunableCommunicators(
      MPI_COMM_WORLD, dimensionD, dimensionC);
    CholeskyQR2<double,int,cblasEngine>::FactorTunable(
      matA, matQ, matR, dimensionD, dimensionC, MPI_COMM_WORLD, commInfoTunable, MMid, TSid, INVid, inverseCutOffMultiplier, baseCaseMultiplier);
    myTimer.clear();
    MPI_Comm_free(&std::get<0>(commInfoTunable));
    MPI_Comm_free(&std::get<1>(commInfoTunable));
    MPI_Comm_free(&std::get<2>(commInfoTunable));
    MPI_Comm_free(&std::get<3>(commInfoTunable));
    MPI_Comm_free(&std::get<4>(commInfoTunable));
    MPI_Comm_free(&std::get<5>(commInfoTunable));

    // Loop for getting a good range of results.
    for (int i=0; i<numIterations; i++)
    {
      // reset the matrix before timer starts
      matA.DistributeRandom(pCoordX, pCoordY, dimensionC, dimensionD, (rank%sliceSize));
      size_t index1 = myTimer.setStartTime("CholeskyQR2::FactorTunable");
      commInfoTunable = getTunableCommunicators(
        MPI_COMM_WORLD, dimensionD, dimensionC);
      CholeskyQR2<double,int,cblasEngine>::FactorTunable(
        matA, matQ, matR, dimensionD, dimensionC, MPI_COMM_WORLD, commInfoTunable, MMid, TSid, INVid, inverseCutOffMultiplier, baseCaseMultiplier);
      myTimer.setEndTime("CholeskyQR2::FactorTunable", index1);
      myTimer.finalize(MPI_COMM_WORLD);
      myTimer.clear();
      MPI_Comm_free(&std::get<0>(commInfoTunable));
      MPI_Comm_free(&std::get<1>(commInfoTunable));
      MPI_Comm_free(&std::get<2>(commInfoTunable));
      MPI_Comm_free(&std::get<3>(commInfoTunable));
      MPI_Comm_free(&std::get<4>(commInfoTunable));
      MPI_Comm_free(&std::get<5>(commInfoTunable));
      //myTimer.printParallelTime(1e-8, MPI_COMM_WORLD, "Tunable CQR2 iteration", i);
    }
    if (methodKey2 == 0)
    {
      // Currently no sequential validation. Not really necessary anymore.
    }
    else if (methodKey2 == 2)
    {
      // reset the matrix that was corrupted by TRSM in CQR2
      matA.DistributeRandom(pCoordX, pCoordY, dimensionC, dimensionD, (rank%sliceSize));
      commInfoTunable = getTunableCommunicators(
        MPI_COMM_WORLD, dimensionD, dimensionC);
      QRvalidate<double,int>::validateParallelTunable(
        matA, matQ, matR, dimensionD, dimensionC, MPI_COMM_WORLD, commInfoTunable);
      MPI_Comm_free(&std::get<0>(commInfoTunable));
      MPI_Comm_free(&std::get<1>(commInfoTunable));
      MPI_Comm_free(&std::get<2>(commInfoTunable));
      MPI_Comm_free(&std::get<3>(commInfoTunable));
      MPI_Comm_free(&std::get<4>(commInfoTunable));
      MPI_Comm_free(&std::get<5>(commInfoTunable));
    }
    else
    {
      //myTimer.printRunStats(MPI_COMM_WORLD, "Tunable CQR2");
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
