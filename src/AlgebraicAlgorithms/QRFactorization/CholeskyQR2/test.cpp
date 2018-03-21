/* Author: Edward Hutter */

// System includes
#include <iostream>
#include <cstdlib>
#include <cmath>

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
    methodKey2 -> 0) Sequential validaton
                  1) Performance
                  2) Parallel validation
  */
  int methodKey2 = atoi(argv[1]);

  // size -- total number of processors in the tunable grid
  int globalMatrixDimensionM = atoi(argv[2]);
  int globalMatrixDimensionN = atoi(argv[3]);

  int baseCaseMultiplier = atoi(argv[4]);
  int inverseCutOffMultiplier = atoi(argv[5]);
  int panelDimensionMultiplier = atoi(argv[6]);
 
  // Use the grid that the user specifies in the command line
  int dimensionD = atoi(argv[7]);
  int dimensionC = atoi(argv[8]);
/* Optimal grid choice not in code path anymore. Its always better to manually specify the tunable grid dimensions.
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
*/
  int sliceSize = dimensionD*dimensionC;
  int pCoordX = rank%dimensionC;
  int pCoordY = (rank%sliceSize)/dimensionC;
  int pCoordZ = rank/sliceSize;

  int localMatrixDimensionM = globalMatrixDimensionM/dimensionD;
  int localMatrixDimensionN = globalMatrixDimensionN/dimensionC;

  // Note: matA and matR are rectangular, but the pieces owned by the individual processors may be square (so also rectangular)
  MatrixTypeR matA(globalMatrixDimensionN,globalMatrixDimensionM, dimensionC, dimensionD);
  MatrixTypeS matR(globalMatrixDimensionN,globalMatrixDimensionN, dimensionC, dimensionC);

  matA.DistributeRandom(pCoordX, pCoordY, dimensionC, dimensionD, (rank%sliceSize));
  // save A for correctness checking, since I am overwriting A now.
  MatrixTypeR saveA = matA;

  if (methodKey2 == 1)
  {
    numIterations = atoi(argv[9]);
  }
  // Loop for getting a good range of results.
  for (int i=0; i<numIterations; i++)
  {
    // reset the matrix before timer starts
    matA.DistributeRandom(pCoordX, pCoordY, dimensionC, dimensionD, (rank%sliceSize));
#ifdef CRITTER
    Critter_Clear();
#endif
    auto commInfoTunable = util<double,int>::buildTunableTopology(
      MPI_COMM_WORLD, dimensionD, dimensionC);
    CholeskyQR2<double,int,cblasEngine>::FactorTunable(
      matA, matR, dimensionD, dimensionC, MPI_COMM_WORLD, commInfoTunable, inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier);
#ifdef CRITTER
    Critter_Print();
#endif
    MPI_Comm_free(&std::get<0>(commInfoTunable));
    MPI_Comm_free(&std::get<1>(commInfoTunable));
    MPI_Comm_free(&std::get<2>(commInfoTunable));
    MPI_Comm_free(&std::get<3>(commInfoTunable));
    MPI_Comm_free(&std::get<4>(commInfoTunable));
    MPI_Comm_free(&std::get<5>(commInfoTunable));
  }
  if (methodKey2 == 0)
  {
    // Currently no sequential validation. Not really necessary anymore.
  }
  else if (methodKey2 == 2)
  {
    auto commInfoTunable = util<double,int>::buildTunableTopology(
      MPI_COMM_WORLD, dimensionD, dimensionC);
    QRvalidate<double,int>::validateParallelTunable(
      saveA, matA, matR, dimensionD, dimensionC, MPI_COMM_WORLD, commInfoTunable);
    MPI_Comm_free(&std::get<0>(commInfoTunable));
    MPI_Comm_free(&std::get<1>(commInfoTunable));
    MPI_Comm_free(&std::get<2>(commInfoTunable));
    MPI_Comm_free(&std::get<3>(commInfoTunable));
    MPI_Comm_free(&std::get<4>(commInfoTunable));
    MPI_Comm_free(&std::get<5>(commInfoTunable));
  }

  MPI_Finalize();
  return 0;
}
