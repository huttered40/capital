/* Author: Edward Hutter */

// System includes
#include <iostream>
#include <cstdlib>
#include <cmath>

// Local includes
#include "./../../../Util/shared.h"
#include "CholeskyQR2.h"
#include "./../QRvalidate/QRvalidate.h"
#include "../../../Timer/Timer.h"

using namespace std;

// Idea: We calculate 3D Summa as usual, and then we pass it into the MMvalidate solo class

int main(int argc, char** argv)
{
  using MatrixTypeS = Matrix<DATATYPE,INTTYPE,MatrixStructureSquare,MatrixDistributerCyclic>;
  using MatrixTypeR = Matrix<DATATYPE,INTTYPE,MatrixStructureRectangle,MatrixDistributerCyclic>;
  using MatrixTypeLT = Matrix<DATATYPE,INTTYPE,MatrixStructureLowerTriangular,MatrixDistributerCyclic>;
  using MatrixTypeUT = Matrix<DATATYPE,INTTYPE,MatrixStructureSquare,MatrixDistributerCyclic>;

#ifdef PROFILE
  TAU_PROFILE_SET_CONTEXT(0)
#endif /*PROFILE*/

  // argv[1] - Matrix size x where x represents 2^x.
  // So in future, we might want t way to test non power of 2 dimension matrices

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  /*
    methodKey2 -> 0) Sequential validaton
                  1) Performance
                  2) Parallel validation
  */
  int methodKey2 = atoi(argv[1]);

  // size -- total number of processors in the tunable grid
  INTTYPE globalMatrixDimensionM = atoi(argv[2]);
  INTTYPE globalMatrixDimensionN = atoi(argv[3]);

  int baseCaseMultiplier = atoi(argv[4]);
  int inverseCutOffMultiplier = atoi(argv[5]);
  int panelDimensionMultiplier = atoi(argv[6]);
 
  // Use the grid that the user specifies in the command line
  int dimensionD = atoi(argv[7]);
  int dimensionC = atoi(argv[8]);
  int sliceSize = dimensionD*dimensionC;
  int pCoordX = rank%dimensionC;
  int pCoordY = (rank%sliceSize)/dimensionC;
  int pCoordZ = rank/sliceSize;

  INTTYPE localMatrixDimensionM = globalMatrixDimensionM/dimensionD;
  INTTYPE localMatrixDimensionN = globalMatrixDimensionN/dimensionC;

  int numIterations=atoi(argv[9]);

  // Note: matA and matR are rectangular, but the pieces owned by the individual processors may be square (so also rectangular)
  MatrixTypeR matA(globalMatrixDimensionN,globalMatrixDimensionM, dimensionC, dimensionD);
  MatrixTypeS matR(globalMatrixDimensionN,globalMatrixDimensionN, dimensionC, dimensionC);

  // Loop for getting a good range of results.
  for (int i=0; i<numIterations; i++)
  {
    // reset the matrix before timer starts
    matA.DistributeRandom(pCoordX, pCoordY, dimensionC, dimensionD, (rank%sliceSize));
    #ifdef CRITTER
    Critter_Clear();
    #endif
    TAU_FSTART(Total);
    auto commInfoTunable = util<DATATYPE,INTTYPE>::buildTunableTopology(
      MPI_COMM_WORLD, dimensionD, dimensionC);
    CholeskyQR2<DATATYPE,INTTYPE,cblasEngine>::FactorTunable(
      matA, matR, dimensionD, dimensionC, MPI_COMM_WORLD, commInfoTunable, inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier);
    util<DATATYPE,INTTYPE>::destroyTunableTopology(commInfoTunable);
    TAU_FSTOP(Total);
    #ifdef CRITTER
    Critter_Print();
    #endif
    if (methodKey2 == 2)
    {
      MatrixTypeR saveA = matA;
      matA.DistributeRandom(pCoordX, pCoordY, dimensionC, dimensionD, (rank%sliceSize));
      auto commInfoTunable = util<DATATYPE,INTTYPE>::buildTunableTopology(
        MPI_COMM_WORLD, dimensionD, dimensionC);
      QRvalidate<DATATYPE,INTTYPE>::validateParallelTunable(
        saveA, matA, matR, dimensionD, dimensionC, MPI_COMM_WORLD, commInfoTunable);
      util<DATATYPE,INTTYPE>::destroyTunableTopology(commInfoTunable);
    }
  }
  MPI_Finalize();
  return 0;
}
