/* Author: Edward Hutter */

// System includes
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <cmath>
#include <utility>

// Local includes
#include "./../../../Util/shared.h"
#include "CholeskyQR2.h"
#include "./../QRvalidate/QRvalidate.h"
#include "../../../Timer/CTFtimer.h"

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

  util<DATATYPE,INTTYPE>::InitialGEMM();

  // size -- total number of processors in the tunable grid
  INTTYPE globalMatrixDimensionM = atoi(argv[1]);
  INTTYPE globalMatrixDimensionN = atoi(argv[2]);

  int baseCaseMultiplier = atoi(argv[3]);
  int inverseCutOffMultiplier = atoi(argv[4]);
  int panelDimensionMultiplier = atoi(argv[5]);
 
  // Use the grid that the user specifies in the command line
  int dimensionD = atoi(argv[6]);
  int dimensionC = atoi(argv[7]);
  int sliceSize = dimensionD*dimensionC;
  #ifdef BLUEWATERS
  int helper = dimensionC*dimensionC;
  int pCoordZ = rank%dimensionC;
  int pCoordY = rank/helper;
  int pCoordX = (rank%helper)/dimensionC;
  #else
  int pCoordX = rank%dimensionC;
  int pCoordY = (rank%sliceSize)/dimensionC;
  int pCoordZ = rank/sliceSize;
  #endif

  INTTYPE localMatrixDimensionM = globalMatrixDimensionM/dimensionD;
  INTTYPE localMatrixDimensionN = globalMatrixDimensionN/dimensionC;

  int numIterations=atoi(argv[8]);
  string fileStr = argv[9];
  string fileStrTotal=fileStr;
  #ifdef PROFILE
  fileStrTotal += "_timer.txt";
  #endif
  #ifdef CRITTER
  fileStrTotal += "_critter.txt";
  #endif
  #ifdef PERFORMANCE
  string fileStrNumericsTotal=fileStr;
  fileStrTotal += "_perf.txt";
  fileStrNumericsTotal += "_numerics.txt";
  ofstream fptrNumericsTotal;
  #endif
  ofstream fptrTotal;
  if (rank == 0)
  {
    fptrTotal.open(fileStrTotal.c_str());
    #ifdef PERFORMANCE
    fptrNumericsTotal.open(fileStrNumericsTotal.c_str());
    #endif
  }

  // Note: matA and matR are rectangular, but the pieces owned by the individual processors may be square (so also rectangular)
  MatrixTypeR matA(globalMatrixDimensionN,globalMatrixDimensionM, dimensionC, dimensionD);
  MatrixTypeS matR(globalMatrixDimensionN,globalMatrixDimensionN, dimensionC, dimensionC);

  #ifdef PERFORMANCE
  DATATYPE totalError1 = 0;
  DATATYPE totalError2 = 0;
  double totalTime = 0;
  #endif

  // Critter debugging
//  #ifdef CRITTER
//  cout << "I am rank " << rank << " of " << size << endl;
//  MPI_Finalize();
//  return 0;
//  #endif

  int numFuncs = 0;				// For figuring out how many functions are being profiled (smart way to find average over all iterations)
  int i;
  for (i=0; i<numIterations; i++)
  {
    double saveTime;
    // reset the matrix before timer starts
    #ifdef BLUEWATERS
    matA.DistributeRandom(pCoordX, pCoordY, dimensionC, dimensionD, rank/dimensionC);
    #else
    matA.DistributeRandom(pCoordX, pCoordY, dimensionC, dimensionD, (rank%sliceSize));
    #endif
    MPI_Barrier(MPI_COMM_WORLD);	// make sure each process starts together
    #ifdef CRITTER
    Critter_Clear();
    #endif
    TAU_FSTART(Total);
    #ifdef PERFORMANCE
    volatile double startTime=MPI_Wtime();
    #endif
    auto commInfoTunable = util<DATATYPE,INTTYPE>::buildTunableTopology(
      MPI_COMM_WORLD, dimensionD, dimensionC);
    CholeskyQR2<DATATYPE,INTTYPE,cblasEngine>::FactorTunable(
      matA, matR, dimensionD, dimensionC, MPI_COMM_WORLD, commInfoTunable, inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier);
    util<DATATYPE,INTTYPE>::destroyTunableTopology(commInfoTunable);
    #ifdef PERFORMANCE
    double iterTimeLocal = MPI_Wtime() - startTime;
    double iterTimeGlobal = 0;
    MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
      fptrTotal << size << "\t" << i << "\t" << globalMatrixDimensionM << "\t" << globalMatrixDimensionN << "\t" << iterTimeGlobal << endl;
      totalTime += iterTimeGlobal;
    }
    #endif
    TAU_FSTOP_FILE(Total, fptrTotal, i, numFuncs);
    #ifdef CRITTER
    Critter_Print(fptrTotal, i);
    #endif

    #ifdef PERFORMANCE
    MatrixTypeR saveA = matA;
    #ifdef BLUEWATERS
    saveA.DistributeRandom(pCoordX, pCoordY, dimensionC, dimensionD, rank/dimensionC);
    #else
    saveA.DistributeRandom(pCoordX, pCoordY, dimensionC, dimensionD, (rank%sliceSize));
    #endif
    commInfoTunable = util<DATATYPE,INTTYPE>::buildTunableTopology(
      MPI_COMM_WORLD, dimensionD, dimensionC);
    pair<DATATYPE,DATATYPE> error = QRvalidate<DATATYPE,INTTYPE>::validateParallelTunable(
      saveA, matA, matR, dimensionD, dimensionC, MPI_COMM_WORLD, commInfoTunable);
    util<DATATYPE,INTTYPE>::destroyTunableTopology(commInfoTunable);
    double residualErrorGlobal,orthogonalityErrorGlobal;
    MPI_Reduce(&error.first, &residualErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&error.second, &orthogonalityErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0)
    {
      fptrNumericsTotal << size << "\t" << i << "\t" << residualErrorGlobal << "\t" << orthogonalityErrorGlobal << endl;
      totalError1 += residualErrorGlobal;
      totalError2 += orthogonalityErrorGlobal;
    }
    #endif
  }
  if (rank == 0)
  {
    fptrTotal.close();
    #ifdef PERFORMANCE
    fptrNumericsTotal.close();
    #endif
  }
  MPI_Finalize();
  return 0;
}
