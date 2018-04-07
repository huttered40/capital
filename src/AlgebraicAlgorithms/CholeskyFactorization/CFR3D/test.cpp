/* Author: Edward Hutter */

// System includes
#include <iostream>
#include <cstdlib>
#include <utility>
#include <cmath>
#include <string>
#include <utility>

// Local includes
#include "./../../../Util/shared.h"
#include "CFR3D.h"
#include "../CFvalidate/CFvalidate.h"
#include "../../../Timer/Timer.h"

using namespace std;

template<
		typename T, typename U,
		template<typename,typename, template<typename,typename,int> class> class StructureA,
  		template<typename,typename, template<typename,typename,int> class> class StructureB,
  		template<typename,typename,int> class Distribution
	>
static pair<T,T> runTestCF(
                        Matrix<T,U,StructureA,Distribution>& matA,
                        Matrix<T,U,StructureB,Distribution>& matT,
			char dir, int inverseCutOffMultiplier, int blockSizeMultiplier, int panelDimensionMultiplier,
			int pCoordX, int pCoordY, int pGridDimensionSize, FILE* fptrTotal, FILE* fptrAvg, FILE* fptrNumericsTotal, FILE* fptrNumericsAvg,
			int iterNum, int numIter, int rank, int size, int& numFuncs
)
{
  double totalTime;
  // Reset matrixA
  matA.DistributeSymmetric(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize+pCoordY, true);
  #ifdef CRITTER
  Critter_Clear();
  #endif
  TAU_FSTART(Total);
  #ifdef PERFORMANCE
  double startTime=MPI_Wtime();
  #endif
  auto commInfo3D = util<T,U>::build3DTopology(MPI_COMM_WORLD);
  CFR3D<T,U,cblasEngine>::Factor(
    matA, matT, inverseCutOffMultiplier, blockSizeMultiplier, panelDimensionMultiplier, dir, MPI_COMM_WORLD, commInfo3D);
  util<T,U>::destroy3DTopology(commInfo3D);
  #ifdef PERFORMANCE
  totalTime=MPI_Wtime() - startTime;
  MPI_Reduce(MPI_IN_PLACE, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (rank == 0) { cout << "\nPERFORMANCE\nTotal time: " << totalTime << endl; fprintf(fptrTotal, "%d\t%d\t%g\n", size, iterNum, totalTime); }
  #endif
  TAU_FSTOP_FILE(Total, fptrTotal, iterNum, numFuncs);
  #ifdef CRITTER
  Critter_Print(fptrTotal, iterNum, fptrAvg, numIter);
  #endif
/* Sequential validation is no longer in the codepath. For use, create a new branch and comment in this code.
  if (methodKey2 == 0)
  {
    Matrix<T,U,StructureA,Distribution> saveA = matA;
    matA.DistributeSymmetric(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize+pCoordY, true);
    matA.DistributeSymmetric(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize+pCoordY, true);
    CFvalidate<T,U>::validateLocal(saveA, matA, dir, MPI_COMM_WORLD);
  }
*/
  if (rank == 0) { std::cout << "\nNUMERICS\n"; }
  Matrix<T,U,StructureA,Distribution> saveA = matA;
  saveA.DistributeSymmetric(pCoordX, pCoordY, pGridDimensionSize, pGridDimensionSize, pCoordX*pGridDimensionSize+pCoordY, true);
  commInfo3D = util<T,U>::build3DTopology(MPI_COMM_WORLD);
  T error = CFvalidate<T,U>::validateParallel(
    saveA, matA, dir, MPI_COMM_WORLD, commInfo3D);
  MPI_Reduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  util<T,U>::destroy3DTopology(commInfo3D);
  return make_pair(error, totalTime);
}

int main(int argc, char** argv)
{
  using MatrixTypeA = Matrix<DATATYPE,INTTYPE,MatrixStructureSquare,MatrixDistributerCyclic>;
  using MatrixTypeL = Matrix<DATATYPE,INTTYPE,MatrixStructureSquare,MatrixDistributerCyclic>;
  using MatrixTypeR = Matrix<DATATYPE,INTTYPE,MatrixStructureSquare,MatrixDistributerCyclic>;

  #ifdef PROFILE
  TAU_PROFILE_SET_CONTEXT(0)
  #endif /*PROFILE*/

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
  char dir = (methodKey1==0 ? 'L' : 'U');
  INTTYPE globalMatrixSize = atoi(argv[2]);
  int blockSizeMultiplier = atoi(argv[3]);
  int inverseCutOffMultiplier = atoi(argv[4]); // multiplies baseCase dimension by sucessive 2
  int panelDimensionMultiplier = atoi(argv[5]);
  int numIterations = atoi(argv[6]);

  string fileStr = argv[7];
  string fileStrTotal=fileStr;
  string fileStrAvg=fileStr;
  string fileStrNumericsTotal=fileStr + "_numerics.txt";
  string fileStrNumericsAvg=fileStr + "_numerics_avg.txt";
  #ifdef PROFILE
  fileStrTotal += "_timer.txt";
  fileStrAvg += "_timer_avg.txt";
  #endif
  #ifdef CRITTER
  fileStrTotal += "_critter.txt";
  fileStrAvg += "_critter_avg.txt";
  #endif
  #ifdef PERFORMANCE
  fileStrTotal += "_perf.txt";
  fileStrAvg += "_perf_avg.txt";
  #endif
  FILE* fptrTotal = fopen(fileStrTotal.c_str(),"w");
  FILE* fptrAvg = fopen(fileStrAvg.c_str(),"w");
  FILE* fptrNumericsTotal = fopen(fileStrNumericsTotal.c_str(),"w");
  FILE* fptrNumericsAvg = fopen(fileStrNumericsAvg.c_str(),"w");

  MatrixTypeA matA(globalMatrixSize,globalMatrixSize, pGridDimensionSize, pGridDimensionSize);
  MatrixTypeA matT(globalMatrixSize,globalMatrixSize, pGridDimensionSize, pGridDimensionSize);

  DATATYPE totalError = 0;
  double totalTime = 0;
  int numFuncs=0;
  for (int i=0; i<numIterations; i++)
  {
    pair<DATATYPE,double> info = runTestCF(matA, matT, dir, inverseCutOffMultiplier, blockSizeMultiplier, panelDimensionMultiplier, pCoordX, pCoordY, pGridDimensionSize,
      fptrTotal, fptrAvg, fptrNumericsTotal, fptrNumericsAvg, i, numIterations, rank, size, numFuncs);
    if (rank == 0)
    {
      fprintf(fptrNumericsTotal, "%d\t%g\n", size, i, info.first);
      totalError += info.first;
      totalTime += info.second;
    }
  }
  if (rank == 0)
  {
    fprintf(fptrNumericsAvg, "%d\t%g\n", size, totalError/numIterations);
    #ifdef PERFORMANCE
    fprintf(fptrAvg, "%d\t%g\n", size, totalTime/numIterations);
    #endif
  }
  fclose(fptrTotal);
  #ifdef PERFORMANCE
  fclose(fptrAvg);
  #endif
  #ifdef CRITTER
  fclose(fptrAvg);
  #endif
  #ifdef PROFILE
  util<DATATYPE,INTTYPE>::processAveragesFromFile(fptrAvg, fileStrTotal, numFuncs, numIterations, rank);
  fclose(fptrAvg);
  #endif

  MPI_Finalize();
  return 0;
}
