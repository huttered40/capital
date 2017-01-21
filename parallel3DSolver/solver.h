/*
	Defines the header class -> solver
*/

#ifndef SOLVER_H_
#define SOLVER_H_

// System includes
#include <iostream>
#include <vector>
#include <mpi.h>
#include <utility>
#include <cstdlib>
#include <map>
#include <cblas.h>	// OpenBLAS library. Will need to be linked in the Makefile
#include "./../OpenBLAS/lapack-netlib/LAPACKE/include/lapacke.h"

// Need this to use fortran scalapack function
extern void PDGETRF(int *m, int *n, char *A, int *iA, int *jA, int *desca, int *ipiv, int *info);

template <typename T>	// I want to be able to use integers, as well as floating points
class solver
{
public:

  solver(int rank, int size, int nDims, int matrixDimSize);
  void startUp(bool &flag);
  void collectDataCyclic();
  void solve();
  void solveScalapack();
  void printL();
  void lapackTest(int n);

private:

  void LURecurse(int dimXstart, int dimXend, int dimYstart, int dimYend, int matrixWindow, int matrixSize, int matrixTrack);
  void MM(int dimXstartA,int dimXendA,int dimYstartA,int dimYendA,int dimXstartB,int dimXendB,int dimYstartB,int dimYendB,int dimXstartC, int dimXendC, int dimYstartC, int dimYendC, int matrixWindow,int matrixSize, int key, int matrixTrack);
  void LURecurseBaseCase(int dimXstart, int dimXend, int dimYstart, int dimYend, int matrixWindow, int matrixSize, int matrixTrack);

  std::vector<std::vector<T> > matrixA;  	// track 1 matrix
  std::vector<std::vector<T> > matrixB;		// track 2 matrix
  

  /*
    Each rank must own a place to put intermediate results for L and U as well build it up
    I want to build it up in place like I did in the sequential case

    I changed these matrix data structures back to 2d vectors. Remember that cyclic data distribution
      makes this more complicated than as seen in lecture/elsewhere
  */
  std::vector<std::vector<T> > matrixU;
  std::vector<std::vector<T> > matrixL;
  std::vector<std::vector<T> > matrixUInverse;
  std::vector<std::vector<T> > matrixLInverse;
  std::vector<std::vector<T> > holdMatrix;	// this guy needs to hold the temporary matrix and can keep resizing himself

  int matrixDimSize;	// nxn matrix, where n=matrixDimSize
  int nDims;		// nDims represents the numDims of the processor grid
  int worldRank;	// represents process rank in MPI_COMM_WORLD
  int worldSize;  	// represents number of processors involved in computation

  int processorGridDimSize;		//Each data owns a cyclic subset of the matrix (n^2/P^(2/3)) in size, which is what the 3-d p-grid algorithm allows
  int baseCaseSize;	// for quick lookUp when deciding when to stop recursing in LURecurse

  std::vector<int> gridDims;
  std::vector<int> gridCoords;		// Do I really need these two as member variables?? Decide later...  
  /*
    I will want a sub-communicator for row and column of each rank for the MM
    Maybe I need some other ones as well. Decide this later...
  */
  MPI_Comm grid3D,layerComm,rowComm,colComm,depthComm;
  int grid3DRank,layerCommRank,rowCommRank,colCommRank,depthCommRank;
  int grid3DSize,layerCommSize,rowCommSize,colCommSize,depthCommSize;
  std::map<int,int> gridSizeLookUp;
};

#include "solverImp.h"		// for template-instantiation reasons

#endif /*SOLVER_H_*/
