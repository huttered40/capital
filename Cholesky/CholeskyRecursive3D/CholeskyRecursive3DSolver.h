/*
	Author: Edward Hutter
*/

#ifndef CHOLESKY_RECURSIVE_3D_SOLVER_H_
#define CHOLESKY_RECURSIVE_3D_SOLVER_H_

// System includes
#include <iostream>
#include <vector>
#include <mpi.h>
#include <cmath>
#include <utility>
#include <cstdlib>
#include <map>
#include <cblas.h>	// OpenBLAS library. Will need to be linked in the Makefile
#include "./../../OpenBLAS/lapack-netlib/LAPACKE/include/lapacke.h"

// Need this to use fortran scalapack function
extern void PDPOTRF(int *m, int *n, char *A, int *iA, int *jA, int *desca, int *ipiv, int *info);

template <typename T>
class solver
{
public:

  solver(int rank, int size, int nDims, int matrixDimSize);
  void startUp(bool &flag);
  void collectDataCyclic();
  void solve();
  void solveScalapack();
  void printL();
  void lapackTest(std::vector<T> &data, std::vector<T> &dataL, std::vector<T> &dataLInverse, int n);
  void compareSolutions();
  void printInputA();

private:

  void CholeskyRecurse(int dimXstart, int dimXend, int dimYstart, int dimYend, int matrixWindow, int matrixSize, int matrixCutSize);
  void MM(int dimXstartA,int dimXendA,int dimYstartA,int dimYendA,int dimXstartB,int dimXendB,int dimYstartB,int dimYendB,int dimXstartC, int dimXendC, int dimYstartC, int dimYendC, int matrixWindow,int matrixSize, int key, int matrixCutSize);
  void CholeskyRecurseBaseCase(int dimXstart, int dimXend, int dimYstart, int dimYend, int matrixWindow, int matrixSize, int matrixCutSize);
  void fillTranspose(int dimXstart, int dimXend, int dimYstart, int dimYend, int matrixWindow, int dir);

/*
    Each data owns a cyclic subset of the matrix (n^2/P^(2/3)) in size, which is what the 3-d p-grid algorithm allows
    Each rank must own a place to put intermediate results for L and U as well build it up
    I want to build it up in place like I did in the sequential case
*/

  std::vector<std::vector<T> > matrixA;  			// Matrix contains all tracks of recursion. Builds recursively

  std::vector<T> matrixL;
  std::vector<T> matrixLInverse;
  std::vector<T> holdMatrix;					// this guy needs to hold the temporary matrix and can keep resizing himself
  std::vector<T> holdTransposeL;

  int matrixDimSize;						// N x N matrix, where N = matrixDimSize
  int nDims;							// Represents the numDims of the processor grid
  int worldRank;						// Represents process rank in MPI_COMM_WORLD
  int worldSize;  						// Represents number of processors involved in computation
  int localSize;						// Represents the size length of a local 2D matrix that is held as a 1D matrix (can be calculated in other ways)
  int processorGridDimSize;					// Represents the size of the 3D processor grid, its the cubic root of worldSize
  int baseCaseSize;						// for quick lookUp when deciding when to stop recursing in LURecurse

  std::vector<int> gridDims;
  std::vector<int> gridCoords;

/*
    Sub-communicator information for 3D Grid, Row, Column, and Layer
*/
  MPI_Comm grid3D,layerComm,rowComm,colComm,depthComm;
  int grid3DRank,layerCommRank,rowCommRank,colCommRank,depthCommRank;
  int grid3DSize,layerCommSize,rowCommSize,colCommSize,depthCommSize;
  std::map<int,int> gridSizeLookUp;

/*
    Residual information
*/
  double matrixLNorm;
  double matrixLInverseNorm;
  double matrixANorm;
};

#include "CholeskyRecursive3DSolverImp.h"		// for template-instantiation reasons

#endif /*CHOLESKY_RECURSIVE_3D_SOLVER_H_*/
