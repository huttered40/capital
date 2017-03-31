/*
	Author: Edward Hutter
*/

#ifndef QR_3D_H_
#define QR_3D_H_

// System includes
#include <iostream>
//#include <stdio.h>
#include <vector>
#include <mpi.h>
#include <cmath>
#include <utility>
#include <cstdlib>
#include <cstdint>	// fixed-width integer types
#include <map>
#include <strings.h>
//#include <math.h>
#include <assert.h>
#include <cblas.h>	// OpenBLAS library. Will need to be linked in the Makefile
#include "./../../OpenBLAS/lapack-netlib/LAPACKE/include/lapacke.h"

// Need this to use fortran scalapack function
//extern void PDPOTRF(int *m, int *n, char *A, int *iA, int *jA, int *desca, int *ipiv, int *info);

template <typename T>
class solver
{
public:

  solver(uint32_t rank, uint32_t size, uint32_t nDims, int argc, char **argv);
  void solve();
  void scalapackCholesky();				// this routine is implemented in a special file, scalapackCholesky.h
  void printL();
  void lapackTest(std::vector<T> &data, std::vector<T> &dataL, std::vector<T> &dataLInverse, uint32_t n);
  void getResidualSequential();
  void getResidualParallel();
  void printInputA();

private:

  void constructGridCholesky();
  void constructGridQR();
  void distributeDataCyclicCholesky(bool inParallel);
  void distributeDataCyclicQR(bool inParallel);
  void QREngine();
  void CholeskyEngine(uint32_t dimXstart, uint32_t dimXend, uint32_t dimYstart, uint32_t dimYend, uint32_t matrixWindow, uint32_t matrixSize, uint32_t matrixCutSize);
  void MM(uint32_t dimXstartA, uint32_t dimXendA, uint32_t dimYstartA, uint32_t dimYendA, uint32_t dimXstartB, uint32_t dimXendB, uint32_t dimYstartB, uint32_t dimYendB, uint32_t dimXstartC, uint32_t dimXendC, uint32_t dimYstartC, uint32_t dimYendC, uint32_t matrixWindow, uint32_t matrixSize, uint32_t key, uint32_t matrixCutSize);
  void CholeskyRecurseBaseCase(uint32_t dimXstart, uint32_t dimXend, uint32_t dimYstart, uint32_t dimYend, uint32_t matrixWindow, uint32_t matrixSize, uint32_t matrixCutSize);
  void fillTranspose(uint32_t dimXstart, uint32_t dimXend, uint32_t dimYstart, uint32_t dimYend, uint32_t matrixWindow, uint32_t dir);

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

  uint32_t matrixDimSize;						// N x N matrix, where N = matrixDimSize
  uint32_t nDims;							// Represents the numDims of the processor grid
  uint32_t worldRank;						// Represents process rank in MPI_COMM_WORLD
  uint32_t worldSize;  						// Represents number of processors involved in computation
  uint32_t localSize;						// Represents the size length of a local 2D matrix that is held as a 1D matrix (can be calculated in other ways)
  uint32_t processorGridDimSize;					// Represents the size of the 3D processor grid, its the cubic root of worldSize
  uint32_t baseCaseSize;						// for quick lookUp when deciding when to stop recursing in LURecurse

  std::vector<int32_t> gridDims;
  std::vector<int32_t> gridCoords;

/*
    Sub-communicator information for 3D Grid, Row, Column, and Layer
*/
  MPI_Comm grid3D,layerComm,rowComm,colComm,depthComm;
  int32_t grid3DRank,layerCommRank,rowCommRank,colCommRank,depthCommRank;
  int32_t grid3DSize,layerCommSize,rowCommSize,colCommSize,depthCommSize;
  std::map<uint32_t,uint32_t> gridSizeLookUp;

/*
    Residual information
*/
  double matrixLNorm;
  double matrixLInverseNorm;
  double matrixANorm;

/*
  Data needed for the scalapack routines
*/
  int argc;
  char **argv;
  bool isQR;		// 0 for Cholesky, 1 for QR

};

#include "qr3DImp.h"		// for template-instantiation reasons
#include "scalapackSolver.h"
#endif /*QR_3D_H_*/
