/*
	Author: Edward Hutter
*/

#ifndef QR_H_
#define QR_H_

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

//#include "./../cholesky/cholesky.h"				// could be wrong

template <typename T>
class qr
{
public:

  qr(uint32_t rank, uint32_t size, int argc, char **argv, MPI_Comm comm);
  
  void qrSolve(std::vector<T> &matrixA, std::vector<T> &matrixL, std::vector<T> &matrixLI, bool isData);
  void qrScalapack();				// this routine is implemented in a special file, scalapackCholesky.h
  void qrLAPack(std::vector<T> &data, std::vector<T> &dataQ, std::vector<T> &dataR, uint32_t m, uint32_t n, bool needData);
  void getResidual(std::vector<T> &matA, std::vector<T> &matL, std::vector<T> &matLI);
  void getResidualParallel();
  void printMatrixSequential(std::vector<T> &matrix, uint32_t n, bool isTriangle);
  void printMatrixParallel(std::vector<T> &matrix, uint32_t n);

private:

  void expandMatrix(std::vector<T> &data, uint32_t n);
  void allocateLayers();
  void trimMatrix(std::vector<T> &data, uint32_t n);
  void constructGridQR();
  void distributeData(std::vector<T> &matA);
  void MM(uint32_t dimXstartA, uint32_t dimXendA, uint32_t dimYstartA, uint32_t dimYendA, uint32_t dimXstartB, uint32_t dimXendB, uint32_t dimYstartB, uint32_t dimYendB, uint32_t dimXstartC, uint32_t dimXendC, uint32_t dimYstartC, uint32_t dimYendC, uint32_t matrixWindow, uint32_t matrixSize, uint32_t key, uint32_t matrixCutSize, uint32_t layer);
  void fillTranspose(uint32_t dimXstart, uint32_t dimXend, uint32_t dimYstart, uint32_t dimYend, uint32_t matrixWindow, uint32_t dir);
  void choleskyQR(std::vector<T> &matrixA, std::vector<T> &matrixQ, std::vector<T> &matrixR);

/*
    Each data owns a cyclic subset of the matrix (n^2/P^(2/3)) in size, which is what the 3-d p-grid algorithm allows
    Each rank must own a place to put intermediate results for L and U as well build it up
    I want to build it up in place like I did in the sequential case
*/

  std::vector<T> matrixA; 
  std::vector<T> matrixL;
  std::vector<T> matrixLInverse;
  std::vector<T> holdMatrix;					// this guy needs to hold the temporary matrix and can keep resizing himself
  std::vector<T> holdTransposeL;
  uint32_t matrixRowSize;					// N x k matrix, where matrixRowSize = N
  uint32_t matrixColSize;					// N x k matrix, where matrixColSize = k
  uint32_t localRowSize;					// Represents row size of local 2D matrix that is held as a 1D matrix
  uint32_t localColSize;

/*
    Processor Grid, Communicator, and Sub-communicator information for 3D Grid, Row, Column, and Layer
*/
  uint32_t nDims;						// Represents the numDims of the processor grid
  uint32_t worldRank;						// Represents process rank in MPI_COMM_WORLD
  uint32_t worldSize;  						// Represents number of processors involved in computation
  uint32_t processorGridDimTune;				// Represents c in a (d x c x c) processor grid
  uint32_t processorGridDimReact;				// Represents d in a (d x c x c) processor grid
  MPI_Comm worldComm, grid3D,layerComm,rowComm,colComm,depthComm;
  int32_t grid3DRank,layerCommRank,rowCommRank,colCommRank,depthCommRank;
  int32_t grid3DSize,layerCommSize,rowCommSize,colCommSize,depthCommSize;
  std::vector<int32_t> gridDims;
  std::vector<int32_t> gridCoords;
  std::map<uint32_t,uint32_t> gridSizeLookUp;			// Might be able to remove this later.

/*
    Residual information
*/
  double matrixANorm;
  double matrixQNorm;
  double matrixRNorm;

/*
  Member variables to hold input specifications
*/
  int argc;
  char **argv;

};

#include "qrImp.h"		// for template-instantiation reasons
//#include "./../cholesky/cholesky.h"
#include "scalapackQR.h"
#endif /*QR_H_*/
