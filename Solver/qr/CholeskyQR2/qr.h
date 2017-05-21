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
#include "./../../../OpenBLAS/lapack-netlib/LAPACKE/include/lapacke.h"

//#include "./../cholesky/cholesky.h"				// could be wrong

template <typename T>
class qr
{
public:

  qr(uint32_t rank, uint32_t size, int argc, char **argv, MPI_Comm comm);
  
  void qrSolve(std::vector<T> &matrixA, std::vector<T> &matrixL, std::vector<T> &matrixLI, bool isData, int mode);
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
  void distributeData(std::vector<T> &matA, bool cheat);
  void distributeDataLocal(std::vector<T> &matA);
  void choleskyQR_Tunable(std::vector<T> &matrixA, std::vector<T> &matrixQ, std::vector<T> &matrixR);
  void choleskyQR_1D(std::vector<T> &matrixA, std::vector<T> &matrixQ, std::vector<T> &matrixR);
  void choleskyQR_3D(void);		// these two functions are unimplemented for now.

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
  uint32_t matrixRowSize;					// m x n matrix, where matrixRowSize = m
  uint32_t matrixColSize;					// m x n matrix, where matrixColSize = n
  uint32_t localRowSize;					// Represents row size of local 2D matrix that is held as a 1D matrix
  uint32_t localColSize;

/*
    Processor Grid, Communicator, and Sub-communicator information for 3D Grid, Row, Column, and Layer
*/
  uint32_t nDims;						// Represents the numDims of the processor grid
  uint32_t worldRank;						// Represents process rank in MPI_COMM_WORLD
  uint32_t worldSize;  						// Represents number of processors involved in computation
  uint32_t pGridDimTune;					// Represents c in a (c x d x c) processor grid
  uint32_t pGridDimReact;					// Represents d in a (c x d x c) processor grid

  
  MPI_Comm worldComm, helperGrid1, helperGrid2;
  MPI_Comm subGrid1, subGrid2, subGrid3, subGrid4, subGrid5, subGrid6;	// look in qrImp.h for descriptions on what each of these does
  
  int32_t helperGrid1Rank, helperGrid2Rank;
  int32_t subGrid1Rank, subGrid2Rank, subGrid3Rank, subGrid4Rank, subGrid5Rank, subGrid6Rank;

  int32_t helperGrid1Size, helperGrid2Size;
  int32_t subGrid1Size, subGrid2Size, subGrid3Size, subGrid4Size, subGrid5Size, subGrid6Size;

  std::vector<int32_t> tunableGridDims;
  std::vector<int32_t> tunableGridCoords;
  std::map<uint32_t,uint32_t> tunableGridSizeLookUp;			// Might be able to remove this later.

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
#include "./../../scalapack/scalapackQR.h"
#endif /*QR_H_*/
