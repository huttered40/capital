/*
	Author: Edward Hutter
*/

#ifndef MATRIX_MULT_H_
#define MATRIX_MULT_H_

#include <iostream>
#include <vector>
#include <mpi.h>

#include <cblas.h>	// OpenBLAS library. Will need to be linked in the Makefile
#include "./../../../OpenBLAS/lapack-netlib/LAPACKE/include/lapacke.h"

using namespace std;

template <typename T>
class matrixMult
{
public:
  matrixMult(MPI_Comm grid, uint32_t dim);		// I should set up the right P-grid that I need to use in here so that there is no confusion

  matrixMult
  (
		MPI_Comm rowCommunicator,
		MPI_Comm columnCommunicator,
		MPI_Comm layerCommunicator,
		MPI_Comm grid3DCommunicator,
		MPI_Comm depthCommunicator,
		uint32_t rowCommunicatorRank,
		uint32_t columnCommunicatorRank,
		uint32_t layerCommunicatorRank,
		uint32_t grid3DCommunicatorRank,
		uint32_t depthCommunicatorRank,
		uint32_t rowCommunicatorSize,
		uint32_t columnCommunicatorSize,
		uint32_t layerCommunicatorSize,
		uint32_t grid3DCommunicatorSize,
		uint32_t depthCommunicatorSize,
		std::vector<int> gridCoordinates
  );

  void multiply
  (
		std::vector<T> &matA,
		std::vector<T> &matB,
		std::vector<T> &matC,			// Should I pass these in as vectors or pointers?
         	uint32_t dimXstartA,
		uint32_t dimXendA,
		uint32_t dimYstartA,
		uint32_t dimYendA,
		uint32_t dimXstartB,
		uint32_t dimXendB,
		uint32_t dimYstartB,
		uint32_t dimYendB,
		uint32_t dimXstartC,
		uint32_t dimXendC,
		uint32_t dimYstartC,
		uint32_t dimYendC,
		uint32_t matrixWindow,
		uint32_t matrixSize,
		uint32_t key,
		uint32_t matrixCutSize,
		uint32_t layer				// Will layer be needed? I can pass in the specific layer, BUT I need to provide correct size
  );

  void scalapackMatrixMult(void);	// Extend this later

private:
  // a ton of different kinds of matrix multiplication goes here, called from multipy method, get these from the cholesky and the one(s) I need for QR
  // For now, these are very specific, maybe later I can generalize these to take care of all different layouts, storage, kinds of MM, rectangular MM, etc

  void constructGridMM(MPI_Comm grid, uint32_t dim);


  void multiply1
  (
         	uint32_t dimXstartA,
		uint32_t dimXendA,
		uint32_t dimYstartA,
		uint32_t dimYendA,
		uint32_t dimXstartB,
		uint32_t dimXendB,
		uint32_t dimYstartB,
		uint32_t dimYendB,
		uint32_t dimXstartC,
		uint32_t dimXendC,
		uint32_t dimYstartC,
		uint32_t dimYendC,
		uint32_t matrixWindow,
		uint32_t matrixSize,
		uint32_t key,
		uint32_t matrixCutSize,
		uint32_t layer				// Will layer be needed? I can pass in the specific layer, BUT I need to provide correct size
  );
  void multiply2
  (
         	uint32_t dimXstartA,
		uint32_t dimXendA,
		uint32_t dimYstartA,
		uint32_t dimYendA,
		uint32_t dimXstartB,
		uint32_t dimXendB,
		uint32_t dimYstartB,
		uint32_t dimYendB,
		uint32_t dimXstartC,
		uint32_t dimXendC,
		uint32_t dimYstartC,
		uint32_t dimYendC,
		uint32_t matrixWindow,
		uint32_t matrixSize,
		uint32_t key,
		uint32_t matrixCutSize,
		uint32_t layer				// Will layer be needed? I can pass in the specific layer, BUT I need to provide correct size
  );
  void multiply3
  (
         	uint32_t dimXstartA,
		uint32_t dimXendA,
		uint32_t dimYstartA,
		uint32_t dimYendA,
		uint32_t dimXstartB,
		uint32_t dimXendB,
		uint32_t dimYstartB,
		uint32_t dimYendB,
		uint32_t dimXstartC,
		uint32_t dimXendC,
		uint32_t dimYstartC,
		uint32_t dimYendC,
		uint32_t matrixWindow,
		uint32_t matrixSize,
		uint32_t key,
		uint32_t matrixCutSize,
		uint32_t layer				// Will layer be needed? I can pass in the specific layer, BUT I need to provide correct size
  );
  void multiply4
  (
         	uint32_t dimXstartA,
		uint32_t dimXendA,
		uint32_t dimYstartA,
		uint32_t dimYendA,
		uint32_t dimXstartB,
		uint32_t dimXendB,
		uint32_t dimYstartB,
		uint32_t dimYendB,
		uint32_t dimXstartC,
		uint32_t dimXendC,
		uint32_t dimYstartC,
		uint32_t dimYendC,
		uint32_t matrixWindow,
		uint32_t matrixSize,
		uint32_t key,
		uint32_t matrixCutSize,
		uint32_t layer				// Will layer be needed? I can pass in the specific layer, BUT I need to provide correct size
  );
  void multiply5
  (
         	uint32_t dimXstartA,
		uint32_t dimXendA,
		uint32_t dimYstartA,
		uint32_t dimYendA,
		uint32_t dimXstartB,
		uint32_t dimXendB,
		uint32_t dimYstartB,
		uint32_t dimYendB,
		uint32_t dimXstartC,
		uint32_t dimXendC,
		uint32_t dimYstartC,
		uint32_t dimYendC,
		uint32_t matrixWindow,
		uint32_t matrixSize,
		uint32_t key,
		uint32_t matrixCutSize,
		uint32_t layer				// Will layer be needed? I can pass in the specific layer, BUT I need to provide correct size
  );

  // The methods above do not need arguments because i will "move" the arguments passed into multiply() into my member variables and then will call one
  // of these methods, and then will "move" them back before the multiply() routine ends.

/*
	C <- A*B

	We "move" the passed-in data into these member vectors, do the computation, and then "move" the values back
*/
  std::vector<T> matrixA;
  std::vector<T> matrixB;
  std::vector<T> matrixC;

  MPI_Comm rowComm;
  MPI_Comm columnComm;
  MPI_Comm layerComm;
  MPI_Comm grid3DComm;
  MPI_Comm depthComm;
  uint32_t rowCommRank;
  uint32_t columnCommRank;
  uint32_t layerCommRank;
  uint32_t grid3DCommRank;
  uint32_t depthCommRank;
  uint32_t rowCommSize;
  uint32_t columnCommSize;
  uint32_t layerCommSize;
  uint32_t grid3DCommSize;
  uint32_t depthCommSize;
  std::vector<int> gridCoords;

};

#include "matrixMultImp.h"
#include "./../../scalapack/scalapackMatrixMult.h"

#endif /*MATRIX_MULT_H_*/
