/* Author: Edward Hutter */

#ifndef UTIL_H_
#define UTIL_H_

class util{
public:
  template<typename MatrixType, typename CommType>
  static typename MatrixType::ScalarType get_identity_residual(MatrixType& Matrix, CommType&& CommInfo, MPI_Comm comm);

  template<typename MatrixType, typename RefMatrixType, typename LambdaType>
  static typename MatrixType::ScalarType
    residual_local(MatrixType& Matrix, RefMatrixType& RefMatrix, LambdaType&& Lambda, MPI_Comm slice, int64_t sliceX, int64_t sliceY, int64_t sliceDimX, int64_t sliceDimY);

  template<typename ScalarType>
  static void block_to_cyclic_triangle(ScalarType* blockedData, ScalarType* cyclicData, int64_t num_elems, int64_t localDimensionRows, int64_t localDimensionColumns, int64_t sliceDim);

  template<typename ScalarType>
  static void block_to_cyclic_rect(ScalarType* blockedData, ScalarType* cyclicData, int64_t localDimensionRows, int64_t localDimensionColumns, int64_t sliceDim);

  template<typename ScalarType>
  static void cyclic_to_local(ScalarType* storeScalarType, ScalarType* storeScalarTypeI, int64_t localDimension, int64_t globalDimension, int64_t bcDimension, int64_t sliceDim, int64_t rankSlice);

  template<typename ScalarType>
  static void cyclic_to_block(ScalarType* dest, ScalarType* src, int64_t localDimension, int64_t globalDimension, int64_t bcDimension, int64_t sliceDim);

  template<typename MatrixType, typename CommType>
  static void transpose(MatrixType& mat, CommType&& CommInfo);

  static int64_t get_next_power2(int64_t localShift);

  template<typename MatrixType>
  static void remove_triangle(MatrixType& matrix, int64_t sliceX, int64_t sliceY, int64_t sliceDim, char dir);

  template<typename MatrixType>
  static void remove_triangle_local(MatrixType& matrix, int64_t sliceX, int64_t sliceY, int64_t sliceDim, char dir);
};

#include "util.hpp"
#endif /*UTIL_H_*/
