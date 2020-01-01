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

  template<typename T, typename U>
  static void block_to_cyclic(std::vector<T>& blockedData, T* cyclicData, U localDimensionRows, U localDimensionColumns, int64_t sliceDim, char dir);

  template<typename T, typename U>
  static void block_to_cyclic(T* blockedData, T* cyclicData, U localDimensionRows, U localDimensionColumns, int64_t sliceDim);

  template<typename T, typename U>
  static void cyclic_to_local(T* storeT, T* storeTI, U localDimension, U globalDimension, U bcDimension, int64_t sliceDim, int64_t rankSlice);

  template<typename MatrixType, typename CommType>
  static void transpose(MatrixType& mat, CommType&& CommInfo);

  template<typename U>
  static U get_next_power2(U localShift);

  template<typename MatrixType>
  static void remove_triangle(MatrixType& matrix, int64_t sliceX, int64_t sliceY, int64_t sliceDim, char dir);

  template<typename T, typename U>
  static void random_fill(T* A, U dimensionX, U dimensionY, U globalDimensionX, int64_t globalDimensionY, int64_t localPgridDimX,
                          int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key);

  template<typename T, typename U>
  static void random_fill_symmetric(T* A, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, int64_t localPgridDimX,
                                    int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key);
};

#include "util.hpp"
#endif /*UTIL_H_*/
