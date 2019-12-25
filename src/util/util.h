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
  static void block_to_cyclic(std::vector<T>& blockedData, std::vector<T>& cyclicData, U localDimensionRows, U localDimensionColumns, int64_t sliceDim, char dir);

  template<typename T, typename U>
  static void block_to_cyclic(T* blockedData, T* cyclicData, U localDimensionRows, U localDimensionColumns, int64_t sliceDim);

//  template<typename MatrixType>
//  static std::vector<typename MatrixType::ScalarType> get_reference_matrix(MatrixType& myMatrix, int64_t key, MPI_Comm slice, int64_t commDim);

  template<typename MatrixType, typename CommType>
  static void transpose(MatrixType& mat, CommType&& CommInfo);

  template<typename U>
  static U get_next_power2(U localShift);

  template<typename MatrixType>
  static void remove_triangle(MatrixType& matrix, typename MatrixType::ScalarType sliceX, typename MatrixType::ScalarType sliceY, typename MatrixType::ScalarType sliceDim, char dir);
};

#include "util.hpp"
#endif /*UTIL_H_*/
