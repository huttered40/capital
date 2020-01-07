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

  template<typename ScalarType, typename DimensionType>
  static void block_to_cyclic(std::vector<ScalarType>& blockedData, ScalarType* cyclicData, DimensionType localDimensionRows, DimensionType localDimensionColumns, int64_t sliceDim, char dir);

  template<typename ScalarType, typename DimensionType>
  static void block_to_cyclic(ScalarType* blockedData, ScalarType* cyclicData, DimensionType localDimensionRows, DimensionType localDimensionColumns, int64_t sliceDim);

  template<typename ScalarType, typename DimensionType>
  static void cyclic_to_local(ScalarType* storeScalarType, ScalarType* storeScalarTypeI, DimensionType localDimension, DimensionType globalDimension, DimensionType bcDimension, int64_t sliceDim, int64_t rankSlice);

  template<typename MatrixType, typename CommType>
  static void transpose(MatrixType& mat, CommType&& CommInfo);

  template<typename DimensionType>
  static DimensionType get_next_power2(DimensionType localShift);

  template<typename MatrixType>
  static void remove_triangle(MatrixType& matrix, int64_t sliceX, int64_t sliceY, int64_t sliceDim, char dir);

  template<typename MatrixType>
  static void remove_triangle_local(MatrixType& matrix, int64_t sliceX, int64_t sliceY, int64_t sliceDim, char dir);

  template<typename ScalarType, typename DimensionType>
  static void random_fill(ScalarType* A, DimensionType dimensionX, DimensionType dimensionY, DimensionType globalDimensionX, int64_t globalDimensionY, int64_t localPgridDimX,
                          int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key);

  template<typename ScalarType, typename DimensionType>
  static void random_fill_symmetric(ScalarType* A, DimensionType dimensionX, DimensionType dimensionY, DimensionType globalDimensionX, DimensionType globalDimensionY, int64_t localPgridDimX,
                                    int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key);
};

#include "util.hpp"
#endif /*UTIL_H_*/
