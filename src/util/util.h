/* Author: Edward Hutter */

#ifndef UTIL_H_
#define UTIL_H_

class util{
public:
  template<typename MatrixType, typename RefMatrixType, typename LambdaType>
  static typename MatrixType::ScalarType
    residual_local(MatrixType& Matrix, RefMatrixType& RefMatrix, LambdaType&& Lambda, MPI_Comm slice, size_t sliceX, size_t sliceY, size_t sliceDimX, size_t sliceDimY);
  template<typename T, typename U>
  static std::vector<T> blocked_to_cyclic(std::vector<T>& blockedData, U localDimensionRows, U localDimensionColumns, size_t sliceDim);
  template<typename T, typename U>
  static std::vector<T> blocked_to_cyclic_special(std::vector<T>& blockedData, U localDimensionRows, U localDimensionColumns, size_t sliceDim, char dir);
  template<typename MatrixType>
  static std::vector<typename MatrixType::ScalarType> get_reference_matrix(MatrixType& myMatrix, size_t key, MPI_Comm slice, size_t commDim);
  template<typename MatrixType, typename CommType>
  static void transpose_swap(MatrixType& mat, CommType&& CommInfo);
  template<typename U>
  static U get_next_power2(U localShift);
  template<typename MatrixType>
  static void remove_triangle(MatrixType& matrix, size_t sliceX, size_t sliceY, size_t sliceDim, char dir);
  static void processAveragesFromFile(std::ofstream& fptrAvg, std::string& fileStrTotal, size_t numFuncs, size_t numIterations, size_t rank);
};

#include "util.hpp"
#endif /*UTIL_H_*/
