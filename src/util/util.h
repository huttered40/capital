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

  template<typename MatrixType>
  static std::vector<typename MatrixType::ScalarType> getReferenceMatrix(MatrixType& myMatrix, int64_t key, MPI_Comm slice, int64_t commDim);

  template<typename MatrixType, typename CommType>
  static void transposeSwap(MatrixType& mat, CommType&& CommInfo);

  template<typename U>
  static U getNextPowerOf2(U localShift);

  template<typename MatrixType>
  static void removeTriangle(MatrixType& matrix, int64_t sliceX, int64_t sliceY, int64_t sliceDim, char dir);

  static void processAveragesFromFile(std::ofstream& fptrAvg, std::string& fileStrTotal, int64_t numFuncs, int64_t numIterations, int64_t rank);
  template<typename T>

  static void InitialGEMM();
};

#include "util.hpp"
#endif /*UTIL_H_*/
