/* Author: Edward Hutter */

#ifndef UTIL_H_
#define UTIL_H_

class util{
public:
  template<typename T, typename U>
  static std::vector<T> blockedToCyclic(std::vector<T>& blockedData, U localDimensionRows, U localDimensionColumns, size_t sliceDim);
  template<typename T, typename U>
  static std::vector<T> blockedToCyclicSpecial(std::vector<T>& blockedData, U localDimensionRows, U localDimensionColumns, size_t sliceDim, char dir);
  template<typename MatrixType>
  static std::vector<typename MatrixType::ScalarType> getReferenceMatrix(MatrixType& myMatrix, size_t key, MPI_Comm slice, size_t commDim);
  template<typename MatrixType>
  static void transposeSwap(MatrixType& mat, size_t myRank, size_t transposeRank, MPI_Comm comm);
  template<typename U>
  static U getNextPowerOf2(U localShift);
  template<typename MatrixType>
  static void removeTriangle(MatrixType& matrix, size_t sliceX, size_t sliceY, size_t sliceDim, char dir);
  static void processAveragesFromFile(std::ofstream& fptrAvg, std::string& fileStrTotal, size_t numFuncs, size_t numIterations, size_t rank);
  template<typename T>
  static void InitialGEMM();
};

#include "util.hpp"
#endif /*UTIL_H_*/
