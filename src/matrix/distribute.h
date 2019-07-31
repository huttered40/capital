/* Author: Edward Hutter */

#ifndef MATRIX_DISTRIBUTER_H_
#define MATRIX_DISTRIBUTER_H_

// Partially specialized class that will allow us to use the partially specialized method.
class cyclic{
public:
  template<typename T, typename U>
  static void _DistributeRandom(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, size_t localPgridDimX,
    size_t localPgridDimY, size_t globalPgridDimX, size_t globalPgridDimY, size_t key, Square);
  template<typename T, typename U>
  static void _DistributeSymmetric(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, size_t localPgridDimX,
    size_t localPgridDimY, size_t globalPgridDimX, size_t globalPgridDimY, size_t key, bool diagonallyDominant);
  template<typename T, typename U>
  static void _DistributeRandom(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, size_t globalDimensionY, size_t localPgridDimX,
    size_t localPgridDimY, size_t globalPgridDimX, size_t globalPgridDimY, size_t key, Rectangular);
  template<typename T, typename U>
  static void _DistributeRandom(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, size_t globalDimensionY, size_t localPgridDimX, size_t localPgridDimY,
    size_t globalPgridDimX, size_t globalPgridDimY, size_t key, UpperTriangular);
  template<typename T, typename U>
  static void _DistributeRandom(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, size_t localPgridDimX, size_t localPgridDimY,
    size_t globalPgridDimX, size_t globalPgridDimY, size_t key, LowerTriangular);
};

// We must provide the template class definitions below. Read up on why exactly we need to do this.
#include "distribute.hpp"

#endif /* MATRIX_DISTRIBUTER_H_ */
