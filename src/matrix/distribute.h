/* Author: Edward Hutter */

#ifndef MATRIX_DISTRIBUTER_H_
#define MATRIX_DISTRIBUTER_H_

// Partially specialized class that will allow us to use the partially specialized method.
class cyclic{
public:
  template<typename T, typename U>
  static void _DistributeRandom(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, int64_t localPgridDimX,
    int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key, square);
  template<typename T, typename U>
  static void _DistributeIdentity(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, int64_t localPgridDimX,
    int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, T val);
  template<typename T, typename U>
  static void _DistributeDebug(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, int64_t localPgridDimX,
    int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY);
  template<typename T, typename U>
  static void _DistributeSymmetric(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, int64_t localPgridDimX,
    int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key, bool diagonallyDominant);
  template<typename T, typename U>
  static void _DistributeRandom(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, int64_t globalDimensionY, int64_t localPgridDimX,
    int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key, rect);
  template<typename T, typename U>
  static void _DistributeRandom(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, int64_t globalDimensionY, int64_t localPgridDimX, int64_t localPgridDimY,
    int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key, uppertri);
  template<typename T, typename U>
  static void _DistributeRandom(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, int64_t localPgridDimX, int64_t localPgridDimY,
    int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key, lowertri);
};

// We must provide the template class definitions below. Read up on why exactly we need to do this.
#include "distribute.hpp"

#endif /* MATRIX_DISTRIBUTER_H_ */
