/* Author: Edward Hutter */

#ifndef MATRIX_STRUCTURE_H_
#define MATRIX_STRUCTURE_H_

// These class policies implement the Structure Policy

class rect{
public:
  template<typename T, typename U>
  static void _assemble(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY);
  template<typename T, typename U>
  static void _assemble_matrix(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, U dimensionX, U dimensionY);
  template<typename T>
  static void _dissamble(std::vector<T*>& matrix);
  template<typename T, typename U>
  static void _copy(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, T* const & source, U dimensionX, U dimensionY);
  template<typename T, typename U>
  static void _print(const std::vector<T*>& matrix, U dimensionX, U dimensionY);
  template<typename U>
  static inline U _num_elems(U rangeX, U rangeY) { return rangeX*rangeY; }
  template<typename T, typename U>
  void _distribute_identity(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, int64_t localPgridDimX,
                            int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, T val);
  template<typename T, typename U>
  static void _distribute_symmetric(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, int64_t localPgridDimX,
                                    int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key, bool diagonallyDominant);
  template<typename T, typename U>
  static void _distribute_random(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, int64_t localPgridDimX,
                                 int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key);
};

class uppertri{
public:
  template<typename T, typename U>
  static void _assemble(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY);
  template<typename T, typename U>
  static void _assemble_matrix(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, U dimensionX, U dimensionY);
  template<typename T>
  static void _dissamble(std::vector<T*>& matrix);
  template<typename T, typename U>
  static void _copy(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, T* const & source, U dimensionX, U dimensionY);
  template<typename T, typename U>
  static void _print(const std::vector<T*>& matrix, U dimensionX, U dimensionY);
  template<typename U>
  static inline U _num_elems(U rangeX, U rangeY) { return ((rangeX*(rangeX+1))>>1); }
  template<typename T, typename U>
  static void _distribute_random(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, int64_t localPgridDimX, int64_t localPgridDimY,
                                 int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key);
};

class lowertri{
public:
  template<typename T, typename U>
  static void _assemble(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY);
  template<typename T, typename U>
  static void _assemble_matrix(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, U dimensionX, U dimensionY);
  template<typename T>
  static void _dissamble(std::vector<T*>& matrix);
  template<typename T, typename U>
  static void _copy(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, T* const & source, U dimensionX, U dimensionY);
  template<typename T, typename U>
  static void _print(const std::vector<T*>& matrix, U dimensionX, U dimensionY);
  template<typename U>
  static inline U _num_elems(U rangeX, U rangeY) { return ((rangeX*(rangeX+1))>>1); }
  template<typename T, typename U>
  static void _distribute_random(std::vector<T*>& matrix, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, int64_t localPgridDimX, int64_t localPgridDimY,
                                 int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key);
};

#include "structure.hpp"

#endif /* MATRIX_STRUCTURE_H_ */
