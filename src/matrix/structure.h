/* Author: Edward Hutter */

#ifndef MATRIX_STRUCTURE_H_
#define MATRIX_STRUCTURE_H_

// These class policies implement the Structure Policy

class rect{
public:
  template<typename ScalarType, typename DimensionType>
  static void _assemble(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, std::vector<ScalarType*>& matrix, DimensionType& matrixNumElems, DimensionType dimensionX, DimensionType dimensionY);
  template<typename ScalarType, typename DimensionType>
  static void _assemble_matrix(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY);
  template<typename ScalarType>
  static void _dissamble(std::vector<ScalarType*>& matrix);
  template<typename ScalarType, typename DimensionType>
  static void _copy(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, std::vector<ScalarType*>& matrix, ScalarType* const & source, DimensionType dimensionX, DimensionType dimensionY);
  template<typename ScalarType, typename DimensionType>
  static void _print(const std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY);
  template<typename DimensionType>
  static inline DimensionType _num_elems(DimensionType rangeX, DimensionType rangeY) { return rangeX*rangeY; }
  template<typename ScalarType, typename DimensionType>
  void _distribute_identity(std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY, DimensionType globalDimensionX, DimensionType globalDimensionY, int64_t localPgridDimX,
                            int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, ScalarType val);
  template<typename ScalarType, typename DimensionType>
  static void _distribute_symmetric(std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY, DimensionType globalDimensionX, DimensionType globalDimensionY, int64_t localPgridDimX,
                                    int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key, bool diagonallyDominant);
  template<typename ScalarType, typename DimensionType>
  static void _distribute_random(std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY, DimensionType globalDimensionX, DimensionType globalDimensionY, int64_t localPgridDimX,
                                 int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key);
};

class uppertri{
public:
  template<typename ScalarType, typename DimensionType>
  static void _assemble(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, std::vector<ScalarType*>& matrix, DimensionType& matrixNumElems, DimensionType dimensionX, DimensionType dimensionY);
  template<typename ScalarType, typename DimensionType>
  static void _assemble_matrix(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY);
  template<typename ScalarType>
  static void _dissamble(std::vector<ScalarType*>& matrix);
  template<typename ScalarType, typename DimensionType>
  static void _copy(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, std::vector<ScalarType*>& matrix, ScalarType* const & source, DimensionType dimensionX, DimensionType dimensionY);
  template<typename ScalarType, typename DimensionType>
  static void _print(const std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY);
  template<typename DimensionType>
  static inline DimensionType _num_elems(DimensionType rangeX, DimensionType rangeY) { return ((rangeX*(rangeX+1))>>1); }
  template<typename ScalarType, typename DimensionType>
  static void _distribute_random(std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY, DimensionType globalDimensionX, DimensionType globalDimensionY, int64_t localPgridDimX, int64_t localPgridDimY,
                                 int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key);
};

class lowertri{
public:
  template<typename ScalarType, typename DimensionType>
  static void _assemble(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, std::vector<ScalarType*>& matrix, DimensionType& matrixNumElems, DimensionType dimensionX, DimensionType dimensionY);
  template<typename ScalarType, typename DimensionType>
  static void _assemble_matrix(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY);
  template<typename ScalarType>
  static void _dissamble(std::vector<ScalarType*>& matrix);
  template<typename ScalarType, typename DimensionType>
  static void _copy(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, std::vector<ScalarType*>& matrix, ScalarType* const & source, DimensionType dimensionX, DimensionType dimensionY);
  template<typename ScalarType, typename DimensionType>
  static void _print(const std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY);
  template<typename DimensionType>
  static inline DimensionType _num_elems(DimensionType rangeX, DimensionType rangeY) { return ((rangeX*(rangeX+1))>>1); }
  template<typename ScalarType, typename DimensionType>
  static void _distribute_random(std::vector<ScalarType*>& matrix, DimensionType dimensionX, DimensionType dimensionY, DimensionType globalDimensionX, DimensionType globalDimensionY, int64_t localPgridDimX, int64_t localPgridDimY,
                                 int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key);
};

#include "structure.hpp"

#endif /* MATRIX_STRUCTURE_H_ */
