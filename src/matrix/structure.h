/* Author: Edward Hutter */

#ifndef MATRIX_STRUCTURE_H_
#define MATRIX_STRUCTURE_H_

// These class policies implement the Structure Policy

class rect{
public:
  template<typename DimensionType>
  static inline DimensionType _num_elems(DimensionType rangeX, DimensionType rangeY) { return rangeX*rangeY; }
  template<typename DimensionType>
  static inline DimensionType _offset(DimensionType coordX, DimensionType coordY, DimensionType dimX, DimensionType dimY) { return coordX*dimY+coordY; }
  template<typename ScalarType, typename DimensionType>
  static void _print(const ScalarType* data, DimensionType dimensionX, DimensionType dimensionY);
protected:
  template<typename ScalarType, typename DimensionType>
  static void _assemble(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, DimensionType& matrixNumElems, DimensionType dimensionX, DimensionType dimensionY);
  template<typename ScalarType, typename DimensionType>
  static void _assemble_matrix(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, DimensionType dimensionX, DimensionType dimensionY);
  template<typename ScalarType, typename DimensionType>
  static void _copy(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, ScalarType* const & source, DimensionType dimensionX, DimensionType dimensionY);
  template<typename ScalarType, typename DimensionType>
  void _distribute_identity(ScalarType* data, DimensionType dimensionX, DimensionType dimensionY, DimensionType globalDimensionX, DimensionType globalDimensionY, int64_t localPgridDimX,
                            int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, ScalarType val);
  template<typename ScalarType, typename DimensionType>
  static void _distribute_symmetric(ScalarType* data, DimensionType dimensionX, DimensionType dimensionY, DimensionType globalDimensionX, DimensionType globalDimensionY, int64_t localPgridDimX,
                                    int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key, bool diagonallyDominant);
  template<typename ScalarType, typename DimensionType>
  static void _distribute_random(ScalarType* data, DimensionType dimensionX, DimensionType dimensionY, DimensionType globalDimensionX, DimensionType globalDimensionY, int64_t localPgridDimX,
                                 int64_t localPgridDimY, int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key);
};

class uppertri{
public:
  template<typename DimensionType>
  static inline DimensionType _num_elems(DimensionType rangeX, DimensionType rangeY) { return ((rangeX*(rangeX+1))>>1); }
  template<typename DimensionType>
  static inline DimensionType _offset(DimensionType coordX, DimensionType coordY, DimensionType dimX, DimensionType dimY) { return ((coordX*(coordX+1))>>1)+coordY; }
  template<typename ScalarType, typename DimensionType>
  static void _print(const ScalarType* data, DimensionType dimensionX, DimensionType dimensionY);
protected:
  template<typename ScalarType, typename DimensionType>
  static void _assemble(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, DimensionType& matrixNumElems, DimensionType dimensionX, DimensionType dimensionY);
  template<typename ScalarType, typename DimensionType>
  static void _assemble_matrix(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, DimensionType dimensionX, DimensionType dimensionY);
  template<typename ScalarType, typename DimensionType>
  static void _copy(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, ScalarType* const & source, DimensionType dimensionX, DimensionType dimensionY);
  template<typename ScalarType, typename DimensionType>
  static void _distribute_random(ScalarType* data, DimensionType dimensionX, DimensionType dimensionY, DimensionType globalDimensionX, DimensionType globalDimensionY, int64_t localPgridDimX, int64_t localPgridDimY,
                                 int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key);
};

class lowertri{
public:
  template<typename DimensionType>
  static inline DimensionType _num_elems(DimensionType rangeX, DimensionType rangeY) { return ((rangeX*(rangeX+1))>>1); }
  template<typename DimensionType>
  static inline DimensionType _offset(DimensionType coordX, DimensionType coordY, DimensionType dimX, DimensionType dimY) { return coordX==0 ? coordY : (coordX*dimY-(coordX-1))+(coordY-coordX); }
  template<typename ScalarType, typename DimensionType>
  static void _print(const ScalarType* data, DimensionType dimensionX, DimensionType dimensionY);
protected:
  template<typename ScalarType, typename DimensionType>
  static void _assemble(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, DimensionType& matrixNumElems, DimensionType dimensionX, DimensionType dimensionY);
  template<typename ScalarType, typename DimensionType>
  static void _assemble_matrix(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, DimensionType dimensionX, DimensionType dimensionY);
  template<typename ScalarType, typename DimensionType>
  static void _copy(ScalarType*& data, ScalarType*& scratch, ScalarType*& pad, ScalarType* const & source, DimensionType dimensionX, DimensionType dimensionY);
  template<typename ScalarType, typename DimensionType>
  static void _distribute_random(ScalarType* data, DimensionType dimensionX, DimensionType dimensionY, DimensionType globalDimensionX, DimensionType globalDimensionY, int64_t localPgridDimX, int64_t localPgridDimY,
                                 int64_t globalPgridDimX, int64_t globalPgridDimY, int64_t key);
};

#include "structure.hpp"

#endif /* MATRIX_STRUCTURE_H_ */
