/* Author: Edward Hutter */

#ifndef MATRIX_H_
#define MATRIX_H_

// Local includes -- the policy classes
#include "structure.h"
#include "distributer.h"

template<typename T, typename U = size_t , typename StructurePolicy = Rectangular, typename DistributionPolicy = Cyclic, typename OffloadPolicy = OffloadEachGemm>
class matrix : public StructurePolicy, DistributionPolicy{
public:
  // Type traits (some inherited from matrixBase)
  using ScalarType = T;
  using DimensionType = U;
  using StructureType = StructurePolicy;
  using DistributionType = DistributionPolicy;
  using OffloadType = OffloadPolicy;

  explicit matrix() = delete;
  explicit matrix(U globalDimensionX, U globalDimensionY, size_t globalPgridX, size_t globalPgridY);				// Regular constructor
  explicit matrix(std::vector<T>&& data, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, bool assemble = false);	// Injection constructor
  matrix(const matrix& rhs);
  matrix(matrix&& rhs);
  matrix& operator=(const matrix& rhs);
  matrix& operator=(matrix&& rhs);
  ~matrix();

  // automatically inlined
  // returning an lvalue by virtue of its reference type -- note: this isnt the safest thing, but it provides better speed. 
  inline T* getRawData() { return &this->_data[0]; }
  inline std::vector<T>& getVectorData() { return this->_data; }
  inline std::vector<T*>& getmatrixData() { return this->_matrix;}
  inline U getNumElems() { return this->_numElems; }
  inline U getNumElems(U rangeX, U rangeY) { return StructurePolicy::_getNumElems(rangeX, rangeY); }
  inline U getNumRowsLocal() { return this->_dimensionY; }
  inline U getNumColumnsLocal() { return this->_dimensionX; }
  inline U getNumRowsGlobal() { return this->_globalDimensionY; }
  inline U getNumColumnsGlobal() { return this->_globalDimensionX; }
  
  inline void setNumRowsLocal(U arg) { this->_dimensionY = arg; }
  inline void setNumColumnsLocal(U arg) { this->_dimensionX = arg; }
  inline void setNumRowsGlobal(U arg) { this->_globalDimensionY = arg; }
  inline void setNumColumnsGlobal(U arg) { this->_globalDimensionX = arg; }
  inline void setNumElems(U arg) { this->_numElems = arg; }

  // dim.first -- column index (X) , dim.second -- row index (Y)
  inline T& operator[](const std::pair<U,U>& dim) {return this->_matrix[dim.first][dim.second];}
  inline T& getAccess(U dimX, U dimY) {return this->_matrix[dimX][dimY];}
  void DistributeRandom(size_t localPgridX, size_t localPgridY, size_t globalPgridX, size_t globalPgridY, size_t key);
  void DistributeSymmetric(size_t localPgridX, size_t localPgridY, size_t globalPgridX, size_t globalPgridY, size_t key, bool diagonallyDominant);
  void print() const;

private:
  void copy(const matrix& rhs);
  void mover(matrix&& rhs);		// will need to use std::forward<T> with this I think.

  std::vector<T> _data;
  std::vector<T*> _matrix;		// Holds offsets into the columns of 1D array of data. So matrix[1] is the pointer to the starting address of the 1st column.

  U _numElems;
  U _dimensionX;			// Number of columns owned locally
  U _dimensionY;			// Number of rows owned locally

  // This matrix is most likely a sub-matrix of a global matrix partitioned among many processors.
  U _globalDimensionX;
  U _globalDimensionY;
};

#include "matrix.hpp"

#endif /* MATRIX_H_ */
