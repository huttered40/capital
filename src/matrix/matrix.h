/* Author: Edward Hutter */

#ifndef MATRIX_H_
#define MATRIX_H_

// Local includes -- the policy classes
#include "structure.h"
#include "distribute.h"

template<typename T, typename U = int64_t, typename StructurePolicy = rect, typename DistributionPolicy = cyclic, typename OffloadPolicy = OffloadEachGemm>
class matrix : public StructurePolicy, DistributionPolicy{
public:
  // Type traits (some inherited from matrixBase)
  using ScalarType = T;
  using DimensionType = U;
  using StructureType = StructurePolicy;
  using DistributionType = DistributionPolicy;
  using OffloadType = OffloadPolicy;

  explicit matrix(){this->danger=true; this->_data=nullptr; this->_scratch=nullptr; this->_pad=nullptr;}// = delete;
  explicit matrix(U globalDimensionX, U globalDimensionY, int64_t globalPgridX, int64_t globalPgridY);	// Regular constructor
  // Injection constructor below assumes data is stored in column-major format
  explicit matrix(T* data, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, U globalPgridX, U globalPgridY);			// Injection constructor
  // Injection constructor below assumes data is nullptr
  explicit matrix(T* data, U dimensionX, U dimensionY, U globalPgridX, U globalPgridY);			// Injection constructor
  matrix(const matrix& rhs);
  matrix(matrix&& rhs);
  matrix& operator=(const matrix& rhs);
  matrix& operator=(matrix&& rhs);
  ~matrix();

  // automatically inlined
  // returning an lvalue by virtue of its reference type -- note: this isnt the safest thing, but it provides better speed. 
  inline T*& data() { return this->_data; }
  inline T* get_data() { T* data = this->_data; this->_data=nullptr; return data; }	// only to be used if internal pointer is needed and instance is never to be used again
  inline T*& scratch() { return this->_scratch; }
  inline T*& pad() { return this->_pad; }
  inline U num_elems() { return this->_numElems; }
  inline U num_elems(U rangeX, U rangeY) { return StructurePolicy::_num_elems(rangeX, rangeY); }
  inline U num_rows_local() { return this->_dimensionY; }
  inline U num_columns_local() { return this->_dimensionX; }
  inline U num_rows_global() { return this->_globalDimensionY; }
  inline U num_columns_global() { return this->_globalDimensionX; }

  inline void swap() { T* ptr = this->data(); this->data() = this->scratch(); this->scratch() = ptr; } 
  inline void swap_pad() { T* ptr = this->scratch(); this->scratch() = this->pad(); this->pad() = ptr; } 

  inline void set_num_rows_local(U arg) { this->_dimensionY = arg; }
  inline void set_num_columns_local(U arg) { this->_dimensionX = arg; }
  inline void set_num_rows_global(U arg) { this->_globalDimensionY = arg; }
  inline void set_num_columns_global(U arg) { this->_globalDimensionX = arg; }
  inline void set_num_elems(U arg) { this->_numElems = arg; }

  // dim.first -- column index (X) , dim.second -- row index (Y)
  void distribute_random(int64_t localPgridX, int64_t localPgridY, int64_t globalPgridX, int64_t globalPgridY, int64_t key);
  void distribute_symmetric(int64_t localPgridX, int64_t localPgridY, int64_t globalPgridX, int64_t globalPgridY, int64_t key, bool diagonallyDominant);
  void distribute_identity(int64_t localPgridX, int64_t localPgridY, int64_t globalPgridX, int64_t globalPgridY, T val=1.);
  void distribute_debug(int64_t localPgridX, int64_t localPgridY, int64_t globalPgridX, int64_t globalPgridY);
  void print() const;
  void print_data() const;
  void print_scratch() const;
  void print_pad() const;

private:
  void copy(const matrix& rhs);
  void mover(matrix&& rhs);		// will need to use std::forward<T> with this I think.

  T* _data;				// Where the matrix data lives as a contiguous 1d array
  T* _scratch;				// Extra storage for summa and other computations that require one2all and all2one communications
  T* _pad;				// Extra storage for uppertri and lowertri structures only used in avoiding extra allocations in summa
  std::vector<T*> _matrix;		// Holds offsets into the columns of 1D array of data. So matrix[1] is the pointer to the starting address of the 1st column.
  bool allocated_data;			// Asks if the raw data was allocated by the user or ourselves
  bool danger;				// notifies me if default constructor was used.

  U _numElems;
  U _dimensionX;			// Number of columns owned locally
  U _dimensionY;			// Number of rows owned locally

  // This matrix is most likely a sub-matrix of a global matrix partitioned among many processors.
  U _globalDimensionX;
  U _globalDimensionY;
};

#include "matrix.hpp"

#endif /* MATRIX_H_ */
