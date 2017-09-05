/* Author: Edward Hutter */

#ifndef MATRIX_H_
#define MATRIX_H_

// System includes
#include <vector>
#include <iostream>

// Local includes -- the policy classes
#include "MatrixStructure.h"
#include "MatrixSerializer.h"

/*
  Divide the Matrix class into policies.
  As of right now, I can identify 2 policies -> creation and distribution
  As explicitey stated in Modern C++ Design, each policiy should be treated as
  its own set of classes, with each different policiy choice called a policy class
  and given its own class.

  I do not want to overload the class template with more than 6 parameters. But I do want
  to give the user of the Matrix class options as to how to build and distribute its data.
  As of right now, the best design choice I can come up with would be to add another template
  parameter to the Matrix class, and write a family of classes for each policy, where each policy class
  can be simple. Each can have just a single static method and take as arguments a reference to a Matrix.
  I choose static methods so that we don't need a class instance in order to use the class, as the class will
  not have any member variables either (only static would work anyways with a single static method).
  
NOTE: Below idea does not work. It has been removed!
  In addition,
  the policy class will be treated as a friend class so that we can make the static method a protected member and prevent
  the user from directly using the class.
*/

// We now use template-template parameters to give the library user a more intuitive interface and to give
//  the library itself more freedome to instantiate particular policy classes with different template parameters.
template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class Structure, template<typename, typename,int> class Distributer>
class Matrix
{

public:
  explicit Matrix() = delete;
  explicit Matrix(U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY);				// Regular constructor
  explicit Matrix(std::vector<T>& data, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY);	// Injection constructor
  Matrix(const Matrix& rhs);
  Matrix(Matrix&& rhs);
  Matrix& operator=(const Matrix& rhs);
  Matrix& operator=(Matrix&& rhs);
  ~Matrix();

  // automatically inlined
  // returning an lvalue by virtue of its reference type -- note: this isnt the safest thing, but it provides better speed. 
  inline T* getRawData() { return this->_matrix[0];}
  inline std::vector<T>& getVectorData() { return this->_data; }
  inline std::vector<T*>& getMatrixData() { return this->_matrix;}

  inline U getNumElems() { return this->_numElems; }
  inline U getNumElems(U rangeX, U rangeY) { return Structure<T,U,Distributer>::getNumElems(rangeX, rangeY); }
  
  inline T& operator[](const std::pair<U,U>& dim) {return this->_matrix[dim.first][dim.second];}
  inline T& getAccess(U dim1, U dim2) {return this->_matrix[dim1][dim2];}

  template<template<typename,typename,template<typename,typename,int> class> class StructureDest>
  void Serialize(Matrix<T,U,StructureDest,Distributer>& dest, bool dir);

  template<template<typename,typename,template<typename,typename,int> class> class StructureDest>
  void Serialize(Matrix<T,U,StructureDest,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir);

  template<template<typename,typename,template<typename,typename,int> class> class StructureDest>
  void Serialize(Matrix<T,U,StructureDest,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros, bool dir);

  // Host method, will call Allocator<T,U,?>::Distribute(...) method
  void DistributeRandom(int localPgridX, int localPgridY, int globalPgridX, int globalPgridY);
  void DistributeSymmetric(int localPgridX, int localPgridY, int globalPgridX, int globalPgridY, bool diagonallyDominant);

  // Just a local print. For a distributed print, we must implement another class that takes the Matrix and operates on it.
  //   That is not something that this class policy needs to worry about.
  void print() const;

  // create my own allocator class using the template parameter? Yes, I want to do this in the future.

private:

  void copy(const Matrix& rhs);
  void mover(Matrix&& rhs);		// will need to use std::forward<T> with this I think.

  std::vector<T> _data;
  std::vector<T*> _matrix;

  U _numElems;
  U _dimensionX;
  U _dimensionY;

  // This Matrix is most likely a sub-matrix of a global Matrix partitioned among many processors.
  U _globalDimensionX;
  U _globalDimensionY;
};

#include "Matrix.hpp"

#endif /* MATRIX_H_ */
