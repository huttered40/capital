/* Author: Edward Hutter */

// #include "Matrix.h"  -> Compiler needs the full definition of the templated class in order to instantiate it.

template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class Structure, template<typename, typename,int> class Distributer>
Matrix<T,U,Structure,Distributer>::Matrix(U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY)
{
  this->_dimensionX = {dimensionX};
  this->_dimensionY = {dimensionY};
  this->_globalDimensionX = {globalDimensionX};
  this->_globalDimensionY = {globalDimensionY};

  Structure<T,U,Distributer>::Assemble(this->_matrix, this->_numElems, this->_dimensionX, this->_dimensionY);
  return;
}

template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class Structure, template<typename, typename,int> class Distributer>
Matrix<T,U,Structure,Distributer>::Matrix(const Matrix& rhs)
{
  copy(rhs);
  return;
}

template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class Structure, template<typename, typename,int> class Distributer>
Matrix<T,U,Structure,Distributer>::Matrix(Matrix&& rhs)
{
  // Use std::forward in the future.
  mover(std::move(rhs));
  return;
}

template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class Structure, template<typename, typename,int> class Distributer>
Matrix<T,U,Structure,Distributer>& Matrix<T,U,Structure,Distributer>::operator=(const Matrix& rhs)
{
  if (this != &rhs)
  {
    copy(rhs);
  }
  return *this;
}

template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class Structure, template<typename, typename,int> class Distributer>
Matrix<T,U,Structure,Distributer>& Matrix<T,U,Structure,Distributer>::operator=(Matrix&& rhs)
{
  // Use std::forward in the future.
  if (this != &rhs)
  {
    mover(std::move(rhs));
  }
  return *this;
}

template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class Structure, template<typename, typename,int> class Distributer>
Matrix<T,U,Structure,Distributer>::~Matrix()
{
  Structure<T,U,Distributer>::Dissamble(this->_matrix);
}

template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class Structure, template<typename, typename,int> class Distributer>
void Matrix<T,U,Structure,Distributer>::copy(const Matrix& rhs)
{
  this->_dimensionX = {rhs._dimensionX};
  this->_dimensionY = {rhs._dimensionY};
  this->_numElems = {rhs._numElems};
  this->_globalDimensionX = {rhs._globalDimensionX};
  this->_globalDimensionY = {rhs._globalDimensionY};
  Structure<T,U,Distributer>::Copy(this->_matrix, rhs._matrix, this->_dimensionX, this->_dimensionY);
  return;
}

template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class Structure, template<typename, typename,int> class Distributer>
void Matrix<T,U,Structure,Distributer>::mover(Matrix&& rhs)
{
  this->_dimensionX = {rhs._dimensionX};
  this->_dimensionY = {rhs._dimensionY};
  this->_numElems = {rhs._numElems};
  this->_globalDimensionX = {rhs._globalDimensionX};
  this->_globalDimensionY = {rhs._globalDimensionY};
  // Suck out the matrix member from rhs and stick it in our field.
  // For now, we don't need a Allocator Policy Interface Move() method.
  this->_matrix = std::move(rhs._matrix); 
  return;
}

template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class Structure, template<typename, typename,int> class Distributer>
template<template<typename,typename,template<typename,typename,int> class> class StructureDest>
void Matrix<T,U,Structure,Distributer>::Serialize(Matrix<T,U,StructureDest,Distributer>& dest)
{
  // Matrix must be already cnstructed with memory. Add a check for this later.

  // So we are not going through the Structure Policy here like we did with the Distributer Policy.
  // Maybe that is something I should change later about the Distributer Policy implementation if this works correctly.
  
  // Note: I cannot just use the private members of dest because it is actually a different Type due to its
  //         different template parameters. This is a problem. I can implement a public getMatrix() method, and hav
  std::vector<T*>& saveDest = dest.getData();	// Should incur no copy. A reference to the inner vector has been given.
  T*& save = saveDest[0];
  T*& saveSrc = this->_matrix[0];
  Serializer<T,U,Structure,StructureDest>::Serialize(saveSrc, save, this->_dimensionX, this->_dimensionY);
}

template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class Structure, template<typename, typename,int> class Distributer>
void Matrix<T,U,Structure,Distributer>::Distribute(int localPgridX, int localPgridY, int globalPgridX, int globalPgridY)
{
  // Matrix must be already cnstructed with memory. Add a check for this later.

  // This is a 2-level Policy-class trick due to the lack of orthogonality between the
  //   Structure Policy and the Distributer Policy.
  
  Structure<T,U,Distributer>::Distribute(this->_matrix, this->_dimensionX, this->_dimensionY, this->_globalDimensionX, this->_globalDimensionY, localPgridX, localPgridY, globalPgridX, globalPgridY);
}

template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class Structure, template<typename, typename,int> class Distributer>
void Matrix<T,U,Structure,Distributer>::print() const
{
  Structure<T,U,Distributer>::Print(this->_matrix, this->_dimensionX, this->_dimensionY);
}
