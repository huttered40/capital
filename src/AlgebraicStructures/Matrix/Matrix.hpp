/* Author: Edward Hutter */

// #include "Matrix.h"  -> Compiler needs the full definition of the templated class in order to instantiate it.

template<typename T, typename U, class Allocator, class Distributer>
Matrix<T,U,Allocator,Distributer>::Matrix(U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY)
{
  // how can we prevent certain types T??????
  this->_dimensionX = {dimensionX};
  this->_dimensionY = {dimensionY};
  this->_globalDimensionX = {globalDimensionX};
  this->_globalDimensionY = {globalDimensionY};
  this->_matrix.resize(dimensionX);
  this->_matrix[0] = new T[dimensionX * dimensionY];
  
  U offset{0};
  for (auto& ptr : this->_matrix)
  {
    ptr = &this->_matrix[0][offset];
    offset += dimensionY;
  }
  return;
  
}

template<typename T, typename U, class Allocator, class Distributer>
Matrix<T,U,Allocator,Distributer>::Matrix(const Matrix& rhs)
{
  // how can we prevent certain types T??????
  
  copy(rhs);
  return;
}

template<typename T, typename U, class Allocator, class Distributer>
Matrix<T,U,Allocator,Distributer>::Matrix(Matrix&& rhs)
{
  // Use std::forward in the future.
  mover(std::move(rhs));
  return;
}

template<typename T, typename U, class Allocator, class Distributer>
Matrix<T,U,Allocator,Distributer>& Matrix<T,U,Allocator,Distributer>::operator=(const Matrix& rhs)
{
  if (this != &rhs)
  {
    copy(rhs);
  }  
  return *this;
}

template<typename T, typename U, class Allocator, class Distributer>
Matrix<T,U,Allocator,Distributer>& Matrix<T,U,Allocator,Distributer>::operator=(Matrix&& rhs)
{
  // Use std::forward in the future.
  if (this != &rhs)
  {
    mover(std::move(rhs));
  }
  return *this;
}

template<typename T, typename U, class Allocator, class Distributer>
Matrix<T,U,Allocator,Distributer>::~Matrix()
{
  if ((this->_matrix.size() > 0) && (this->_matrix[0] != nullptr))
  {
    delete[] this->_matrix[0];
  }
}

template<typename T, typename U, class Allocator, class Distributer>
void Matrix<T,U,Allocator,Distributer>::copy(const Matrix& rhs)
{
  this->_dimensionX = {rhs._dimensionX};
  this->_dimensionY = {rhs._dimensionY};
  this->_matrix.resize(rhs._dimensionX);
  
  // Deep copy is needed with the current implementation
  this->_matrix[0] = new T[rhs._dimensionX * rhs._dimensionY];
  
  U offset{0};
  for (auto& ptr : this->_matrix)
  {
    ptr = this->_matrix[offset];
    offset += rhs._dimensionY;
  }
  return;
}

template<typename T, typename U, class Allocator, class Distributer>
void Matrix<T,U,Allocator,Distributer>::mover(Matrix&& rhs)
{
  this->_dimensionX = {rhs._dimensionX};
  this->_dimensionY = {rhs._dimensionY};
  // Suck out the matrix member from rhs and stick it in our field.
  this->_matrix = std::move(rhs._matrix); 
  return;
}

template<typename T, typename U, class Allocator, class Distributer>
void Matrix<T,U,Allocator,Distributer>::Serialize(const Matrix& rhs)
{
  // call a MatrixSerialize protected static method using the member variabe _matrix, NOT Matrix,
  //   since we don't to create a circular definition.
}

template<typename T, typename U, class Allocator, class Distributer>
void Matrix<T,U,Allocator,Distributer>::Serialize(Matrix&& rhs)
{
  // call a MatrixSerialize protected static method using the member variabe _matrix, NOT Matrix,
  //   since we don't to create a circular definition.
}

template<typename T, typename U, class Allocator, class Distributer>
void Matrix<T,U,Allocator,Distributer>::Distribute()
{
  // call a MatrixSerialize protected static method
}

template<typename T, typename U, class Allocator, class Distributer>
void Matrix<T,U,Allocator,Distributer>::print()
{
  // just regular print of the local matrix.
}
