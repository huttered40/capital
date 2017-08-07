/* Author: Edward Hutter */

// #include "Matrix.h"  -> Compiler needs the full definition of the templated class in order to instantiate it.

template<typename T, typename U>
Matrix<T,U>::Matrix()
{
  // I want to prevent this entirely. Need to read into = delete() and = default
  // Maybe I just dont provide a templated member function for it? Aren't there corner cases here?
}

template<typename T, typename U>
Matrix<T,U>::Matrix(U dimensionX, U dimensionY)
{
  // how can we prevent certain types T??????
  this->_dimensionX = {dimensionX};
  this->_dimensionY = {dimensionY};
  this->_matrix.resize(dimensionX);
  this->_matrix[0] = new T[dimensionX * dimensionY];
  
  U offset{0};
  for (auto& ptr : this->_matrix)
  {
    ptr = &this->_matrix[offset];
    offset += dimensionY;
  }
  return;
  
}

template<typename T, typename U>
Matrix<T,U>::Matrix(const Matrix& rhs)
{
  // how can we prevent certain types T??????
  
  copy(rhs);
  return;
}

template<typename T, typename U>
Matrix<T,U>::Matrix(Matrix&& rhs)
{
  // Use std::forward in the future.
  mover(std::move(rhs));
  return;
}

template<typename T, typename U>
Matrix<T,U>& Matrix<T,U>::operator=(const Matrix& rhs)
{
  if (this != &rhs)
  {
    copy(rhs);
  }  
  return *this;
}

template<typename T, typename U>
Matrix<T,U>& Matrix<T,U>::operator=(Matrix&& rhs)
{
  // Use std::forward in the future.
  if (this != &rhs)
  {
    mover(std::move(rhs));
  }
  return *this;
}

template<typename T, typename U>
Matrix<T,U>::~Matrix()
{
  if ((this->_matrix.size() > 0) && (this->_matrix[0] != nullptr))
  {
    delete[] this->_matrix[0];
  }
}

template<typename T, typename U>
void Matrix<T,U>::copy(const Matrix& rhs)
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

template<typename T, typename U>
void Matrix<T,U>::mover(Matrix&& rhs)
{
  this->_dimensionX = {rhs._dimensionX};
  this->_dimensionY = {rhs._dimensionY};
  // Suck out the matrix member from rhs and stick it in our field.
  this->_matrix = std::move(rhs._matrix); 
  return;
}

template<typename T, typename U>
void Matrix<T,U>::serialize(const Matrix& rhs)
{}

template<typename T, typename U>
void Matrix<T,U>::serialize(Matrix&& rhs)
{}

