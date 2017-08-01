/* Author: Edward Hutter */

// #include "Matrix.h"  -> Compiler needs the full definition of the templated class in order to instantiate it.

template<typename T, typename U>
Matrix<T,U>::Matrix()
{
  // I want to prevent this entirely. Need to read into = delete() and = default
  // Maybe I just dont provide a templated member function for it? Aren't there corner cases here?
}

template<typename T, typename U>
Matrix<T,U>::Matrix(const Matrix& rhs)
{
  // how can we prevent certain types T??????
  this->_dimensionX{rhs._dimensionX};
  this->_dimensionY{rhs._dimensionY};
  this->_matrix.resize(rhs_.dimensionX);
  this->_data.resize(rhs._dimensionX * rhs._dimensionY);	// give a default value for whatever T is?
  
  U offset{0};
  for (auto& ptr : this->_matrix)
  {
    ptr = &this->_matrix[offset];
    offset += rhs.dimensionY;
  }
  return;
}

template<typename T, typename U>
Matrix<T,U>::Matrix(Matrix&& rhs)
{
  // lets guess how this works, and then look it up formally later.
}

template<typename T, typename U>
Matrix& Matrix<T,U>::operator=(const Matrix& rhs)
{
}

template<typename T, typename U>
Matrix& Matrix<T,U>::operator=(Matrix&& rhs)
{
}

template<typename T, typename U>
Matrix<T,U>::~Matrix()
{
  // I could manually clear the data structures here, but its not necessary
}

