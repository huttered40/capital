/* Author: Edward Hutter */

// #include "Matrix.h"  -> Compiler needs the full definition of the templated class in order to instantiate it.

template<typename T, typename U, template<typename,typename> class Allocator, template<typename, typename, typename> class Distributer>
Matrix<T,U,Allocator,Distributer>::Matrix(U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY)
{
  // In the future, I would like the Allocator to do all this work. But for now, its ok
  // I need to read Chapter 4 in Modern C++ Design before I write my own allocators.
	
  this->_dimensionX = {dimensionX};
  this->_dimensionY = {dimensionY};
  this->_globalDimensionX = {globalDimensionX};
  this->_globalDimensionY = {globalDimensionY};

  Allocator<T,U>::Assemble(this->_matrix, this->_dimensionX, this->_dimensionY);
  return;
  
}

template<typename T, typename U, template<typename,typename> class Allocator, template<typename, typename, typename> class Distributer>
Matrix<T,U,Allocator,Distributer>::Matrix(const Matrix& rhs)
{
  copy(rhs);
  return;
}

template<typename T, typename U, template<typename,typename> class Allocator, template<typename, typename, typename> class Distributer>
Matrix<T,U,Allocator,Distributer>::Matrix(Matrix&& rhs)
{
  // Use std::forward in the future.
  mover(std::move(rhs));
  return;
}

template<typename T, typename U, template<typename,typename> class Allocator, template<typename, typename, typename> class Distributer>
Matrix<T,U,Allocator,Distributer>& Matrix<T,U,Allocator,Distributer>::operator=(const Matrix& rhs)
{
  if (this != &rhs)
  {
    copy(rhs);
  }  
  return *this;
}

template<typename T, typename U, template<typename,typename> class Allocator, template<typename, typename, typename> class Distributer>
Matrix<T,U,Allocator,Distributer>& Matrix<T,U,Allocator,Distributer>::operator=(Matrix&& rhs)
{
  // Use std::forward in the future.
  if (this != &rhs)
  {
    mover(std::move(rhs));
  }
  return *this;
}

template<typename T, typename U, template<typename,typename> class Allocator, template<typename, typename, typename> class Distributer>
Matrix<T,U,Allocator,Distributer>::~Matrix()
{
  Allocator<T,U>::Dissamble(this->_matrix);
}

template<typename T, typename U, template<typename,typename> class Allocator, template<typename, typename, typename> class Distributer>
void Matrix<T,U,Allocator,Distributer>::copy(const Matrix& rhs)
{
  this->_dimensionX = {rhs._dimensionX};
  this->_dimensionY = {rhs._dimensionY};
  Allocator<T,U>::Copy(this->_matrix, rhs._matrix, this->_dimensionX, this->_dimensionY);
  return;
}

template<typename T, typename U, template<typename,typename> class Allocator, template<typename, typename, typename> class Distributer>
void Matrix<T,U,Allocator,Distributer>::mover(Matrix&& rhs)
{
  this->_dimensionX = {rhs._dimensionX};
  this->_dimensionY = {rhs._dimensionY};
  // Suck out the matrix member from rhs and stick it in our field.
  // For now, we don't need a Allocator Policy Interface Move() method.
  this->_matrix = std::move(rhs._matrix); 
  return;
}

template<typename T, typename U, template<typename,typename> class Allocator, template<typename, typename, typename> class Distributer>
void Matrix<T,U,Allocator,Distributer>::Serialize(const Matrix& rhs)
{
  // call a MatrixSerialize protected static method using the member variabe _matrix, NOT Matrix,
  //   since we don't to create a circular definition.
}

template<typename T, typename U, template<typename,typename> class Allocator, template<typename, typename, typename> class Distributer>
void Matrix<T,U,Allocator,Distributer>::Serialize(Matrix&& rhs)
{
  // call a MatrixSerialize protected static method using the member variabe _matrix, NOT Matrix,
  //   since we don't to create a circular definition.
}

template<typename T, typename U, template<typename,typename> class Allocator, template<typename, typename, typename> class Distributer>
void Matrix<T,U,Allocator,Distributer>::Distribute(int localPgridX, int localPgridY, int globalPgridX, int globalPgridY)
{
  // call a MatrixSerialize protected static method
  // We may want to change vector<T*> to something more general later on.
  Distributer<T,U,std::vector<T*>>::Distribute(this->_matrix, this->_dimensionX, this->_dimensionY, localPgridX, localPgridY, globalPgridX, globalPgridY);
}

template<typename T, typename U, template<typename,typename> class Allocator, template<typename, typename, typename> class Distributer>
void Matrix<T,U,Allocator,Distributer>::print() const
{
  Allocator<T,U>::Print(this->_matrix, this->_dimensionX, this->_dimensionY);
}
