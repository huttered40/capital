/* Author: Edward Hutter */

// #include "Matrix.h"  -> Compiler needs the full definition of the templated class in order to instantiate it.

template<typename T, typename U, template<typename,typename,template<typename,typename> class> class Structure, template<typename, typename> class Distributer>
Matrix<T,U,Structure,Distributer>::Matrix(U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY)
{
  this->_dimensionX = {dimensionX};
  this->_dimensionY = {dimensionY};
  this->_globalDimensionX = {globalDimensionX};
  this->_globalDimensionY = {globalDimensionY};

  Structure<T,U,Distributer>::Assemble(this->_matrix, this->_dimensionX, this->_dimensionY);
  return;
}

template<typename T, typename U, template<typename,typename,template<typename,typename> class> class Structure, template<typename, typename> class Distributer>
Matrix<T,U,Structure,Distributer>::Matrix(const Matrix& rhs)
{
  copy(rhs);
  return;
}

template<typename T, typename U, template<typename,typename,template<typename,typename> class> class Structure, template<typename, typename> class Distributer>
Matrix<T,U,Structure,Distributer>::Matrix(Matrix&& rhs)
{
  // Use std::forward in the future.
  mover(std::move(rhs));
  return;
}

template<typename T, typename U, template<typename,typename,template<typename,typename> class> class Structure, template<typename, typename> class Distributer>
Matrix<T,U,Structure,Distributer>& Matrix<T,U,Structure,Distributer>::operator=(const Matrix& rhs)
{
  if (this != &rhs)
  {
    copy(rhs);
  }
  return *this;
}

template<typename T, typename U, template<typename,typename,template<typename,typename> class> class Structure, template<typename, typename> class Distributer>
Matrix<T,U,Structure,Distributer>& Matrix<T,U,Structure,Distributer>::operator=(Matrix&& rhs)
{
  // Use std::forward in the future.
  if (this != &rhs)
  {
    mover(std::move(rhs));
  }
  return *this;
}

template<typename T, typename U, template<typename,typename,template<typename,typename> class> class Structure, template<typename, typename> class Distributer>
Matrix<T,U,Structure,Distributer>::~Matrix()
{
  Structure<T,U,Distributer>::Dissamble(this->_matrix);
}

template<typename T, typename U, template<typename,typename,template<typename,typename> class> class Structure, template<typename, typename> class Distributer>
void Matrix<T,U,Structure,Distributer>::copy(const Matrix& rhs)
{
  this->_dimensionX = {rhs._dimensionX};
  this->_dimensionY = {rhs._dimensionY};
  Structure<T,U,Distributer>::Copy(this->_matrix, rhs._matrix, this->_dimensionX, this->_dimensionY);
  return;
}

template<typename T, typename U, template<typename,typename,template<typename,typename> class> class Structure, template<typename, typename> class Distributer>
void Matrix<T,U,Structure,Distributer>::mover(Matrix&& rhs)
{
  this->_dimensionX = {rhs._dimensionX};
  this->_dimensionY = {rhs._dimensionY};
  // Suck out the matrix member from rhs and stick it in our field.
  // For now, we don't need a Allocator Policy Interface Move() method.
  this->_matrix = std::move(rhs._matrix); 
  return;
}

template<typename T, typename U, template<typename,typename,template<typename,typename> class> class Structure, template<typename, typename> class Distributer>
void Matrix<T,U,Structure,Distributer>::Serialize(const Matrix& rhs)
{
  // call a MatrixSerialize protected static method using the member variabe _matrix, NOT Matrix,
  //   since we don't to create a circular definition.
}

template<typename T, typename U, template<typename,typename,template<typename,typename> class> class Structure, template<typename, typename> class Distributer>
void Matrix<T,U,Structure,Distributer>::Serialize(Matrix&& rhs)
{
  // call a MatrixSerialize protected static method using the member variabe _matrix, NOT Matrix,
  //   since we don't to create a circular definition.
}

template<typename T, typename U, template<typename,typename,template<typename,typename> class> class Structure, template<typename, typename> class Distributer>
void Matrix<T,U,Structure,Distributer>::Distribute(int localPgridX, int localPgridY, int globalPgridX, int globalPgridY)
{
  // This is a 2-level Policy-class trick due to the lack of orthogonality between the
  //   Structure Policy and the Distributer Policy.
  
  Structure<T,U,Distributer>::Distribute(this->_matrix, this->_dimensionX, this->_dimensionY, localPgridX, localPgridY, globalPgridX, globalPgridY);
}

template<typename T, typename U, template<typename,typename,template<typename,typename> class> class Structure, template<typename, typename> class Distributer>
void Matrix<T,U,Structure,Distributer>::print() const
{
  Structure<T,U,Distributer>::Print(this->_matrix, this->_dimensionX, this->_dimensionY);
}
