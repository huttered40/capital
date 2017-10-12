/* Author: Edward Hutter */

// #include "Matrix.h"  -> Compiler needs the full definition of the templated class in order to instantiate it.

template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class Structure, template<typename, typename,int> class Distributer>
Matrix<T,U,Structure,Distributer>::Matrix(U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY)
{
  this->_dimensionX = {dimensionX};
  this->_dimensionY = {dimensionY};
  this->_globalDimensionX = {globalDimensionX};
  this->_globalDimensionY = {globalDimensionY};

  Structure<T,U,Distributer>::Assemble(this->_data, this->_matrix, this->_numElems, this->_dimensionX, this->_dimensionY);
  return;
}

// This guy could be changed to use pass-by-value via rvalue constructor
template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class Structure, template<typename, typename,int> class Distributer>
Matrix<T,U,Structure,Distributer>::Matrix(std::vector<T>&& data, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, bool assemble)
{
  // Idea: move the data argument into this_data, and then set up the matrix rows (this_matrix)
  // Note that the owner of data and positions should be aware that the vectors they pass in will be destroyed and the data sucked out upon return.

  this->_dimensionX = {dimensionX};
  this->_dimensionY = {dimensionY};
  this->_globalDimensionX = {globalDimensionX};
  this->_globalDimensionY = {globalDimensionY};
  this->_numElems = getNumElems(dimensionX, dimensionY);
  this->_data = std::move(data);		// suck out the data from the argument into our member variable

  // Reason: sometimes, I just want to enter in an empty vector that will be filled up in Serializer. Other times, I want to truly
  //   assemble a vector for use somewhere else.
  if (assemble)
  {
    // Now call the MatrixAssemble method
    this->_matrix.resize(this->_dimensionX);
    Structure<T,U,Distributer>::AssembleMatrix(this->_data, this->_matrix, dimensionX, dimensionY);
  }
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
  // Actually, now that we are purly using vectors, I don't think we need to delete anything. Once the instance
  //   of the class goes out of scope, the vector data gets deleted automatically.
  //Structure<T,U,Distributer>::Dissamble(this->_data);
}

template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class Structure, template<typename, typename,int> class Distributer>
void Matrix<T,U,Structure,Distributer>::copy(const Matrix& rhs)
{
  this->_dimensionX = {rhs._dimensionX};
  this->_dimensionY = {rhs._dimensionY};
  this->_numElems = {rhs._numElems};
  this->_globalDimensionX = {rhs._globalDimensionX};
  this->_globalDimensionY = {rhs._globalDimensionY};
  Structure<T,U,Distributer>::Copy(this->_data, this->_matrix, rhs._data, this->_dimensionX, this->_dimensionY);
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
  this->_data = std::move(rhs._data);
  this->_matrix = std::move(rhs._matrix);
  return;
}

template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class Structure, template<typename, typename,int> class Distributer>
void Matrix<T,U,Structure,Distributer>::DistributeRandom(int localPgridX, int localPgridY, int globalPgridX, int globalPgridY)
{
  // Matrix must be already constructed with memory. Add a check for this later.

  // This is a 2-level Policy-class trick due to the lack of orthogonality between the
  //   Structure Policy and the Distributer Policy.

  Structure<T,U,Distributer>::DistributeRandom(this->_matrix, this->_dimensionX, this->_dimensionY, this->_globalDimensionX, this->_globalDimensionY, localPgridX, localPgridY, globalPgridX, globalPgridY);
}

template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class Structure, template<typename, typename,int> class Distributer>
void Matrix<T,U,Structure,Distributer>::DistributeSymmetric(int localPgridX, int localPgridY, int globalPgridX, int globalPgridY, bool diagonallyDominant)
{
  // Matrix must be already constructed with memory. Add a check for this later.

  // This is a 2-level Policy-class trick due to the lack of orthogonality between the
  //   Structure Policy and the Distributer Policy.

  // Note: this method is only defined for Square matrices 
  Structure<T,U,Distributer>::DistributeSymmetric(this->_matrix, this->_dimensionX, this->_dimensionY, this->_globalDimensionX, this->_globalDimensionY, localPgridX, localPgridY, globalPgridX, globalPgridY, diagonallyDominant);
}

template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class Structure, template<typename, typename,int> class Distributer>
void Matrix<T,U,Structure,Distributer>::print() const
{
  Structure<T,U,Distributer>::Print(this->_matrix, this->_dimensionX, this->_dimensionY);
}
