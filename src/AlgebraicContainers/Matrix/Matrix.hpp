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

template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class Structure, template<typename, typename,int> class Distributer>
Matrix<T,U,Structure,Distributer>::Matrix(std::vector<T>& data, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY)
{
  // Idea: move the data argument into this_data, and then set up the matrix rows (this_matrix)

  std::cout << "Constructor not yet implemented yet.\n";
  abort();
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
  Structure<T,U,Distributer>::Copy(this->_data, this->_matrix, rhs._matrix, this->_dimensionX, this->_dimensionY);
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
template<template<typename,typename,template<typename,typename,int> class> class StructureDest>
void Matrix<T,U,Structure,Distributer>::Serialize(Matrix<T,U,StructureDest,Distributer>& dest)
{
  // Matrix must be already constructed with memory. Add a check for this later.

  // So we are not going through the Structure Policy here like we did with the Distributer Policy.
  // Maybe that is something I should change later about the Distributer Policy implementation if this works correctly.
  
  // Note: I cannot just use the private members of dest because it is actually a different Type due to its
  //         different template parameters. This is a problem. I can implement a public getMatrix() method, and hav
  std::vector<T>& destVector = dest.getVectorData();	// Should incur no copy. A reference to the inner vector has been given.
  std::vector<T>& srcVector = this->_data;
  Serializer<T,U,Structure,StructureDest>::Serialize(srcVector, destVector, this->_dimensionX, this->_dimensionY);
}

template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class Structure, template<typename, typename,int> class Distributer>
template<template<typename,typename,template<typename,typename,int> class> class StructureDest>
void Matrix<T,U,Structure,Distributer>::Serialize(Matrix<T,U,StructureDest,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend)
{
  // See notes in first Serialize method above
  std::vector<T>& destVector = dest.getVectorData();	// Should incur no copy. A reference to the inner vector has been given.
  std::vector<T>& srcVector = this->_data;
  Serializer<T,U,Structure,StructureDest>::Serialize(srcVector, destVector, this->_dimensionX, this->_dimensionY, cutDimensionXstart, cutDimensionXend, cutDimensionYstart, cutDimensionYend);
}

template<typename T, typename U, template<typename,typename,template<typename,typename,int> class> class Structure, template<typename, typename,int> class Distributer>
template<template<typename,typename,template<typename,typename,int> class> class StructureDest>
void Matrix<T,U,Structure,Distributer>::Serialize(Matrix<T,U,StructureDest,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros)
{
  // See notes in first Serialize method above
  std::vector<T>& destVector = dest.getVectorData();	// Should incur no copy. A reference to the inner vector has been given.
  std::vector<T>& srcVector = this->_data;
  Serializer<T,U,Structure,StructureDest>::Serialize(srcVector, destVector, this->_dimensionX, this->_dimensionY, cutDimensionXstart, cutDimensionXend, cutDimensionYstart, cutDimensionYend, fillZeros);
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
