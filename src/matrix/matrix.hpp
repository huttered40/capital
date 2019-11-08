/* Author: Edward Hutter */

// #include "matrix.h"  -> Compiler needs the full definition of the templated class in order to instantiate it.

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::matrix(U globalDimensionX, U globalDimensionY, size_t globalPgridX, size_t globalPgridY){
  // Extra padding of zeros is at most 1 in either dimension
  size_t pHelper = globalDimensionX%globalPgridX;
  this->_dimensionX = {globalDimensionX/globalPgridX + (pHelper ? 1 : 0)};
  pHelper = globalDimensionY%globalPgridY;
  this->_dimensionY = {globalDimensionY/globalPgridY + (pHelper ? 1 : 0)};
  // We also need to pad the global dimensions

/*
  I dont like this. I think its wrong to do this. If anything needs changed, its the validate code, which I think is the only thing causing the issue.
  this->_globalDimensionX = {this->_dimensionX*globalPgridX};
  this->_globalDimensionY = {this->_dimensionY*globalPgridY};
*/
  this->_globalDimensionX = {globalDimensionX};
  this->_globalDimensionY = {globalDimensionY};

  StructurePolicy::_Assemble(this->_data, this->_matrix, this->_numElems, this->_dimensionX, this->_dimensionY);
  return;
}

// This guy could be changed to use pass-by-value via rvalue constructor
template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::matrix(std::vector<T>&& data, U dimensionX, U dimensionY, U globalDimensionX, U globalDimensionY, bool assemble){
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
  if (assemble){
    // Now call the matrixAssemble method
    this->_matrix.resize(this->_dimensionX);
    StructurePolicy::_AssembleMatrix(this->_data, this->_matrix, dimensionX, dimensionY);
  }
}

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::matrix(const matrix& rhs){
  copy(rhs);
  return;
}

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::matrix(matrix&& rhs){
  // Use std::forward in the future.
  mover(std::move(rhs));
  return;
}

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>& matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::operator=(const matrix& rhs){
  if (this != &rhs){
    copy(rhs);
  }
  return *this;
}

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>& matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::operator=(matrix&& rhs){
  // Use std::forward in the future.
  if (this != &rhs){
    mover(std::move(rhs));
  }
  return *this;
}

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::~matrix(){
  // Actually, now that we are purly using vectors, I don't think we need to delete anything. Once the instance
  //   of the class goes out of scope, the vector data gets deleted automatically.
  //Structure<T,U,Distributer>::Dissamble(this->_data);
}

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
void matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::copy(const matrix& rhs){
  this->_dimensionX = {rhs._dimensionX};
  this->_dimensionY = {rhs._dimensionY};
  this->_numElems = {rhs._numElems};
  this->_globalDimensionX = {rhs._globalDimensionX};
  this->_globalDimensionY = {rhs._globalDimensionY};
  StructurePolicy::_Copy(this->_data, this->_matrix, rhs._data, this->_dimensionX, this->_dimensionY);
  return;
}

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
void matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::mover(matrix&& rhs){
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

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
void matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::DistributeRandom(size_t localPgridX, size_t localPgridY, size_t globalPgridX, size_t globalPgridY, size_t key){
  // matrix must be already constructed with memory. Add a check for this later.
  DistributionPolicy::_DistributeRandom(this->_matrix,this->_dimensionX,this->_dimensionY,this->_globalDimensionX,this->_globalDimensionY,localPgridX,localPgridY,globalPgridX,globalPgridY,key,StructurePolicy());
}

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
void matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::DistributeSymmetric(size_t localPgridX, size_t localPgridY, size_t globalPgridX, size_t globalPgridY, size_t key, bool diagonallyDominant){
  // matrix must be already constructed with memory. Add a check for this later.
  DistributionPolicy::_DistributeSymmetric(this->_matrix,this->_dimensionX,this->_dimensionY,this->_globalDimensionX,this->_globalDimensionY,localPgridX,localPgridY,globalPgridX,globalPgridY,key,diagonallyDominant);
}

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
void matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::DistributeIdentity(size_t localPgridX, size_t localPgridY, size_t globalPgridX, size_t globalPgridY, T val){
  // matrix must be already constructed with memory. Add a check for this later.
  DistributionPolicy::_DistributeIdentity(this->_matrix,this->_dimensionX,this->_dimensionY,this->_globalDimensionX,this->_globalDimensionY,localPgridX,localPgridY,globalPgridX,globalPgridY,val);
}

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
void matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::DistributeDebug(size_t localPgridX, size_t localPgridY, size_t globalPgridX, size_t globalPgridY){
  // matrix must be already constructed with memory. Add a check for this later.
  DistributionPolicy::_DistributeDebug(this->_matrix,this->_dimensionX,this->_dimensionY,this->_globalDimensionX,this->_globalDimensionY,localPgridX,localPgridY,globalPgridX,globalPgridY);
}

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
void matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::print() const{
  StructurePolicy::_Print(this->_matrix,this->_dimensionX,this->_dimensionY);
}
