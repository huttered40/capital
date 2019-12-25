/* Author: Edward Hutter */

// #include "matrix.h"  -> Compiler needs the full definition of the templated class in order to instantiate it.

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::matrix(U globalDimensionX, U globalDimensionY, int64_t globalPgridX, int64_t globalPgridY){
  // Extra padding of zeros is at most 1 in either dimension
  int64_t pHelper = globalDimensionX%globalPgridX;
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

  StructurePolicy::_assemble(this->_data, this->_scratch, this->_pad, this->_matrix, this->_numElems, this->_dimensionX, this->_dimensionY);
  this->allocated_data=true;
  return;
}

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::matrix(T* data, U dimensionX, U dimensionY, U globalPgridX, U globalPgridY){
  // Idea: move the data argument into this_data, and then set up the matrix rows (this_matrix)
  // Note that the owner of data and positions should be aware that the vectors they pass in will be destroyed and the data sucked out upon return.

  this->_dimensionX = {dimensionX};
  this->_dimensionY = {dimensionY};
  this->_globalDimensionX = {dimensionX*globalPgridX};
  this->_globalDimensionY = {dimensionY*globalPgridY};
  this->_numElems = num_elems(dimensionX, dimensionY);
  this->_data = data;		// suck out the data from the argument into our member variable

  // Reason: sometimes, I just want to enter in an empty vector that will be filled up in Serializer. Other times, I want to truly
  //   assemble a vector for use somewhere else.
  if (this->_data == nullptr){
    StructurePolicy::_assemble(this->_data, this->_scratch, this->_pad, this->_matrix, this->_numElems, dimensionX, dimensionY);
    this->allocated_data=true;
  }
  else{
    // I could add an assert here to make sure non-rect/square StructurePolicies get in this branch
    StructurePolicy::_assemble_matrix(this->_data, this->_scratch, this->_pad, this->_matrix, dimensionX, dimensionY);
    this->allocated_data=false;
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
  if (this->_scratch != nullptr){ delete[] this->_scratch; }	// could add an assert here for StructurePolicy==lowertri,uppertri
  if (this->_pad != nullptr){ delete[] this->_pad; }	// could add an assert here for StructurePolicy==lowertri,uppertri
  if (this->allocated_data && this->_data != nullptr){ delete[] this->_data; }
}

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
void matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::copy(const matrix& rhs){
  this->_dimensionX = {rhs._dimensionX};
  this->_dimensionY = {rhs._dimensionY};
  this->_numElems = {rhs._numElems};
  this->_globalDimensionX = {rhs._globalDimensionX};
  this->_globalDimensionY = {rhs._globalDimensionY};
  StructurePolicy::_copy(this->_data, this->_scratch, this->_pad, this->_matrix, rhs._data, this->_dimensionX, this->_dimensionY);
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
  this->_matrix = std::move(rhs._matrix);
  this->_data = rhs._data; rhs._data = nullptr;
  this->_scratch = rhs._scratch; rhs._scratch = nullptr;
  this->_pad = rhs._pad; rhs._pad = nullptr;
  return;
}

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
void matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::distribute_random(int64_t localPgridX, int64_t localPgridY, int64_t globalPgridX, int64_t globalPgridY, int64_t key){
  // matrix must be already constructed with memory. Add a check for this later.
  DistributionPolicy::_distribute_random(this->_matrix,this->_dimensionX,this->_dimensionY,this->_globalDimensionX,this->_globalDimensionY,localPgridX,localPgridY,globalPgridX,globalPgridY,key,StructurePolicy());
}

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
void matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::distribute_symmetric(int64_t localPgridX, int64_t localPgridY, int64_t globalPgridX, int64_t globalPgridY, int64_t key, bool diagonallyDominant){
  // matrix must be already constructed with memory. Add a check for this later.
  DistributionPolicy::_distribute_symmetric(this->_matrix,this->_dimensionX,this->_dimensionY,this->_globalDimensionX,this->_globalDimensionY,localPgridX,localPgridY,globalPgridX,globalPgridY,key,diagonallyDominant);
}

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
void matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::distribute_identity(int64_t localPgridX, int64_t localPgridY, int64_t globalPgridX, int64_t globalPgridY, T val){
  // matrix must be already constructed with memory. Add a check for this later.
  DistributionPolicy::_distribute_identity(this->_matrix,this->_dimensionX,this->_dimensionY,this->_globalDimensionX,this->_globalDimensionY,localPgridX,localPgridY,globalPgridX,globalPgridY,val);
}

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
void matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::distribute_debug(int64_t localPgridX, int64_t localPgridY, int64_t globalPgridX, int64_t globalPgridY){
  // matrix must be already constructed with memory. Add a check for this later.
  DistributionPolicy::_distribute_debug(this->_matrix,this->_dimensionX,this->_dimensionY,this->_globalDimensionX,this->_globalDimensionY,localPgridX,localPgridY,globalPgridX,globalPgridY);
}

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
void matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::print() const{
  StructurePolicy::_print(this->_matrix,this->_dimensionX,this->_dimensionY);
}

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
void matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::print_data() const{
  StructurePolicy::_print(this->_matrix,this->_dimensionX,this->_dimensionY);
}

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
void matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::print_scratch() const{
  for (int i=0; i<this->_numElems; i++){ std::cout << this->_scratch[i] << std::endl;}
  std::cout << "\n\n";
}

template<typename T, typename U, typename StructurePolicy, typename DistributionPolicy, typename OffloadPolicy>
void matrix<T,U,StructurePolicy,DistributionPolicy,OffloadPolicy>::print_pad() const{
  for (int i=0; i<this->_dimensionX*this->_dimensionY; i++){ std::cout << this->_pad[i] << std::endl;}
  std::cout << "\n\n";
}
