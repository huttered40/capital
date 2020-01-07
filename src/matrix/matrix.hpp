/* Author: Edward Hutter */

// #include "matrix.h"  -> Compiler needs the full definition of the templated class in order to instantiate it.

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::matrix(DimensionType globalDimensionX, DimensionType globalDimensionY, int64_t globalPgridX, int64_t globalPgridY){
  // Extra padding of zeros is at most 1 in either dimension
  int64_t pHelper = globalDimensionX%globalPgridX;
  this->_dimensionX = {globalDimensionX/globalPgridX + (pHelper ? 1 : 0)};
  pHelper = globalDimensionY%globalPgridY;
  this->_dimensionY = {globalDimensionY/globalPgridY + (pHelper ? 1 : 0)};

  this->_globalDimensionX = {globalDimensionX};
  this->_globalDimensionY = {globalDimensionY};

  _assemble(this->_data, this->_scratch, this->_pad, this->_numElems, this->_dimensionX, this->_dimensionY);
  this->allocated_data=true; this->filled=true;
  return;
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::matrix(ScalarType* data, DimensionType dimensionX, DimensionType dimensionY, DimensionType globalDimensionX, DimensionType globalDimensionY, DimensionType globalPgridX, DimensionType globalPgridY){
  // Idea: move the data argument into this_data, and then set up the matrix rows (this_matrix)
  // Note that the owner of data and positions should be aware that the vectors they pass in will be destroyed and the data sucked out upon return.

  // If matrix dimensions do not align with what is necessary, we will need to perform a deep copy.
  int64_t pHelper = globalDimensionX%globalPgridX;
  this->_dimensionX = {globalDimensionX/globalPgridX + (pHelper ? 1 : 0)};
  pHelper = globalDimensionY%globalPgridY;
  this->_dimensionY = {globalDimensionY/globalPgridY + (pHelper ? 1 : 0)};
  bool valid = ((this->_dimensionX==dimensionX) && (this->_dimensionY==dimensionY));

  this->_globalDimensionX = {globalDimensionX};
  this->_globalDimensionY = {globalDimensionY};
  this->_numElems = num_elems(dimensionX, dimensionY);	// will get overwritten if necessary
  this->_data = data;					// will get overwritten if necessary

  // Reason: sometimes, I just want to enter in an empty vector that will be filled up in Serializer. Other times, I want to truly
  //   assemble a vector for use somewhere else.
  if ((this->_data == nullptr) || (!valid)){
    _assemble(this->_data, this->_scratch, this->_pad, this->_numElems, dimensionX, dimensionY);
    this->allocated_data=true;
  }
  else{
    // No longer supporting cheap copies if pointer is valid, because the algorithm internals take extreme liberties in optimizations
    _copy(this->_data, this->_scratch, this->_pad, data, this->_dimensionX, this->_dimensionY);
    this->allocated_data=true;
  }
  this->filled=true;
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::matrix(ScalarType* data, DimensionType dimensionX, DimensionType dimensionY, DimensionType globalPgridX, DimensionType globalPgridY){
  // Idea: move the data argument into this_data, and then set up the matrix rows (this_matrix)
  // Note that the owner of data and positions should be aware that the vectors they pass in will be destroyed and the data sucked out upon return.

  // If matrix dimensions do not align with what is necessary, we will need to perform a deep copy.
  this->_dimensionX = {dimensionX};
  this->_dimensionY = {dimensionY};
  this->_globalDimensionX = {dimensionX*globalPgridX};
  this->_globalDimensionY = {dimensionY*globalPgridY};
  this->_numElems = num_elems(dimensionX, dimensionY);	// will get overwritten if necessary
  this->_data = data;					// will get overwritten if necessary

  if (data != nullptr){
    _assemble_matrix(this->_data, this->_scratch, this->_pad, this->_dimensionX, this->_dimensionY);
    this->allocated_data=false;
  }
  else{
    _assemble(this->_data, this->_scratch, this->_pad, this->_numElems, this->_dimensionX, this->_dimensionY);
    this->allocated_data=true;
  }
  this->filled=true;
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::matrix(ScalarType* data, DimensionType dimensionX, DimensionType dimensionY, DimensionType globalPgridX, DimensionType globalPgridY, bool){
  this->_dimensionX = {dimensionX};
  this->_dimensionY = {dimensionY};
  this->_globalDimensionX = {dimensionX*globalPgridX};
  this->_globalDimensionY = {dimensionY*globalPgridY};
  this->_numElems = num_elems(dimensionX, dimensionY);	// will get overwritten if necessary
  this->_data = data;					// will get overwritten if necessary
  this->allocated_data=false;
  this->filled=false;
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::matrix(const matrix& rhs){
  copy(rhs);
  this->filled=true;
  return;
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::matrix(matrix&& rhs){
  // DimensionTypese std::forward in the future.
  mover(std::move(rhs));
  this->filled=true;
  return;
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>& matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::operator=(const matrix& rhs){
  if (this != &rhs){
    copy(rhs);
  }
  this->filled=true;
  return *this;
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>& matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::operator=(matrix&& rhs){
  if (this != &rhs){
    mover(std::move(rhs));
  }
  this->filled=true;
  return *this;
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::~matrix(){
  this->_destroy_();
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
void matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::_fill_(){
  if (!this->filled){
    if (this->_data != nullptr){
      _assemble_matrix(this->_data, this->_scratch, this->_pad, this->_dimensionX, this->_dimensionY);
      this->allocated_data=false;
    }
    else{
      _assemble(this->_data, this->_scratch, this->_pad, this->_numElems, this->_dimensionX, this->_dimensionY);
      this->allocated_data=true;
    }
    this->filled=true;
  }
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
void matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::_register_(DimensionType globalDimensionX, DimensionType globalDimensionY, int64_t globalPgridX, int64_t globalPgridY){
  if (!this->filled){
    // Extra padding of zeros is at most 1 in either dimension
    int64_t pHelper = globalDimensionX%globalPgridX;
    this->_dimensionX = {globalDimensionX/globalPgridX + (pHelper ? 1 : 0)};
    pHelper = globalDimensionY%globalPgridY;
    this->_dimensionY = {globalDimensionY/globalPgridY + (pHelper ? 1 : 0)};
    this->_globalDimensionX = {globalDimensionX};
    this->_globalDimensionY = {globalDimensionY};
    this->_data=nullptr;
    // Note: do not modify 'filled' member, as this method should essentially be a no-op if 'filled'=true
    _fill_();
  }
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
void matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::_destroy_(){
  // Actually, now that we are purly using vectors, I don't think we need to delete anything. Once the instance
  //   of the class goes out of scope, the vector data gets deleted automatically.
  if (this->filled){
    if (this->_scratch != nullptr){ delete[] this->_scratch; this->_scratch=nullptr;}	// could add an assert here for StructurePolicy==lowertri,uppertri
    if (this->_pad != nullptr){ delete[] this->_pad; this->_pad=nullptr;}	// could add an assert here for StructurePolicy==lowertri,uppertri
    if (this->allocated_data && (this->_data != nullptr)){ delete[] this->_data; this->_data=nullptr;}
    this->allocated_data=false;
    this->filled=false;
  }
  this->filled=false;
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
void matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::_restrict_(DimensionType startX, DimensionType endX, DimensionType startY, DimensionType endY){
  this->_data_=this->_data; this->_scratch_=this->_scratch; this->_dimensionX_=this->_dimensionX; this->_dimensionY_=this->_dimensionY; this->_numElems_=this->_numElems;
  this->_data=&this->_data_[offset_local(startX,startY)]; this->_scratch=&this->_scratch_[offset_local(startX,startY)]; this->_dimensionX=endX-startX; this->_dimensionY=endY-startY; this->_numElems=num_elems(endX-startX,endY-startY);
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
void matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::_derestrict_(){
  this->_data=this->_data_; this->_scratch=this->_scratch_; this->_dimensionX=this->_dimensionX_; this->_dimensionY=this->_dimensionY_; this->_numElems=this->_numElems_;
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
void matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::copy(const matrix& rhs){
  this->_dimensionX = {rhs._dimensionX};
  this->_dimensionY = {rhs._dimensionY};
  this->_numElems = {rhs._numElems};
  this->_globalDimensionX = {rhs._globalDimensionX};
  this->_globalDimensionY = {rhs._globalDimensionY};
  _copy(this->_data, this->_scratch, this->_pad, rhs._data, this->_dimensionX, this->_dimensionY);
  this->allocated_data=true;
  this->filled=true;
  return;
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
void matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::mover(matrix&& rhs){
  assert(rhs.allocated_data);	// we don't support "move"ing from pointer-generated matrix instances
  this->_dimensionX = {rhs._dimensionX};
  this->_dimensionY = {rhs._dimensionY};
  this->_numElems = {rhs._numElems};
  this->_globalDimensionX = {rhs._globalDimensionX};
  this->_globalDimensionY = {rhs._globalDimensionY};
  // Suck out the matrix member from rhs and stick it in our field.
  // For now, we don't need a Allocator Policy Interface Move() method.
  this->_data = rhs._data; rhs._data = nullptr;
  this->_scratch = rhs._scratch; rhs._scratch = nullptr;
  this->_pad = rhs._pad; rhs._pad = nullptr;
  this->allocated_data=rhs.allocated_data;
  this->filled=rhs.filled;
  return;
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
void matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::distribute_random(int64_t localPgridX, int64_t localPgridY, int64_t globalPgridX, int64_t globalPgridY, int64_t key){
  // matrix must be already constructed with memory. Add a check for this later.
  _distribute_random(this->_data,this->_dimensionX,this->_dimensionY,this->_globalDimensionX,this->_globalDimensionY,localPgridX,localPgridY,globalPgridX,globalPgridY,key);
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
void matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::distribute_symmetric(int64_t localPgridX, int64_t localPgridY, int64_t globalPgridX, int64_t globalPgridY, int64_t key, bool diagonallyDominant){
  // matrix must be already constructed with memory. Add a check for this later.
  _distribute_symmetric(this->_data,this->_dimensionX,this->_dimensionY,this->_globalDimensionX,this->_globalDimensionY,localPgridX,localPgridY,globalPgridX,globalPgridY,key,diagonallyDominant);
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
void matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::distribute_identity(int64_t localPgridX, int64_t localPgridY, int64_t globalPgridX, int64_t globalPgridY, ScalarType val){
  // matrix must be already constructed with memory. Add a check for this later.
  _distribute_identity(this->_data,this->_dimensionX,this->_dimensionY,this->_globalDimensionX,this->_globalDimensionY,localPgridX,localPgridY,globalPgridX,globalPgridY,val);
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
void matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::distribute_debug(int64_t localPgridX, int64_t localPgridY, int64_t globalPgridX, int64_t globalPgridY){
  // matrix must be already constructed with memory. Add a check for this later.
  _distribute_debug(this->_data,this->_dimensionX,this->_dimensionY,this->_globalDimensionX,this->_globalDimensionY,localPgridX,localPgridY,globalPgridX,globalPgridY);
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
void matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::print() const{
  _print(this->_data,this->_dimensionX,this->_dimensionY);
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
void matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::print_data() const{
  _print(this->_data,this->_dimensionX,this->_dimensionY);
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
void matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::print_scratch() const{
  _print(this->_scratch,this->_dimensionX,this->_dimensionY);
}

template<typename ScalarType, typename DimensionType, typename StructurePolicy, typename OffloadPolicy>
void matrix<ScalarType,DimensionType,StructurePolicy,OffloadPolicy>::print_pad() const{
  rect::_print(this->_pad,this->_dimensionX,this->_dimensionY);
}
