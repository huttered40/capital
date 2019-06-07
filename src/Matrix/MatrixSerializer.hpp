/* Author: Edward Hutter */

/*
  Note: For bool dir, if dir == false, we serialize the bigger object into the smaller one, using the necessary indexing
                      if dir == true, we serialize the smaller object into the bigger one, using the opposite indexing as above

                      Both directions use source and dest buffers as the name suggests

	Also, we assume that the source bufer is allocated properly. But, the destination buffer is checked for appropriate size to prevent seg faults
              ans also to aid the user of thi Policy if he doesn't know the correct number of elements given th complicated Structure Policy abstraction.
*/


// Helper static method -- fills a range with zeros
template<typename T, typename U>
static void fillZerosContig(T* addr, U size){
  for (U i=0; i<size; i++){
    addr[i] = 0;
  }
}


template<typename SrcType, typename DestType>
void Serializer<Square,Square>::Serialize(SrcType& src, DestType& dest){
  TAU_FSTART(Serialize);

  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.getNumRowsLocal();
  U srcNumColumns = src.getNumColumnsLocal();
  U srcNumElems = srcNumRows*srcNumColumns;

  std::vector<T>& srcVectorData = src.getVectorData();
  std::vector<T*>& srcMatrixData = src.getMatrixData();
  std::vector<T>& destVectorData = dest.getVectorData();
  std::vector<T*>& destMatrixData = dest.getMatrixData();

  bool assembleFinder = false;
  if (static_cast<U>(destVectorData.size()) < srcNumElems){
    assembleFinder = true;
    destVectorData.resize(srcNumElems);
  }
  if (static_cast<U>(destMatrixData.size()) < srcNumColumns){
    assembleFinder = true;
    destMatrixData.resize(srcNumColumns);
  }

  // direction doesn't matter here since no indexing here
  memcpy(&destVectorData[0], &srcVectorData[0], sizeof(T)*srcNumElems);

  if (assembleFinder){
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    // User can assume that everything except for global dimensions are set. If he needs global dimensions too, he can set them himself.
    dest.setNumRowsLocal(srcNumRows);
    dest.setNumColumnsLocal(srcNumColumns);
    dest.setNumElems(srcNumElems);
    Square::_AssembleMatrix(destVectorData, destMatrixData, srcNumColumns, srcNumRows);
  }
  TAU_FSTOP(Serialize);
  return;
}

template<typename BigType, typename SmallType>
void Serializer<Square,Square>::Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart,
                                           typename BigType::DimensionType cutDimensionXend, typename BigType::DimensionType cutDimensionYstart,
                                           typename BigType::DimensionType cutDimensionYend, bool dir){
  TAU_FSTART(Serialize);

  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
/*  Commenting this assert out for now, since CFR3D requires non-square partitioning
  if (rangeX != rangeY)
  {
    std::cout << "rangeX - " << rangeX << " and rangeY - " << rangeY << std::endl;
    std::cout << "cutDimensionXstart - " << cutDimensionXstart << ", cutDimensionXend - " << cutDimensionXend << "cutDimensionYstart - " << cutDimensionYstart << ", cutDimensionYend - " << cutDimensionYend << std::endl;
    assert(rangeX == rangeY);
  }
*/
  U bigNumRows = big.getNumRowsLocal();
  U bigNumColumns = big.getNumColumnsLocal();

  std::vector<T>& bigVectorData = big.getVectorData();
  std::vector<T*>& bigMatrixData = big.getMatrixData();
  std::vector<T>& smallVectorData = small.getVectorData();
  std::vector<T*>& smallMatrixData = small.getMatrixData();

  // We assume that if dir==true, the user passed in a destination matrix that is properly sized
  // If I find a use case where that is not true, then I can pass in big x and y dimensions, but I do not wan to do that.
  // In most cases, the lae matri will be known and we are simply trying to fill in the smaller into the existing bigger

  U numElems = (dir ? bigNumRows*bigNumColumns : rangeX*rangeY);
  U numColumns = (dir ? bigNumColumns : rangeX);
  bool assembleFinder = false;
  if (static_cast<U>((dir ? bigVectorData.size() : smallVectorData.size())) < numElems){
    assembleFinder = true;
    dir ? bigVectorData.resize(numElems) : smallVectorData.resize(numElems);
  }
  if (static_cast<U>((dir ? bigMatrixData.size() : smallMatrixData.size())) < numColumns){
    assembleFinder = true;
    dir ? bigMatrixData.resize(numColumns) : smallMatrixData.resize(numColumns);
  }

  U destIndex = (dir ? cutDimensionYstart+bigNumRows*cutDimensionXstart : 0);
  U srcIndex = (dir ? 0 : cutDimensionYstart+bigNumRows*cutDimensionXstart);
  U srcCounter = (dir ? rangeY : bigNumRows);
  U destCounter = (dir ? bigNumRows : rangeY);
  auto& destVectorData = dir ? bigVectorData : smallVectorData;
  auto& srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destIndex], &srcVectorData[srcIndex], sizeof(T)*rangeY);		// rangeX is fine. blocks of size rangeX are still being copied.
    destIndex += destCounter;
    srcIndex += srcCounter;
  }

  if (assembleFinder){
    if (dir) {abort();}		// weird case that I want to check against
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    small.setNumRowsLocal(rangeY);
    small.setNumColumnsLocal(numColumns);
    small.setNumElems(numElems);
    Square::_AssembleMatrix(destVectorData, smallMatrixData, numColumns, rangeY);
  }
  TAU_FSTOP(Serialize);
}

template<typename SrcType, typename DestType>
void Serializer<Square,Rectangular>::Serialize(SrcType& src, DestType& dest){
  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  std::cout << "Not fully implemented yet in MatrixSerializer Square -> Rectangular\n";
  return;
}

template<typename BigType, typename SmallType>
void Serializer<Square,Rectangular>::Serialize(BigType& src, SmallType& dest, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                               typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){
  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  std::cout << "Not fully implemented yet in MatrixSerializer Square -> Rectangular\n";
  return;
}

template<typename SrcType, typename DestType>
void Serializer<Square,UpperTriangular>::Serialize(SrcType& src, DestType& dest){
  TAU_FSTART(Serialize);

  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.getNumRowsLocal();
  U srcNumColumns = src.getNumColumnsLocal();

  std::vector<T>& srcVectorData = src.getVectorData();
  std::vector<T>& destVectorData = dest.getVectorData();
  std::vector<T*>& destMatrixData = dest.getMatrixData();

  U numElems = ((srcNumColumns*(srcNumColumns+1))>>1);
  bool assembleFinder = false;
  if (static_cast<U>(destVectorData.size()) < numElems){
    assembleFinder = true;
    destVectorData.resize(numElems);
  }
  if (static_cast<U>(destMatrixData.size()) < srcNumRows){
    assembleFinder = true;
    destMatrixData.resize(srcNumRows);
  }

  U counter{1};
  U srcOffset{0};
  U destOffset{0};
  U counter2{srcNumColumns};

  for (U i=0; i<srcNumColumns; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], counter*sizeof(T));
    srcOffset += counter2;
    destOffset += counter;
    counter++;
  }

  if (assembleFinder){
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    dest.setNumRowsLocal(srcNumRows);
    dest.setNumColumnsLocal(srcNumColumns);
    dest.setNumElems(numElems);
    UpperTriangular::_AssembleMatrix(destVectorData, destMatrixData, srcNumColumns, srcNumRows);
  }
  TAU_FSTOP(Serialize);
  return;
}

/*
template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,Square, UpperTriangular>::Serialize(Matrix<T,U,Square,Distributer>& src,
  Matrix<T,U,UpperTriangular,Distributer>& dest, bool fillZeros, bool dir)
{
  std::cout << "Not updated. Only Square is. Makes no sense to change the implementation of all of these Structures until we think that the Square is right.\n";
  abort();
  if (dir == true)
  {
    std::cout << "Not finished yet. Complete when necessary\n";
    abort();
  }

  U numElems = dimensionX*dimensionX;
  if (dest.size() < numElems)
  {
    dest.resize(numElems);
  }

  U counter{dimensionX};
  U srcOffset{0};
  U destOffset{0};
  U counter2{dimensionX+1};
  for (U i=0; i<dimensionY; i++)
  {
    U fillSize = dimensionX-counter;
    fillZerosContig(&destVectorData[destOffset], fillSize);
    destOffset += fillSize;
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], counter*sizeof(T));
    srcOffset += counter2;
    destOffset += counter;
    counter--;
  }
  return;
}
*/


template<typename BigType, typename SmallType>
void Serializer<Square,UpperTriangular>::Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                   typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){
  TAU_FSTART(Serialize);

  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
  assert(rangeX == rangeY);

  U bigNumRows = big.getNumRowsLocal();
  U bigNumColumns = big.getNumColumnsLocal();

  std::vector<T>& bigVectorData = big.getVectorData();
  std::vector<T*>& bigMatrixData = big.getMatrixData();
  std::vector<T>& smallVectorData = small.getVectorData();
  std::vector<T*>& smallMatrixData = small.getMatrixData();

  // We assume that if dir==true, the user passed in a destination matrix that is properly sized
  // If I find a use case where that is not true, then I can pass in big x and y dimensions, but I do not wan to do that.
  // In most cases, the lae matri will be known and we are simply trying to fill in the smaller into the existing bigger

  U numElems = (dir ? bigNumRows*bigNumRows : ((rangeX*(rangeX+1))>>1));
  U numColumns = (dir ? bigNumColumns : rangeX);
  bool assembleFinder = false;
  if ((dir ? bigVectorData.size() : smallVectorData.size()) < numElems){
    assembleFinder = true;
    dir ? bigVectorData.resize(numElems) : smallVectorData.resize(numElems);
  }
  if ((dir ? bigMatrixData.size() : smallMatrixData.size()) < numColumns){
    assembleFinder = true;
    dir ? bigMatrixData.resize(numColumns) : smallMatrixData.resize(numColumns);
  }

  U destIndex = (dir ? cutDimensionYstart+bigNumRows*cutDimensionXstart : 0);
  U counter{1};
  U srcIndexSave = (dir ? 0 : cutDimensionYstart+bigNumRows*cutDimensionXstart);
  auto& destVectorData = (dir ? bigVectorData : smallVectorData);
  auto& srcVectorData = (dir ? smallVectorData : bigVectorData);
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destIndex], &srcVectorData[srcIndexSave], sizeof(T)*counter);
    destIndex += (dir ? bigNumRows : counter);
    srcIndexSave += (dir ? counter : bigNumRows);
    counter++;
  }

  if (assembleFinder){
    if (dir) {abort();}		// weird case that I want to check against
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    small.setNumRowsLocal(rangeY);
    small.setNumColumnsLocal(numColumns);
    small.setNumElems(numElems);
    // I am only providing UT here, not square, because if square, it would have aborted
    UpperTriangular::_AssembleMatrix(destVectorData, smallMatrixData, numColumns, rangeY);
  }
  TAU_FSTOP(Serialize);
}

/*
template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,Square, UpperTriangular>::Serialize(Matrix<T,U,Square,Distributer>& src,
  Matrix<T,U,UpperTriangular,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros, bool dir)
{
  std::cout << "Not updated. Only Square is. Makes no sense to change the implementation of all of these Structures until we think that the Square is right.\n";
  abort();
  if (dir == true)
  {
    std::cout << "Not finished yet. Complete when necessary\n";
    abort();
  }
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  assert(rangeX == rangeY);

  U numElems = rangeX*rangeX;
  if (dest.size() < numElems)
  {
    dest.resize(numElems);
  }

  U destIndex = 0;
  U counter{rangeX};
  U srcIndexSave = cutDimensionYstart*dimensionX+cutDimensionXstart;
  for (U i=0; i<rangeY; i++)
  {
    U fillSize = rangeX-counter;
    fillZerosContig(&destVectorData[destIndex], fillSize);
    destIndex += fillSize;
    memcpy(&destVectorData[destIndex], &srcVectorData[srcIndexSave], sizeof(T)*counter);
    destIndex += counter;
    srcIndexSave += dimensionX;
    counter--;
  }
}
*/


template<typename SrcType, typename DestType>
void Serializer<Square,LowerTriangular>::Serialize(SrcType& src, DestType& dest){
  TAU_FSTART(Serialize);

  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.getNumRowsLocal();
  U srcNumColumns = src.getNumColumnsLocal();
  U destNumRows = dest.getNumRowsLocal();
  U destNumColumns = dest.getNumColumnsLocal();

  std::vector<T>& srcVectorData = src.getVectorData();
  std::vector<T*>& srcMatrixData = src.getMatrixData();
  std::vector<T>& destVectorData = dest.getVectorData();
  std::vector<T*>& destMatrixData = dest.getMatrixData();

  U numElems = ((srcNumColumns*(srcNumColumns+1))>>1);
  bool assembleFinder = false;
  if (static_cast<U>(destVectorData.size()) < numElems){
    assembleFinder = true;
    destVectorData.resize(numElems);
  }
  if (static_cast<U>(destMatrixData.size()) < srcNumColumns){
    assembleFinder = true;
    destMatrixData.resize(srcNumColumns);
  }

  U counter{srcNumRows};
  U srcOffset{0};
  U destOffset{0};
  U counter2{srcNumRows+1};
  for (U i=0; i<srcNumColumns; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], counter*sizeof(T));
    srcOffset += counter2;
    destOffset += counter;
    counter--;
  }

  if (assembleFinder){
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    dest.setNumRowsLocal(srcNumRows);
    dest.setNumColumnsLocal(srcNumColumns);
    dest.setNumElems(numElems);
    LowerTriangular::_AssembleMatrix(destVectorData, destMatrixData, srcNumColumns, srcNumRows);
  }
  TAU_FSTOP(Serialize);
  return;
}

/*
template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,Square, LowerTriangular>::Serialize(Matrix<T,U,Square,Distributer>& src,
  Matrix<T,U,LowerTriangular,Distributer>& dest, bool fillZeros, bool dir)
{
  std::cout << "Not updated. Only Square is. Makes no sense to change the implementation of all of these Structures until we think that the Square is right.\n";
  abort();
  if (dir == true)
  {
    std::cout << "Not finished yet. Complete when necessary\n";
    abort();
  }

  U numElems = dimensionX*dimensionX;
  if (dest.size() < numElems)
  {
    dest.resize(numElems);
  }

  U counter{1};
  U srcOffset{0};
  U destOffset{0};
  U counter2{dimensionX};
  for (U i=0; i<dimensionY; i++)
  {
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], counter*sizeof(T));
    srcOffset += counter2;
    destOffset += counter;
    U fillSize = dimensionX - counter;
    fillZerosContig(&destVectorData[destOffset], fillSize);
    destOffset += fillSize;
    counter++;
  }
  return;
}
*/


template<typename BigType, typename SmallType>
void Serializer<Square,LowerTriangular>::Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                   typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){
  TAU_FSTART(Serialize);

  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
  assert(rangeX == rangeY);

  U bigNumRows = big.getNumRowsLocal();
  U bigNumColumns = big.getNumColumnsLocal();

  std::vector<T>& bigVectorData = big.getVectorData();
  std::vector<T*>& bigMatrixData = big.getMatrixData();
  std::vector<T>& smallVectorData = small.getVectorData();
  std::vector<T*>& smallMatrixData = small.getMatrixData();

  // We assume that if dir==true, the user passed in a destination matrix that is properly sized
  // If I find a use case where that is not true, then I can pass in big x and y dimensions, but I do not wan to do that.
  // In most cases, the lae matri will be known and we are simply trying to fill in the smaller into the existing bigger

  U numElems = (dir ? bigNumRows*bigNumRows : ((rangeX*(rangeX+1))>>1));
  U numColumns = (dir ? bigNumColumns : rangeX);
  bool assembleFinder = false;
  if ((dir ? bigVectorData.size() : smallVectorData.size()) < numElems){
    assembleFinder = true;
    dir ? bigVectorData.resize(numElems) : smallVectorData.resize(numElems);
  }
  if ((dir ? bigMatrixData.size() : smallMatrixData.size()) < numColumns){
    assembleFinder = true;
    dir ? bigMatrixData.resize(numColumns) : smallMatrixData.resize(numColumns);
  }

  U counter{rangeY};
  U srcOffset = (dir ? 0 : cutDimensionYstart+bigNumRows*cutDimensionXstart);
  U destOffset = (dir ? cutDimensionYstart+bigNumRows*cutDimensionXstart  : 0);
  auto& destVectorData = dir ? bigVectorData : smallVectorData;
  auto& srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*counter);
    destOffset += (dir ? bigNumRows+1 : counter);
    srcOffset += (dir ? counter : bigNumRows+1);
    counter--;
  }

  if (assembleFinder){
    if (dir) {abort();}		// weird case that I want to check against
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    small.setNumRowsLocal(rangeY);
    small.setNumColumnsLocal(numColumns);
    small.setNumElems(numElems);
    // I am only providing UT here, not square, because if square, it would have aborted
    LowerTriangular::_AssembleMatrix(destVectorData, smallMatrixData, numColumns, rangeY);
  }
  TAU_FSTOP(Serialize);
}

/*
template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,Square, LowerTriangular>::Serialize(Matrix<T,U,Square,Distributer>& src,
  Matrix<T,U,LowerTriangular,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros, bool dir)
{
  std::cout << "Not updated. Only Square is. Makes no sense to change the implementation of all of these Structures until we think that the Square is right.\n";
  abort();
  if (dir == true)
  {
    std::cout << "Not finished yet. Complete when necessary\n";
    abort();
  }
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  assert(rangeX == rangeY);

  U numElems = rangeX*rangeX;
  if (dest.size() < numElems)
  {
    dest.resize(numElems);
  }

  U counter{1};
  U srcOffset{cutDimensionYstart*dimensionX+cutDimensionXstart};
  U destOffset{0};
  for (U i=0; i<rangeY; i++)
  {
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*counter);
    destOffset += counter;
    srcOffset += dimensionX;
    U fillSize = rangeX - counter;
    fillZerosContig(&destVectorData[destOffset], fillSize);
    destOffset += fillSize;
    counter++;
  }
}
*/


template<typename SrcType, typename DestType>
void Serializer<Rectangular,Square>::Serialize(SrcType& src, DestType& dest){
  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  std::cout << "Not fully implemented yet in MatrixSerializer for Rectangular -> Square\n";
  return;
}

template<typename BigType, typename SmallType>
void Serializer<Rectangular,Square>::Serialize(BigType& src, SmallType& dest, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                               typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){
  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  std::cout << "Not fully implemented yet in MatrixSerializer for Rectangular -> Square\n";
  return;
}

template<typename SrcType, typename DestType>
void Serializer<Rectangular,Rectangular>::Serialize(SrcType& src, DestType& dest){
  TAU_FSTART(Serialize);

  // For now, just call Square counterpart, it should be the same --- Actually I can't unless I try to do a weird cast.
  // Annoying code bloat here
  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.getNumRowsLocal();
  U srcNumColumns = src.getNumColumnsLocal();
  U srcNumElems = srcNumRows*srcNumColumns;

  std::vector<T>& srcVectorData = src.getVectorData();
  std::vector<T*>& srcMatrixData = src.getMatrixData();
  std::vector<T>& destVectorData = dest.getVectorData();
  std::vector<T*>& destMatrixData = dest.getMatrixData();

  bool assembleFinder = false;
  if (static_cast<U>(destVectorData.size()) < srcNumElems){
    assembleFinder = true;
    destVectorData.resize(srcNumElems);
  }
  if (static_cast<U>(destMatrixData.size()) < srcNumColumns){
    assembleFinder = true;
    destMatrixData.resize(srcNumColumns);
  }

  // direction doesn't matter here since no indexing here
  memcpy(&destVectorData[0], &srcVectorData[0], sizeof(T)*srcNumElems);

  if (assembleFinder){
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    // User can assume that everything except for global dimensions are set. If he needs global dimensions too, he can set them himself.
    dest.setNumRowsLocal(srcNumRows);
    dest.setNumColumnsLocal(srcNumColumns);
    dest.setNumElems(srcNumElems);
    Square::_AssembleMatrix(destVectorData, destMatrixData, srcNumColumns, srcNumRows);
  }
  TAU_FSTOP(Serialize);
  return;
}

template<typename BigType, typename SmallType>
void Serializer<Rectangular,Rectangular>::Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                    typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){
  TAU_FSTART(Serialize);

  // For now, just call Square counterpart, it should be the same --- Actually I can't unless I try to do a weird cast.
  // Annoying code bloat here
  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
/*  Commenting this assert out for now, since CFR3D requires non-square partitioning
  if (rangeX != rangeY)
  {
    std::cout << "rangeX - " << rangeX << " and rangeY - " << rangeY << std::endl;
    std::cout << "cutDimensionXstart - " << cutDimensionXstart << ", cutDimensionXend - " << cutDimensionXend << "cutDimensionYstart - " << cutDimensionYstart << ", cutDimensionYend - " << cutDimensionYend << std::endl;
    assert(rangeX == rangeY);
  }
*/
  U bigNumRows = big.getNumRowsLocal();
  U bigNumColumns = big.getNumColumnsLocal();

  std::vector<T>& bigVectorData = big.getVectorData();
  std::vector<T*>& bigMatrixData = big.getMatrixData();
  std::vector<T>& smallVectorData = small.getVectorData();
  std::vector<T*>& smallMatrixData = small.getMatrixData();

  // We assume that if dir==true, the user passed in a destination matrix that is properly sized
  // If I find a use case where that is not true, then I can pass in big x and y dimensions, but I do not wan to do that.
  // In most cases, the lae matri will be known and we are simply trying to fill in the smaller into the existing bigger

  U numElems = (dir ? bigNumRows*bigNumColumns : rangeX*rangeY);
  U numColumns = (dir ? bigNumColumns : rangeX);
  bool assembleFinder = false;
  if (static_cast<U>((dir ? bigVectorData.size() : smallVectorData.size())) < numElems){
    assembleFinder = true;
    dir ? bigVectorData.resize(numElems) : smallVectorData.resize(numElems);
  }
  if (static_cast<U>((dir ? bigMatrixData.size() : smallMatrixData.size())) < numColumns){
    assembleFinder = true;
    dir ? bigMatrixData.resize(numColumns) : smallMatrixData.resize(numColumns);
  }

  U destIndex = (dir ? cutDimensionYstart+bigNumRows*cutDimensionXstart : 0);
  U srcIndex = (dir ? 0 : cutDimensionYstart+bigNumRows*cutDimensionXstart);
  U srcCounter = (dir ? rangeY : bigNumRows);
  U destCounter = (dir ? bigNumRows : rangeY);
  auto& destVectorData = dir ? bigVectorData : smallVectorData;
  auto& srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destIndex], &srcVectorData[srcIndex], sizeof(T)*rangeY);		// rangeX is fine. blocks of size rangeX are still being copied.
    destIndex += destCounter;
    srcIndex += srcCounter;
  }

  if (assembleFinder){
    if (dir) {abort();}		// weird case that I want to check against
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    small.setNumRowsLocal(rangeY);
    small.setNumColumnsLocal(numColumns);
    small.setNumElems(numElems);
    Square::_AssembleMatrix(destVectorData, smallMatrixData, numColumns, rangeY);
  }
  TAU_FSTOP(Serialize);
  return;
}

template<typename SrcType, typename DestType>
void Serializer<Rectangular,UpperTriangular>::Serialize(SrcType& src, DestType& dest){
  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  std::cout << "Not fully implemented yet\n";
  return;
}

template<typename BigType, typename SmallType>
void Serializer<Rectangular,UpperTriangular>::Serialize(BigType& src, SmallType& dest, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){
  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  std::cout << "Not fully implemented yet\n";
  return;
}

template<typename SrcType, typename DestType>
void Serializer<Rectangular,LowerTriangular>::Serialize(SrcType& src, DestType& dest){
  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  std::cout << "Not fully implemented yet\n";
  return;
}

template<typename BigType, typename SmallType>
void Serializer<Rectangular,LowerTriangular>::Serialize(BigType& src, SmallType& dest, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){
  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  std::cout << "Not fully implemented yet\n";
  return;
}


template<typename SrcType, typename DestType>
void Serializer<UpperTriangular,Square>::Serialize(SrcType& src, DestType& dest){
  TAU_FSTART(Serialize);

  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.getNumRowsLocal();
  U srcNumColumns = src.getNumColumnsLocal();
  U destNumRows = dest.getNumRowsLocal();
  U destNumColumns = dest.getNumColumnsLocal();

  std::vector<T>& srcVectorData = src.getVectorData();
  std::vector<T*>& srcMatrixData = src.getMatrixData();
  std::vector<T>& destVectorData = dest.getVectorData();
  std::vector<T*>& destMatrixData = dest.getMatrixData();

  U numElems = srcNumColumns*srcNumColumns;
  bool assembleFinder = false;
  if (static_cast<U>(destVectorData.size()) < numElems){
    assembleFinder = true;
    destVectorData.resize(numElems);
  }
  if (static_cast<U>(destMatrixData.size()) < srcNumColumns){
    assembleFinder = true;
    destMatrixData.resize(srcNumRows);
  }

  U counter{1};
  U srcOffset{0};
  U destOffset{0};
  U counter2{srcNumRows};
  for (U i=0; i<srcNumRows; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], counter*sizeof(T));
    U fillZeros = srcNumRows-counter;
    fillZerosContig(&destVectorData[destOffset+counter], fillZeros);
    srcOffset += counter;
    destOffset += counter2;
    counter++;
  }

  if (assembleFinder){
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    dest.setNumRowsLocal(srcNumRows);
    dest.setNumColumnsLocal(srcNumColumns);
    dest.setNumElems(numElems);
    UpperTriangular::_AssembleMatrix(destVectorData, destMatrixData, srcNumColumns, srcNumRows);
  }
  TAU_FSTOP(Serialize);
  return;
}

template<typename BigType, typename SmallType>
void Serializer<UpperTriangular, Square>::Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){
  TAU_FSTART(Serialize);

  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
  assert(rangeX == rangeY);

  U bigNumRows = big.getNumRowsLocal();
  U bigNumColumns = big.getNumColumnsLocal();

  std::vector<T>& bigVectorData = big.getVectorData();
  std::vector<T*>& bigMatrixData = big.getMatrixData();
  std::vector<T>& smallVectorData = small.getVectorData();
  std::vector<T*>& smallMatrixData = small.getMatrixData();

  U numElems = (dir ? ((bigNumColumns*(bigNumColumns+1))>>1) : rangeX*rangeX);
  U numColumns = (dir ? bigNumColumns : rangeX);
  bool assembleFinder = false;
  if (static_cast<U>((dir ? bigVectorData.size() : smallVectorData.size())) < numElems){
    assembleFinder = true;
    dir ? bigVectorData.resize(numElems) : smallVectorData.resize(numElems);
  }
  if (static_cast<U>((dir ? bigMatrixData.size() : smallVectorData.size())) < numColumns){
    assembleFinder = true;
    dir ? bigMatrixData.resize(numColumns) : smallMatrixData.resize(numColumns);
  }

  U bigMatOffset = ((cutDimensionXstart*(cutDimensionXstart+1))>>1);
  bigMatOffset += cutDimensionYstart;
  U bigMatCounter = cutDimensionXstart;
  U srcOffset = (dir ? 0 : bigMatOffset);
  U destOffset = (dir ? bigMatOffset : 0);
  auto& destVectorData = dir ? bigVectorData : smallVectorData;
  auto& srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*rangeY);
    destOffset += (dir ? (bigMatCounter+1) : rangeY);
    srcOffset += (dir ? rangeY : (bigMatCounter+1));
    bigMatCounter++;
  }

  if (assembleFinder){
    if (dir) {abort();}		// weird case that I want to check against
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    small.setNumRowsLocal(numColumns);
    small.setNumColumnsLocal(rangeY);
    small.setNumElems(numElems);
    // I am only providing Square here, not UT, because if UT, it would have aborted
    Square::_AssembleMatrix(destVectorData, smallMatrixData, numColumns, rangeY);
  }
  TAU_FSTOP(Serialize);
}

/* No reason for this method. Just use UT to little square
template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,UpperTriangular, Square>::Serialize(Matrix<T,U,UpperTriangular,Distributer>& src,
  Matrix<T,U,Square,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros, bool dir)
{
  std::cout << "Not updated. Only Square is. Makes no sense to change the implementation of all of these Structures until we think that the Square is right.\n";
  abort();
  if (dir == true)
  {
    std::cout << "Not finished yet. Complete when necessary\n";
    abort();
  }
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  assert(rangeX == rangeY);

  U numElems = rangeX*rangeX;
  if (dest.size() < numElems)
  {
    dest.resize(numElems);
  }

  U counter{dimensionX-cutDimensionYstart-1};
  U counter2{rangeX};
  U srcOffset = ((dimensionX*(dimensionX+1))>>1);
  U helper = dimensionX - cutDimensionYstart;
  srcOffset -= ((helper*(helper+1))>>1);		// Watch out for 64-bit rvalue implicit cast problems!
  srcOffset += (cutDimensionXstart-cutDimensionYstart);
  U destOffset{0};
  for (U i=0; i<rangeY; i++)
  {
    U fillZeros = rangeX - counter2;
    fillZerosContig(&destVectorData[destOffset], fillZeros);
    destOffset += fillZeros;
    memcpy(&dest[destOffset], &src[srcOffset], sizeof(T)*counter2);
    destOffset += counter2;
    srcOffset += (counter+1);
    counter--;
    counter2--;
  }
}
*/
  

template<typename SrcType, typename DestType>
void Serializer<UpperTriangular,Rectangular>::Serialize(SrcType& src, DestType& dest){
  TAU_FSTART(Serialize);

  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  // But now, I am going to have this call the Serializer from UT to Square, because thats what this will actually be doing
  // I tried a simple static_cast, but it didn't work, so now I will just copy code. Ugh! Fix later.
  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.getNumRowsLocal();
  U srcNumColumns = src.getNumColumnsLocal();

  std::vector<T>& srcVectorData = src.getVectorData();
  std::vector<T>& destVectorData = dest.getVectorData();
  std::vector<T*>& destMatrixData = dest.getMatrixData();

  U numElems = srcNumColumns*srcNumColumns;
  bool assembleFinder = false;
  if (static_cast<U>(destVectorData.size()) < numElems){
    assembleFinder = true;
    destVectorData.resize(numElems);
  }
  if (static_cast<U>(destMatrixData.size()) < srcNumColumns){
    assembleFinder = true;
    destMatrixData.resize(srcNumRows);
  }

  U counter{1};
  U srcOffset{0};
  U destOffset{0};
  U counter2{srcNumRows};
  for (U i=0; i<srcNumRows; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], counter*sizeof(T));
    U fillZeros = srcNumRows-counter;
    fillZerosContig(&destVectorData[destOffset+counter], fillZeros);
    srcOffset += counter;
    destOffset += counter2;
    counter++;
  }

  if (assembleFinder){
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    dest.setNumRowsLocal(srcNumRows);
    dest.setNumColumnsLocal(srcNumColumns);
    dest.setNumElems(numElems);
    UpperTriangular::_AssembleMatrix(destVectorData, destMatrixData, srcNumColumns, srcNumRows);
  }
  TAU_FSTOP(Serialize);
  return;
}


template<typename BigType, typename SmallType>
void Serializer<UpperTriangular,Rectangular>::Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){
  TAU_FSTART(Serialize);

  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  // But now, I am going to have this call the Serializer from UT to Square, because thats what this will actually be doing
  // I tried a simple static_cast, but it didn't work, so now I will just copy code. Ugh! Fix later.
  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  U bigNumRows = big.getNumRowsLocal();
  U bigNumColumns = big.getNumColumnsLocal();

  std::vector<T>& bigVectorData = big.getVectorData();
  std::vector<T*>& bigMatrixData = big.getMatrixData();
  std::vector<T>& smallVectorData = small.getVectorData();
  std::vector<T*>& smallMatrixData = small.getMatrixData();

  U numElems = (dir ? ((bigNumColumns*(bigNumColumns+1))>>1) : rangeX*rangeY);
  U numColumns = (dir ? bigNumColumns : rangeX);
  bool assembleFinder = false;
  if (static_cast<U>((dir ? bigVectorData.size() : smallVectorData.size())) < numElems){
    assembleFinder = true;
    dir ? bigVectorData.resize(numElems) : smallVectorData.resize(numElems);
  }
  if (static_cast<U>((dir ? bigMatrixData.size() : smallMatrixData.size())) < numColumns){
    assembleFinder = true;
    dir ? bigMatrixData.resize(numColumns) : smallMatrixData.resize(numColumns);
  }

  U bigMatOffset = ((cutDimensionXstart*(cutDimensionXstart+1))>>1);
  bigMatOffset += cutDimensionYstart;
  U bigMatCounter = cutDimensionXstart;
  U srcOffset = (dir ? 0 : bigMatOffset);
  U destOffset = (dir ? bigMatOffset : 0);
  auto& destVectorData = dir ? bigVectorData : smallVectorData;
  auto& srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*rangeY);
    destOffset += (dir ? (bigMatCounter+1) : rangeY);
    srcOffset += (dir ? rangeY : (bigMatCounter+1));
    bigMatCounter++;
  }

  if (assembleFinder){
    if (dir) {abort();}		// weird case that I want to check against
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    small.setNumRowsLocal(rangeY);
    small.setNumColumnsLocal(numColumns);
    small.setNumElems(numElems);
    // I am only providing Square here, not UT, because if UT, it would have aborted
    Square::_AssembleMatrix(destVectorData, smallMatrixData, numColumns, rangeY);
  }
  TAU_FSTOP(Serialize);
}


template<typename SrcType, typename DestType>
void Serializer<UpperTriangular,UpperTriangular>::Serialize(SrcType& src, DestType& dest){
  TAU_FSTART(Serialize);

  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.getNumRowsLocal();
  U srcNumColumns = src.getNumColumnsLocal();
  U srcNumElems = srcNumRows*srcNumColumns;

  std::vector<T>& srcVectorData = src.getVectorData();
  std::vector<T*>& srcMatrixData = src.getMatrixData();
  std::vector<T>& destVectorData = dest.getVectorData();
  std::vector<T*>& destMatrixData = dest.getMatrixData();

  bool assembleFinder = false;
  if (static_cast<U>(destVectorData.size()) < srcNumElems){
    assembleFinder = true;
    destVectorData.resize(srcNumElems);
  }
  if (static_cast<U>(destMatrixData.size()) < srcNumColumns){
    assembleFinder = true;
    destMatrixData.resize(srcNumColumns);
  }

  // direction doesn't matter here since no indexing here
  memcpy(&destVectorData[0], &srcVectorData[0], sizeof(T)*srcNumElems);

  if (assembleFinder){
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    // User can assume that everything except for global dimensions are set. If he needs global dimensions too, he can set them himself.
    dest.setNumRowsLocal(srcNumRows);
    dest.setNumColumnsLocal(srcNumColumns);
    dest.setNumElems(srcNumElems);
    UpperTriangular::_AssembleMatrix(destVectorData, destMatrixData, srcNumColumns, srcNumRows);
  }
  TAU_FSTOP(Serialize);
  return;
}


template<typename BigType, typename SmallType>
void Serializer<UpperTriangular,UpperTriangular>::Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                            typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){
  TAU_FSTART(Serialize);

  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
  assert(rangeX == rangeY);

  U bigNumColumns = big.getNumColumnsLocal();

  std::vector<T>& bigVectorData = big.getVectorData();
  std::vector<T*>& bigMatrixData = big.getMatrixData();
  std::vector<T>& smallVectorData = small.getVectorData();
  std::vector<T*>& smallMatrixData = small.getMatrixData();

  U numElems = (dir ? ((bigNumColumns*(bigNumColumns+1))>>1) : ((rangeX*(rangeX+1))>>1));
  U numColumns = (dir ? bigNumColumns : rangeX);
  bool assembleFinder = false;
  if (static_cast<U>((dir ? bigVectorData.size() : smallVectorData.size())) < numElems){
    assembleFinder = true;
    dir ? bigVectorData.resize(numElems) : smallVectorData.resize(numElems);
  }
  if (static_cast<U>((dir ? bigMatrixData.size() : smallMatrixData.size())) < numColumns){
    assembleFinder = true;
    dir ? bigMatrixData.resize(numColumns) : smallMatrixData.resize(numColumns);
  }

  U bigMatOffset = ((cutDimensionXstart*(cutDimensionXstart+1))>>1);
  bigMatOffset += cutDimensionYstart;
  U bigMatCounter = cutDimensionXstart;
  U smallMatOffset = 0;
  U smallMatCounter{1};
  U srcOffset = (dir ? smallMatOffset : bigMatOffset);
  U destOffset = (dir ? bigMatOffset : smallMatOffset);
  auto& destVectorData = dir ? bigVectorData : smallVectorData;
  auto& srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*smallMatCounter);
    destOffset += (dir ? (bigMatCounter+1) : smallMatCounter);
    srcOffset += (dir ? smallMatCounter : (bigMatCounter+1));
    bigMatCounter++;
    smallMatCounter++;
  }

  if (assembleFinder){
    if (dir) {abort();}
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    // User can assume that everything except for global dimensions are set. If he needs global dimensions too, he can set them himself.
    small.setNumRowsLocal(rangeY);
    small.setNumColumnsLocal(numColumns);	// no dir needed here due to abort above
    small.setNumElems(numElems);
    UpperTriangular::_AssembleMatrix(destVectorData, smallMatrixData, numColumns, rangeY);
  }
  TAU_FSTOP(Serialize);
  return;
}


template<typename SrcType, typename DestType>
void Serializer<LowerTriangular, Square>::Serialize(SrcType& src, DestType& dest){
  TAU_FSTART(Serialize);

  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.getNumRowsLocal();
  U srcNumColumns = src.getNumColumnsLocal();
  U destNumRows = dest.getNumRowsLocal();
  U destNumColumns = dest.getNumColumnsLocal();

  std::vector<T>& srcVectorData = src.getVectorData();
  std::vector<T*>& srcMatrixData = src.getMatrixData();
  std::vector<T>& destVectorData = dest.getVectorData();
  std::vector<T*>& destMatrixData = dest.getMatrixData();

  U numElems = srcNumColumns*srcNumColumns;
  bool assembleFinder = false;
  if (static_cast<U>(destVectorData.size()) < numElems){
    assembleFinder = true;
    destVectorData.resize(numElems);
  }
  if (static_cast<U>(destMatrixData.size()) < srcNumColumns){
    assembleFinder = true;
    destMatrixData.resize(srcNumColumns);
  }

  U counter{srcNumRows};
  U srcOffset{0};
  U destOffset{0};
  U counter2{srcNumColumns};
  for (U i=0; i<srcNumColumns; i++){
    fillZerosContig(&destVectorData[destOffset], i);
    memcpy(&destVectorData[destOffset+i], &srcVectorData[srcOffset], counter*sizeof(T));
    srcOffset += counter;
    destOffset += srcNumRows;
    counter--;
  }

  if (assembleFinder){
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    // User can assume that everything except for global dimensions are set. If he needs global dimensions too, he can set them himself.
    dest.setNumRowsLocal(srcNumRows);
    dest.setNumColumnsLocal(srcNumColumns);	// no dir needed here due to abort above
    dest.setNumElems(numElems);
    Square::_AssembleMatrix(destVectorData, destMatrixData, srcNumColumns, srcNumRows);	// again, no dir ? needed here
  }
  TAU_FSTOP(Serialize);
  return;
}



template<typename BigType, typename SmallType>
void Serializer<LowerTriangular,Square>::Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                   typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){
  TAU_FSTART(Serialize);

  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
  std::cout << "rangeX,rangeY - " << rangeX << " " << rangeY << std::endl;
  assert(rangeX == rangeY);

  U bigNumRows = big.getNumRowsLocal();
  U bigNumColumns = big.getNumColumnsLocal();

  std::vector<T>& bigVectorData = big.getVectorData();
  std::vector<T*>& bigMatrixData = big.getMatrixData();
  std::vector<T>& smallVectorData = small.getVectorData();
  std::vector<T*>& smallMatrixData = small.getMatrixData();

  U numElems = (dir ? ((bigNumColumns*(bigNumColumns+1))>>1) : rangeX*rangeX);
  U numColumns = (dir ? bigNumColumns : rangeX);
  bool assembleFinder = false;
  if (static_cast<U>((dir ? bigVectorData.size() : smallVectorData.size())) < numElems){
    assembleFinder = true;
    dir ? bigVectorData.resize(numElems) : smallVectorData.resize(numElems);
  }
  if (static_cast<U>((dir ? bigMatrixData.size() : smallMatrixData.size())) < numColumns){
    assembleFinder = true;
    dir ? bigMatrixData.resize(numColumns) : smallMatrixData.resize(numColumns);
  }

  U smallMatOffset = 0;
  U smallMatCounter = rangeY;
  U bigMatOffset = ((bigNumColumns*(bigNumColumns+1))>>1);
  U helper = bigNumColumns - cutDimensionXstart;
  bigMatOffset -= ((helper*(helper+1))>>1);
  bigMatOffset += (cutDimensionYstart-cutDimensionXstart);
  U bigMatCounter = bigNumRows-cutDimensionXstart;
  U srcOffset = (dir ? smallMatOffset : bigMatOffset);
  U destOffset = (dir ? bigMatOffset : smallMatOffset);
  auto& destVectorData = dir ? bigVectorData : smallVectorData;
  auto& srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*rangeY);
    destOffset += (dir ? (bigMatCounter-1) : smallMatCounter);
    srcOffset += (dir ? smallMatCounter : (bigMatCounter-1));
    bigMatCounter--;
  }
  if (assembleFinder){
    if (dir) {abort();}
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    // User can assume that everything except for global dimensions are set. If he needs global dimensions too, he can set them himself.
    small.setNumRowsLocal(rangeY);
    small.setNumColumnsLocal(numColumns);	// no dir needed here due to abort above
    small.setNumElems(numElems);
    Square::_AssembleMatrix(destVectorData, smallMatrixData, numColumns, rangeY);
  }
  TAU_FSTOP(Serialize);
  return;
}


/*
template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,LowerTriangular, Square>::Serialize(Matrix<T,U,LowerTriangular,Distributer>& src,
  Matrix<T,U,Square,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros, bool dir)
{
  std::cout << "Not updated. Only Square is. Makes no sense to change the implementation of all of these Structures until we think that the Square is right.\n";
  abort();
  if (dir == true)
  {
    std::cout << "Not finished yet. Complete when necessary\n";
    abort();
  }
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  assert(rangeX == rangeY);

  U numElems = rangeX*rangeX;
  if (dest.size() < numElems)
  {
    dest.resize(numElems);
  }

  U counter{cutDimensionYstart};
  U counter2{1};
  U srcOffset = ((counter*(counter+1))>>1);
  srcOffset += cutDimensionXstart;
  U destOffset{0};
  for (U i=0; i<rangeY; i++)
  {
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*counter2);
    destOffset += counter2;
    U fillSize = rangeX - counter2;
    fillZerosContig(&destVectorData[destOffset], fillSize);
    destOffset += fillSize;
    srcOffset += (counter+1);
    counter++;
    counter2++;
  }
}
*/

template<typename SrcType, typename DestType>
void Serializer<LowerTriangular,Rectangular>::Serialize(SrcType& src, DestType& dest){
  TAU_FSTART(Serialize);

  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  // But now, I am going to have this call the Serializer from UT to Square, because thats what this will actually be doing
  // I tried a simple static_cast, but it didn't work, so now I will just copy code. Ugh! Fix later.
  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.getNumRowsLocal();
  U srcNumColumns = src.getNumColumnsLocal();

  std::vector<T>& srcVectorData = src.getVectorData();
  std::vector<T>& destVectorData = dest.getVectorData();
  std::vector<T*>& destMatrixData = dest.getMatrixData();

  U numElems = srcNumColumns*srcNumRows;
  bool assembleFinder = false;
  if (static_cast<U>(destVectorData.size()) < numElems){
    assembleFinder = true;
    destVectorData.resize(numElems);
  }
  if (static_cast<U>(destMatrixData.size()) < srcNumColumns){
    assembleFinder = true;
    destMatrixData.resize(srcNumColumns);
  }

  U counter{srcNumRows};
  U srcOffset{0};
  U destOffset{0};
  for (U i=0; i<srcNumColumns; i++){
    fillZerosContig(&destVectorData[destOffset], i);
    memcpy(&destVectorData[destOffset+i], &srcVectorData[srcOffset], counter*sizeof(T));
    srcOffset += counter;
    destOffset += srcNumRows;
    counter--;
  }

  if (assembleFinder){
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    // User can assume that everything except for global dimensions are set. If he needs global dimensions too, he can set them himself.
    dest.setNumRowsLocal(srcNumRows);
    dest.setNumColumnsLocal(srcNumColumns);	// no dir needed here due to abort above
    dest.setNumElems(numElems);
    Square::_AssembleMatrix(destVectorData, destMatrixData, srcNumColumns, srcNumRows);	// again, no dir ? needed here
  }
  TAU_FSTOP(Serialize);
  return;
}

template<typename BigType, typename SmallType>
void Serializer<LowerTriangular,Rectangular>::Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){
  TAU_FSTART(Serialize);

  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  // But now, I am going to have this call the Serializer from UT to Square, because thats what this will actually be doing
  // I tried a simple static_cast, but it didn't work, so now I will just copy code. Ugh! Fix later.
  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  U bigNumRows = big.getNumRowsLocal();
  U bigNumColumns = big.getNumColumnsLocal();

  std::vector<T>& bigVectorData = big.getVectorData();
  std::vector<T*>& bigMatrixData = big.getMatrixData();
  std::vector<T>& smallVectorData = small.getVectorData();
  std::vector<T*>& smallMatrixData = small.getMatrixData();

  U numElems = (dir ? ((bigNumColumns*(bigNumColumns+1))>>1) : rangeY*rangeX);
  U numColumns = (dir ? bigNumColumns : rangeX);
  bool assembleFinder = false;
  if (static_cast<U>((dir ? bigVectorData.size() : smallVectorData.size())) < numElems){
    assembleFinder = true;
    dir ? bigVectorData.resize(numElems) : smallVectorData.resize(numElems);
  }
  if (static_cast<U>((dir ? bigMatrixData.size() : smallMatrixData.size())) < numColumns){
    assembleFinder = true;
    dir ? bigMatrixData.resize(numColumns) : smallMatrixData.resize(numColumns);
  }

  U smallMatOffset = 0;
  U smallMatCounter = rangeY;
  U bigMatOffset = ((bigNumColumns*(bigNumColumns+1))>>1);
  U helper = bigNumColumns - cutDimensionXstart;
  bigMatOffset -= ((helper*(helper+1))>>1);
  bigMatOffset += (cutDimensionYstart-cutDimensionXstart);
  U bigMatCounter = bigNumRows-cutDimensionXstart;
  U srcOffset = (dir ? smallMatOffset : bigMatOffset);
  U destOffset = (dir ? bigMatOffset : smallMatOffset);
  auto& destVectorData = dir ? bigVectorData : smallVectorData;
  auto& srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*rangeY);
    destOffset += (dir ? (bigMatCounter-1) : smallMatCounter);
    srcOffset += (dir ? smallMatCounter : (bigMatCounter-1));
    bigMatCounter--;
  }
  if (assembleFinder){
    if (dir) {abort();}
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    // User can assume that everything except for global dimensions are set. If he needs global dimensions too, he can set them himself.
    small.setNumRowsLocal(rangeY);
    small.setNumColumnsLocal(numColumns);	// no dir needed here due to abort above
    small.setNumElems(numElems);
    Square::_AssembleMatrix(destVectorData, smallMatrixData, numColumns, rangeY);
  }
  TAU_FSTOP(Serialize);
  return;
}


template<typename SrcType, typename DestType>
void Serializer<LowerTriangular,LowerTriangular>::Serialize(SrcType& src, DestType& dest){
  TAU_FSTART(Serialize);

  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.getNumRowsLocal();
  U srcNumColumns = src.getNumColumnsLocal();
  U srcNumElems = srcNumRows*srcNumColumns;

  std::vector<T>& srcVectorData = src.getVectorData();
  std::vector<T*>& srcMatrixData = src.getMatrixData();
  std::vector<T>& destVectorData = dest.getVectorData();
  std::vector<T*>& destMatrixData = dest.getMatrixData();

  bool assembleFinder = false;
  if (static_cast<U>(destVectorData.size()) < srcNumElems){
    assembleFinder = true;
    destVectorData.resize(srcNumElems);
  }
  if (static_cast<U>(destMatrixData.size()) < srcNumColumns){
    assembleFinder = true;
    destMatrixData.resize(srcNumColumns);
  }

  // direction doesn't matter here since no indexing here
  memcpy(&destVectorData[0], &srcVectorData[0], sizeof(T)*srcNumElems);

  if (assembleFinder){
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    // User can assume that everything except for global dimensions are set. If he needs global dimensions too, he can set them himself.
    dest.setNumRowsLocal(srcNumRows);
    dest.setNumColumnsLocal(srcNumColumns);
    dest.setNumElems(srcNumElems);
    LowerTriangular::_AssembleMatrix(destVectorData, destMatrixData, srcNumColumns, srcNumRows);
  }
  TAU_FSTOP(Serialize);
  return;
}

template<typename BigType, typename SmallType>
void Serializer<LowerTriangular,LowerTriangular>::Serialize(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                            typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){
  TAU_FSTART(Serialize);

  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
  assert(rangeX == rangeY);

  U bigNumRows = big.getNumRowsLocal();
  U bigNumColumns = big.getNumColumnsLocal();

  std::vector<T>& bigVectorData = big.getVectorData();
  std::vector<T*>& bigMatrixData = big.getMatrixData();
  std::vector<T>& smallVectorData = small.getVectorData();
  std::vector<T*>& smallMatrixData = small.getMatrixData();

  U numElems = (dir ? ((bigNumColumns*(bigNumColumns+1))>>1) : ((rangeX*(rangeX+1))>>1));
  U numColumns = (dir ? bigNumColumns : rangeX);
  bool assembleFinder = false;
  if (static_cast<U>((dir ? bigVectorData.size() : smallVectorData.size())) < numElems){
    assembleFinder = true;
    dir ? bigVectorData.resize(numElems) : smallVectorData.resize(numElems);
  }
  if (static_cast<U>((dir ? bigMatrixData.size() : smallMatrixData.size())) < numColumns){
    assembleFinder = true;
    dir ? bigMatrixData.resize(numColumns) : smallMatrixData.resize(numColumns);
  }

  U smallMatOffset = 0;
  U smallMatCounter = rangeY;
  U bigMatOffset = ((bigNumColumns*(bigNumColumns+1))>>1);
  U helper = bigNumColumns - cutDimensionXstart;
  bigMatOffset -= ((helper*(helper+1))>>1);
  bigMatOffset += (cutDimensionYstart-cutDimensionXstart);
  U bigMatCounter = bigNumRows-cutDimensionXstart;
  U srcOffset = (dir ? smallMatOffset : bigMatOffset);
  U destOffset = (dir ? bigMatOffset : smallMatOffset);
  auto& destVectorData = dir ? bigVectorData : smallVectorData;
  auto& srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeY; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*smallMatCounter);
    destOffset += (dir ? bigMatCounter : smallMatCounter);
    srcOffset += (dir ? smallMatCounter : bigMatCounter);
    bigMatCounter--;
    smallMatCounter--;
  }

  if (assembleFinder){
    if (dir) {abort();}
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    // User can assume that everything except for global dimensions are set. If he needs global dimensions too, he can set them himself.
    small.setNumRowsLocal(rangeY);
    small.setNumColumnsLocal(numColumns);	// no dir needed here due to abort above
    small.setNumElems(numElems);
    LowerTriangular::_AssembleMatrix(destVectorData, smallMatrixData, numColumns, rangeY);
  }
  TAU_FSTOP(Serialize);
  return;
}
