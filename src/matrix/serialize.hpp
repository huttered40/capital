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
void serialize<square,square>::invoke(SrcType& src, DestType& dest){

  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();
  U srcNumElems = srcNumRows*srcNumColumns;

  T* srcVectorData = src.data();
  T* destVectorData = dest.data();
  // direction doesn't matter here since no indexing here
  memcpy(destVectorData, srcVectorData, sizeof(T)*srcNumElems);
  return;
}

template<typename BigType, typename SmallType>
void serialize<square,square>::invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart,
                                           typename BigType::DimensionType cutDimensionXend, typename BigType::DimensionType cutDimensionYstart,
                                           typename BigType::DimensionType cutDimensionYend, bool dir){

  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
  U bigNumRows = big.num_rows_local();
  U bigNumColumns = big.num_columns_local();

  T* bigVectorData = big.data();
  T* smallVectorData = small.data();

  // We assume that if dir==true, the user passed in a destination matrix that is properly sized
  // If I find a use case where that is not true, then I can pass in big x and y dimensions, but I do not wan to do that.
  // In most cases, the lae matri will be known and we are simply trying to fill in the smaller into the existing bigger

  U destIndex = (dir ? cutDimensionYstart+bigNumRows*cutDimensionXstart : 0);
  U srcIndex = (dir ? 0 : cutDimensionYstart+bigNumRows*cutDimensionXstart);
  U srcCounter = (dir ? rangeY : bigNumRows);
  U destCounter = (dir ? bigNumRows : rangeY);
  T* destVectorData = dir ? bigVectorData : smallVectorData;
  T* srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destIndex], &srcVectorData[srcIndex], sizeof(T)*rangeY);		// rangeX is fine. blocks of size rangeX are still being copied.
    destIndex += destCounter;
    srcIndex += srcCounter;
  }
}

template<typename SrcType, typename DestType>
void serialize<square,rect>::invoke(SrcType& src, DestType& dest){
  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  std::cout << "Not fully implemented yet in Matrixserialize square -> rect\n";
  assert(0);
}

template<typename BigType, typename SmallType>
void serialize<square,rect>::invoke(BigType& src, SmallType& dest, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                               typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){
  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  std::cout << "Not fully implemented yet in Matrixserialize square -> rect\n";
  assert(0);
}

template<typename SrcType, typename DestType>
void serialize<square,uppertri>::invoke(SrcType& src, DestType& dest){

  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();

  T* srcVectorData = src.data();
  T* destVectorData = dest.data();

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
  return;
}

/*
template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void serialize<T,U,square, uppertri>::invoke(Matrix<T,U,square,Distributer>& src,
  Matrix<T,U,uppertri,Distributer>& dest, bool fillZeros, bool dir)
{
  std::cout << "Not updated. Only square is. Makes no sense to change the implementation of all of these Structures until we think that the square is right.\n";
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
void serialize<square,uppertri>::invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                   typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){

  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
//  assert(rangeX == rangeY);

  U bigNumRows = big.num_rows_local();
  U bigNumColumns = big.num_columns_local();

  T* bigVectorData = big.data();
  T* smallVectorData = small.data();

  // We assume that if dir==true, the user passed in a destination matrix that is properly sized
  // If I find a use case where that is not true, then I can pass in big x and y dimensions, but I do not wan to do that.
  // In most cases, the lae matri will be known and we are simply trying to fill in the smaller into the existing bigger

  U destIndex = (dir ? cutDimensionYstart+bigNumRows*cutDimensionXstart : 0);
  U counter{1};
  U srcIndexSave = (dir ? 0 : cutDimensionYstart+bigNumRows*cutDimensionXstart);
  T* destVectorData = (dir ? bigVectorData : smallVectorData);
  T* srcVectorData = (dir ? smallVectorData : bigVectorData);
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destIndex], &srcVectorData[srcIndexSave], sizeof(T)*counter);
    destIndex += (dir ? bigNumRows : counter);
    srcIndexSave += (dir ? counter : bigNumRows);
    counter++;
  }
}

/*
template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void serialize<T,U,square, uppertri>::invoke(Matrix<T,U,square,Distributer>& src,
  Matrix<T,U,uppertri,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros, bool dir)
{
  std::cout << "Not updated. Only square is. Makes no sense to change the implementation of all of these Structures until we think that the square is right.\n";
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
void serialize<square,lowertri>::invoke(SrcType& src, DestType& dest){

  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();
  U destNumRows = dest.num_rows_local();
  U destNumColumns = dest.num_columns_local();

  T* srcVectorData = src.data();
  T* destVectorData = dest.data();

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
  return;
}

/*
template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void serialize<T,U,square, lowertri>::invoke(Matrix<T,U,square,Distributer>& src,
  Matrix<T,U,lowertri,Distributer>& dest, bool fillZeros, bool dir)
{
  std::cout << "Not updated. Only square is. Makes no sense to change the implementation of all of these Structures until we think that the square is right.\n";
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
void serialize<square,lowertri>::invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                   typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){

  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
//  assert(rangeX == rangeY);

  U bigNumRows = big.num_rows_local();
  U bigNumColumns = big.num_columns_local();

  T* bigVectorData = big.data();
  T* smallVectorData = small.data();

  U counter{rangeY};
  U srcOffset = (dir ? 0 : cutDimensionYstart+bigNumRows*cutDimensionXstart);
  U destOffset = (dir ? cutDimensionYstart+bigNumRows*cutDimensionXstart  : 0);
  T* destVectorData = dir ? bigVectorData : smallVectorData;
  T* srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*counter);
    destOffset += (dir ? bigNumRows+1 : counter);
    srcOffset += (dir ? counter : bigNumRows+1);
    counter--;
  }
}

/*
template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void serialize<T,U,square, lowertri>::invoke(Matrix<T,U,square,Distributer>& src,
  Matrix<T,U,lowertri,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros, bool dir)
{
  std::cout << "Not updated. Only square is. Makes no sense to change the implementation of all of these Structures until we think that the square is right.\n";
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
void serialize<rect,square>::invoke(SrcType& src, DestType& dest){
  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  std::cout << "Not fully implemented yet in Matrixserialize for rect -> square\n";
  return;
}

template<typename BigType, typename SmallType>
void serialize<rect,square>::invoke(BigType& src, SmallType& dest, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                               typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){
  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  std::cout << "Not fully implemented yet in Matrixserialize for rect -> square\n";
  return;
}

template<typename SrcType, typename DestType>
void serialize<rect,rect>::invoke(SrcType& src, DestType& dest){

  // For now, just call square counterpart, it should be the same --- Actually I can't unless I try to do a weird cast.
  // Annoying code bloat here
  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();
  U srcNumElems = srcNumRows*srcNumColumns;

  T* srcVectorData = src.data();
  T* destVectorData = dest.data();

  // direction doesn't matter here since no indexing here
  memcpy(&destVectorData[0], &srcVectorData[0], sizeof(T)*srcNumElems);
  return;
}

template<typename BigType, typename SmallType>
void serialize<rect,rect>::invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                    typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){

  // For now, just call square counterpart, it should be the same --- Actually I can't unless I try to do a weird cast.
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
  U bigNumRows = big.num_rows_local();
  U bigNumColumns = big.num_columns_local();

  T* bigVectorData = big.data();
  T* smallVectorData = small.data();

  U destIndex = (dir ? cutDimensionYstart+bigNumRows*cutDimensionXstart : 0);
  U srcIndex = (dir ? 0 : cutDimensionYstart+bigNumRows*cutDimensionXstart);
  U srcCounter = (dir ? rangeY : bigNumRows);
  U destCounter = (dir ? bigNumRows : rangeY);
  T* destVectorData = dir ? bigVectorData : smallVectorData;
  T* srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destIndex], &srcVectorData[srcIndex], sizeof(T)*rangeY);		// rangeX is fine. blocks of size rangeX are still being copied.
    destIndex += destCounter;
    srcIndex += srcCounter;
  }
  return;
}

template<typename SrcType, typename DestType>
void serialize<rect,uppertri>::invoke(SrcType& src, DestType& dest){
  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  std::cout << "Not fully implemented yet\n";
  return;
}

template<typename BigType, typename SmallType>
void serialize<rect,uppertri>::invoke(BigType& src, SmallType& dest, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){
  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  std::cout << "Not fully implemented yet\n";
  return;
}

template<typename SrcType, typename DestType>
void serialize<rect,lowertri>::invoke(SrcType& src, DestType& dest){
  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  std::cout << "Not fully implemented yet\n";
  return;
}

template<typename BigType, typename SmallType>
void serialize<rect,lowertri>::invoke(BigType& src, SmallType& dest, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){
  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  std::cout << "Not fully implemented yet\n";
  return;
}


template<typename SrcType>
void serialize<uppertri,square>::invoke(SrcType& src){

  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();
  U destNumRows = dest.num_rows_local();
  U destNumColumns = dest.num_columns_local();

  T* srcVectorData = src.scratch();
  T* destVectorData = src.pad();

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
  return;
}

template<typename SrcType, typename DestType>
void serialize<uppertri,square>::invoke(SrcType& src, DestType& dest){

  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();
  U destNumRows = dest.num_rows_local();
  U destNumColumns = dest.num_columns_local();

  T* srcVectorData = src.data();
  T* destVectorData = dest.data();

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
}

template<typename BigType, typename SmallType>
void serialize<uppertri,square>::invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){

  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
//  assert(rangeX == rangeY);

  U bigNumRows = big.num_rows_local();
  U bigNumColumns = big.num_columns_local();

  T* bigVectorData = big.data();
  T* smallVectorData = small.data();

  U bigMatOffset = ((cutDimensionXstart*(cutDimensionXstart+1))>>1);
  bigMatOffset += cutDimensionYstart;
  U bigMatCounter = cutDimensionXstart;
  U srcOffset = (dir ? 0 : bigMatOffset);
  U destOffset = (dir ? bigMatOffset : 0);
  T* destVectorData = dir ? bigVectorData : smallVectorData;
  T* srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*rangeY);
    destOffset += (dir ? (bigMatCounter+1) : rangeY);
    srcOffset += (dir ? rangeY : (bigMatCounter+1));
    bigMatCounter++;
  }
}

/* No reason for this method. Just use UT to little square
template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void serialize<T,U,uppertri, square>::invoke(Matrix<T,U,uppertri,Distributer>& src,
  Matrix<T,U,square,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros, bool dir)
{
  std::cout << "Not updated. Only square is. Makes no sense to change the implementation of all of these Structures until we think that the square is right.\n";
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
  
template<typename SrcType>
void serialize<uppertri,rect>::invoke(SrcType& src){

  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  // But now, I am going to have this call the serialize from UT to square, because thats what this will actually be doing
  // I tried a simple static_cast, but it didn't work, so now I will just copy code. Ugh! Fix later.
  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();

  T* srcVectorData = src.scratch();
  T* destVectorData = src.pad();

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
  return;
}

template<typename SrcType, typename DestType>
void serialize<uppertri,rect>::invoke(SrcType& src, DestType& dest){

  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  // But now, I am going to have this call the serialize from UT to square, because thats what this will actually be doing
  // I tried a simple static_cast, but it didn't work, so now I will just copy code. Ugh! Fix later.
  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();

  T* srcVectorData = src.data();
  T* destVectorData = dest.data();

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
  return;
}


template<typename BigType, typename SmallType>
void serialize<uppertri,rect>::invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                      typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){

  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  // But now, I am going to have this call the serialize from UT to square, because thats what this will actually be doing
  // I tried a simple static_cast, but it didn't work, so now I will just copy code. Ugh! Fix later.
  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  U bigNumColumns = big.num_columns_local();

  T* bigVectorData = big.data();
  T* smallVectorData = small.data();

  U bigMatOffset = ((cutDimensionXstart*(cutDimensionXstart+1))>>1);
  bigMatOffset += cutDimensionYstart;
  U bigMatCounter = cutDimensionXstart;
  U srcOffset = (dir ? 0 : bigMatOffset);
  U destOffset = (dir ? bigMatOffset : 0);
  T* destVectorData = dir ? bigVectorData : smallVectorData;
  T* srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*rangeY);
    destOffset += (dir ? (bigMatCounter+1) : rangeY);
    srcOffset += (dir ? rangeY : (bigMatCounter+1));
    bigMatCounter++;
  }
}


template<typename SrcType, typename DestType>
void serialize<uppertri,uppertri>::invoke(SrcType& src, DestType& dest){

  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();
  U srcNumElems = srcNumRows*srcNumColumns;

  T* srcVectorData = src.data();
  T* destVectorData = dest.data();

  // direction doesn't matter here since no indexing here
  memcpy(&destVectorData[0], &srcVectorData[0], sizeof(T)*srcNumElems);
  return;
}


template<typename BigType, typename SmallType>
void serialize<uppertri,uppertri>::invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                            typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){

  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
//  assert(rangeX == rangeY);

  U bigNumColumns = big.num_columns_local();

  T* bigVectorData = big.data();
  T* smallVectorData = small.data();

  U bigMatOffset = ((cutDimensionXstart*(cutDimensionXstart+1))>>1);
  bigMatOffset += cutDimensionYstart;
  U bigMatCounter = cutDimensionXstart;
  U smallMatOffset = 0;
  U smallMatCounter{1};
  U srcOffset = (dir ? smallMatOffset : bigMatOffset);
  U destOffset = (dir ? bigMatOffset : smallMatOffset);
  T* destVectorData = dir ? bigVectorData : smallVectorData;
  T* srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*smallMatCounter);
    destOffset += (dir ? (bigMatCounter+1) : smallMatCounter);
    srcOffset += (dir ? smallMatCounter : (bigMatCounter+1));
    bigMatCounter++;
    smallMatCounter++;
  }
  return;
}


template<typename SrcType>
void serialize<lowertri, square>::invoke(SrcType& src){

  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();
  U destNumRows = dest.num_rows_local();
  U destNumColumns = dest.num_columns_local();

  T* srcVectorData = src.scratch();
  T* destVectorData = src.pad();

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
  return;
}


template<typename SrcType, typename DestType>
void serialize<lowertri, square>::invoke(SrcType& src, DestType& dest){

  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();
  U destNumRows = dest.num_rows_local();
  U destNumColumns = dest.num_columns_local();

  T* srcVectorData = src.data();
  T* destVectorData = src.data();

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
  return;
}



template<typename BigType, typename SmallType>
void serialize<lowertri,square>::invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                   typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){

  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
  std::cout << "rangeX,rangeY - " << rangeX << " " << rangeY << std::endl;
//  assert(rangeX == rangeY);

  U bigNumRows = big.num_rows_local();
  U bigNumColumns = big.num_columns_local();

  T* bigVectorData = big.data();
  T* smallVectorData = small.data();

  U smallMatOffset = 0;
  U smallMatCounter = rangeY;
  U bigMatOffset = ((bigNumColumns*(bigNumColumns+1))>>1);
  U helper = bigNumColumns - cutDimensionXstart;
  bigMatOffset -= ((helper*(helper+1))>>1);
  bigMatOffset += (cutDimensionYstart-cutDimensionXstart);
  U bigMatCounter = bigNumRows-cutDimensionXstart;
  U srcOffset = (dir ? smallMatOffset : bigMatOffset);
  U destOffset = (dir ? bigMatOffset : smallMatOffset);
  T* destVectorData = dir ? bigVectorData : smallVectorData;
  T* srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*rangeY);
    destOffset += (dir ? (bigMatCounter-1) : smallMatCounter);
    srcOffset += (dir ? smallMatCounter : (bigMatCounter-1));
    bigMatCounter--;
  }
  return;
}


/*
template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void serialize<T,U,lowertri, square>::invoke(Matrix<T,U,lowertri,Distributer>& src,
  Matrix<T,U,square,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros, bool dir)
{
  std::cout << "Not updated. Only square is. Makes no sense to change the implementation of all of these Structures until we think that the square is right.\n";
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


template<typename SrcType>
void serialize<lowertri,rect>::invoke(SrcType& src){

  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  // But now, I am going to have this call the serialize from UT to square, because thats what this will actually be doing
  // I tried a simple static_cast, but it didn't work, so now I will just copy code. Ugh! Fix later.
  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();

  T* srcVectorData = src.scratch();
  T* destVectorData = src.pad();

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
}

template<typename SrcType, typename DestType>
void serialize<lowertri,rect>::invoke(SrcType& src, DestType& dest){

  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  // But now, I am going to have this call the serialize from UT to square, because thats what this will actually be doing
  // I tried a simple static_cast, but it didn't work, so now I will just copy code. Ugh! Fix later.
  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();

  T* srcVectorData = src.data();
  T* destVectorData = dest.data();

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
  return;
}

template<typename BigType, typename SmallType>
void serialize<lowertri,rect>::invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){

  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  // But now, I am going to have this call the serialize from UT to square, because thats what this will actually be doing
  // I tried a simple static_cast, but it didn't work, so now I will just copy code. Ugh! Fix later.
  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  U bigNumRows = big.num_rows_local();
  U bigNumColumns = big.num_columns_local();

  T* bigVectorData = big.data();
  T* smallVectorData = small.data();

  U smallMatOffset = 0;
  U smallMatCounter = rangeY;
  U bigMatOffset = ((bigNumColumns*(bigNumColumns+1))>>1);
  U helper = bigNumColumns - cutDimensionXstart;
  bigMatOffset -= ((helper*(helper+1))>>1);
  bigMatOffset += (cutDimensionYstart-cutDimensionXstart);
  U bigMatCounter = bigNumRows-cutDimensionXstart;
  U srcOffset = (dir ? smallMatOffset : bigMatOffset);
  U destOffset = (dir ? bigMatOffset : smallMatOffset);
  T* destVectorData = dir ? bigVectorData : smallVectorData;
  T* srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*rangeY);
    destOffset += (dir ? (bigMatCounter-1) : smallMatCounter);
    srcOffset += (dir ? smallMatCounter : (bigMatCounter-1));
    bigMatCounter--;
  }
  return;
}


template<typename SrcType, typename DestType>
void serialize<lowertri,lowertri>::invoke(SrcType& src, DestType& dest){

  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();
  U srcNumElems = srcNumRows*srcNumColumns;

  T* srcVectorData = src.data();
  T* destVectorData = dest.data();

  // direction doesn't matter here since no indexing here
  memcpy(&destVectorData[0], &srcVectorData[0], sizeof(T)*srcNumElems);
}

template<typename BigType, typename SmallType>
void serialize<lowertri,lowertri>::invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                            typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){

  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
  assert(rangeX == rangeY);

  U bigNumRows = big.num_rows_local();
  U bigNumColumns = big.num_columns_local();

  T* bigVectorData = big.data();
  T* smallVectorData = small.data();

  U smallMatOffset = 0;
  U smallMatCounter = rangeY;
  U bigMatOffset = ((bigNumColumns*(bigNumColumns+1))>>1);
  U helper = bigNumColumns - cutDimensionXstart;
  bigMatOffset -= ((helper*(helper+1))>>1);
  bigMatOffset += (cutDimensionYstart-cutDimensionXstart);
  U bigMatCounter = bigNumRows-cutDimensionXstart;
  U srcOffset = (dir ? smallMatOffset : bigMatOffset);
  U destOffset = (dir ? bigMatOffset : smallMatOffset);
  T* destVectorData = dir ? bigVectorData : smallVectorData;
  T* srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeY; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*smallMatCounter);
    destOffset += (dir ? bigMatCounter : smallMatCounter);
    srcOffset += (dir ? smallMatCounter : bigMatCounter);
    bigMatCounter--;
    smallMatCounter--;
  }
  return;
}
