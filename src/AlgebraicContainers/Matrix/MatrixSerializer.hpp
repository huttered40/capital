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
static void fillZerosContig(T* addr, U size)
{
  for (U i=0; i<size; i++)
  {
    addr[i] = 0;
  }
}


template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureSquare>::Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src,
  Matrix<T,U,MatrixStructureSquare,Distributer>& dest)
{
  U srcNumRows = src.getNumRowsLocal();
  U srcNumColumns = src.getNumColumnsLocal();
  U srcNumElems = srcNumRows*srcNumColumns;

  std::vector<T>& srcVectorData = src.getVectorData();
  std::vector<T*>& srcMatrixData = src.getMatrixData();
  std::vector<T>& destVectorData = dest.getVectorData();
  std::vector<T*>& destMatrixData = dest.getMatrixData();

  bool assembleFinder = false;
  if (static_cast<U>(destVectorData.size()) < srcNumElems)
  {
    assembleFinder = true;
    destVectorData.resize(srcNumElems);
  }
  if (static_cast<U>(destMatrixData.size()) < srcNumRows)
  {
    assembleFinder = true;
    destMatrixData.resize(srcNumRows);
  }

  // direction doesn't matter here since no indexing here
  memcpy(&destVectorData[0], &srcVectorData[0], sizeof(T)*srcNumElems);

  if (assembleFinder)
  {
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    // User can assume that everything except for global dimensions are set. If he needs global dimensions too, he can set them himself.
    dest.setNumRowsLocal(srcNumRows);
    dest.setNumColumnsLocal(srcNumColumns);
    dest.setNumElems(srcNumElems);
    MatrixStructureSquare<T,U,Distributer>::AssembleMatrix(destVectorData, destMatrixData, srcNumColumns, srcNumRows);
  }
  return;
}
  
template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureSquare>::Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& big,
  Matrix<T,U,MatrixStructureSquare,Distributer>& small, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir)
{
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
  assert(rangeX == rangeY);

  U bigNumRows = big.getNumRowsLocal();
  U bigNumColumns = big.getNumColumnsLocal();
  U smallNumRows = small.getNumRowsLocal();
  U smallNumColumns = small.getNumColumnsLocal();

  std::vector<T>& bigVectorData = big.getVectorData();
  std::vector<T*>& bigMatrixData = big.getMatrixData();
  std::vector<T>& smallVectorData = small.getVectorData();
  std::vector<T*>& smallMatrixData = small.getMatrixData();

  // We assume that if dir==true, the user passed in a destination matrix that is properly sized
  // If I find a use case where that is not true, then I can pass in big x and y dimensions, but I do not wan to do that.
  // In most cases, the lae matri will be known and we are simply trying to fill in the smaller into the existing bigger

  U numElems = (dir ? bigNumRows*bigNumColumns : rangeX*rangeY);
  U numRows = (dir ? bigNumRows : rangeY);
  bool assembleFinder = false;
  if (static_cast<U>((dir ? bigVectorData.size() : smallVectorData.size())) < numElems)
  {
    assembleFinder = true;
    dir ? bigVectorData.resize(numElems) : smallVectorData.resize(numElems);
  }
  if (static_cast<U>((dir ? bigMatrixData.size() : smallMatrixData.size())) < numRows)
  {
    assembleFinder = true;
    dir ? bigMatrixData.resize(numRows) : smallMatrixData.resize(numRows);
  }

  U destIndex = (dir ? cutDimensionYstart*bigNumColumns+cutDimensionXstart : 0);
  U srcIndex = (dir ? 0 : cutDimensionYstart*bigNumColumns+cutDimensionXstart);
  U srcCounter = (dir ? rangeX : bigNumColumns);
  U destCounter = (dir ? bigNumColumns : rangeX);
  auto& destVectorData = dir ? bigVectorData : smallVectorData;
  auto& srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeY; i++)					// rangeY is fine.
  {
    memcpy(&destVectorData[destIndex], &srcVectorData[srcIndex], sizeof(T)*rangeX);		// rangeX is fine. blocks of size rangeX are still being copied.
    destIndex += destCounter;
    srcIndex += srcCounter;
  }

  if (assembleFinder)
  {
    if (dir) {abort();}		// weird case that I want to check against
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    small.setNumRowsLocal(numRows);
    small.setNumColumnsLocal(rangeX);
    small.setNumElems(numElems);
    MatrixStructureSquare<T,U,Distributer>::AssembleMatrix(destVectorData, smallMatrixData, rangeX, numRows);
  }
}


template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureUpperTriangular>::Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src,
  Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& dest)
{
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
  if (static_cast<U>(destVectorData.size()) < numElems)
  {
    assembleFinder = true;
    destVectorData.resize(numElems);
  }
  if (static_cast<U>(destMatrixData.size()) < srcNumRows)
  {
    assembleFinder = true;
    destMatrixData.resize(srcNumRows);
  }

  U counter{srcNumColumns};
  U srcOffset{0};
  U destOffset{0};
  U counter2{srcNumColumns+1};

  for (U i=0; i<srcNumRows; i++)
  {
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], counter*sizeof(T));
    srcOffset += counter2;
    destOffset += counter;
    counter--;
  }

  if (assembleFinder)
  {
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    dest.setNumRowsLocal(srcNumRows);
    dest.setNumColumnsLocal(srcNumColumns);
    dest.setNumElems(numElems);
    MatrixStructureUpperTriangular<T,U,Distributer>::AssembleMatrix(destVectorData, destMatrixData, srcNumColumns, srcNumRows);
  }
  return;
}

/*
template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureUpperTriangular>::Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src,
  Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& dest, bool fillZeros, bool dir)
{
  std::cout << "Not updated. Only MatrixStructureSquare is. Makes no sense to change the implementation of all of these Structures until we think that the Square is right.\n";
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


template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureUpperTriangular>::Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& big,
  Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& small, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir)
{
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
  assert(rangeX == rangeY);

  U bigNumRows = big.getNumRowsLocal();
  U bigNumColumns = big.getNumColumnsLocal();
  U smallNumRows = small.getNumRowsLocal();
  U smallNumColumns = small.getNumColumnsLocal();

  std::vector<T>& bigVectorData = big.getVectorData();
  std::vector<T*>& bigMatrixData = big.getMatrixData();
  std::vector<T>& smallVectorData = small.getVectorData();
  std::vector<T*>& smallMatrixData = small.getMatrixData();

  // We assume that if dir==true, the user passed in a destination matrix that is properly sized
  // If I find a use case where that is not true, then I can pass in big x and y dimensions, but I do not wan to do that.
  // In most cases, the lae matri will be known and we are simply trying to fill in the smaller into the existing bigger

  U numElems = (dir ? bigNumRows*bigNumRows : ((rangeX*(rangeX+1))>>1));
  U numRows = (dir ? bigNumRows : rangeX);
  bool assembleFinder = false;
  if ((dir ? bigVectorData.size() : smallVectorData.size()) < numElems)
  {
    assembleFinder = true;
    dir ? bigVectorData.resize(numElems) : smallVectorData.resize(numElems);
  }
  if ((dir ? bigMatrixData.size() : smallMatrixData.size()) < numRows)
  {
    assembleFinder = true;
    dir ? bigMatrixData.resize(numRows) : smallMatrixData.resize(numRows);
  }

  U destIndex = (dir ? cutDimensionYstart*bigNumColumns+cutDimensionXstart : 0);
  U counter{rangeX};
  U srcIndexSave = (dir ? 0 : cutDimensionYstart*bigNumColumns+cutDimensionXstart);
  auto& destVectorData = (dir ? bigVectorData : smallVectorData);
  auto& srcVectorData = (dir ? smallVectorData : bigVectorData);
  for (U i=0; i<rangeY; i++)
  {
    memcpy(&destVectorData[destIndex], &srcVectorData[srcIndexSave], sizeof(T)*counter);
    destIndex += (dir ? (bigNumColumns+1) : counter);
    srcIndexSave += (dir ? counter : (bigNumColumns+1));
    counter--;
  }

  if (assembleFinder)
  {
    if (dir) {abort();}		// weird case that I want to check against
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    small.setNumRowsLocal(numRows);
    small.setNumColumnsLocal(rangeX);
    small.setNumElems(numElems);
    // I am only providing UT here, not square, because if square, it would have aborted
    MatrixStructureUpperTriangular<T,U,Distributer>::AssembleMatrix(destVectorData, smallMatrixData, rangeX, numRows);
  }
}

/*
template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureUpperTriangular>::Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src,
  Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros, bool dir)
{
  std::cout << "Not updated. Only MatrixStructureSquare is. Makes no sense to change the implementation of all of these Structures until we think that the Square is right.\n";
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


template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureLowerTriangular>::Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src,
  Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& dest)
{
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
  if (static_cast<U>(destVectorData.size()) < numElems)
  {
    assembleFinder = true;
    destVectorData.resize(numElems);
  }
  if (static_cast<U>(destMatrixData.size()) < srcNumRows)
  {
    assembleFinder = true;
    destMatrixData.resize(srcNumRows);
  }

  U counter{1};
  U srcOffset{0};
  U destOffset{0};
  U counter2{srcNumColumns};
  for (U i=0; i<srcNumRows; i++)
  {
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], counter*sizeof(T));
    srcOffset += counter2;
    destOffset += counter;
    counter++;
  }

  if (assembleFinder)
  {
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    dest.setNumRowsLocal(srcNumRows);
    dest.setNumColumnsLocal(srcNumColumns);
    dest.setNumElems(numElems);
    MatrixStructureLowerTriangular<T,U,Distributer>::AssembleMatrix(destVectorData, destMatrixData, srcNumColumns, srcNumRows);
  }
  return;
}

/*
template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureLowerTriangular>::Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src,
  Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& dest, bool fillZeros, bool dir)
{
  std::cout << "Not updated. Only MatrixStructureSquare is. Makes no sense to change the implementation of all of these Structures until we think that the Square is right.\n";
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


template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureLowerTriangular>::Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& big,
  Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& small, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir)
{
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
  assert(rangeX == rangeY);

  U bigNumRows = big.getNumRowsLocal();
  U bigNumColumns = big.getNumColumnsLocal();
  U smallNumRows = small.getNumRowsLocal();
  U smallNumColumns = small.getNumColumnsLocal();

  std::vector<T>& bigVectorData = big.getVectorData();
  std::vector<T*>& bigMatrixData = big.getMatrixData();
  std::vector<T>& smallVectorData = small.getVectorData();
  std::vector<T*>& smallMatrixData = small.getMatrixData();

  // We assume that if dir==true, the user passed in a destination matrix that is properly sized
  // If I find a use case where that is not true, then I can pass in big x and y dimensions, but I do not wan to do that.
  // In most cases, the lae matri will be known and we are simply trying to fill in the smaller into the existing bigger

  U numElems = (dir ? bigNumRows*bigNumRows : ((rangeX*(rangeX+1))>>1));
  U numRows = (dir ? bigNumRows : rangeX);
  bool assembleFinder = false;
  if ((dir ? bigVectorData.size() : smallVectorData.size()) < numElems)
  {
    assembleFinder = true;
    dir ? bigVectorData.resize(numElems) : smallVectorData.resize(numElems);
  }
  if ((dir ? bigMatrixData.size() : smallMatrixData.size()) < numRows)
  {
    assembleFinder = true;
    dir ? bigMatrixData.resize(numRows) : smallMatrixData.resize(numRows);
  }

  U counter{1};
  U srcOffset = (dir ? 0 : cutDimensionYstart*bigNumColumns+cutDimensionXstart);
  U destOffset = (dir ? cutDimensionYstart*bigNumColumns+cutDimensionXstart  : 0);
  auto& destVectorData = dir ? bigVectorData : smallVectorData;
  auto& srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeY; i++)
  {
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*counter);
    destOffset += (dir ? bigNumColumns : counter);
    srcOffset += (dir ? counter : bigNumColumns);
    counter++;
  }

  if (assembleFinder)
  {
    if (dir) {abort();}		// weird case that I want to check against
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    small.setNumRowsLocal(numRows);
    small.setNumColumnsLocal(rangeX);
    small.setNumElems(numElems);
    // I am only providing UT here, not square, because if square, it would have aborted
    MatrixStructureLowerTriangular<T,U,Distributer>::AssembleMatrix(destVectorData, smallMatrixData, rangeX, numRows);
  }
}

/*
template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureLowerTriangular>::Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src,
  Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros, bool dir)
{
  std::cout << "Not updated. Only MatrixStructureSquare is. Makes no sense to change the implementation of all of these Structures until we think that the Square is right.\n";
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


template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureSquare>::Serialize(Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& src,
  Matrix<T,U,MatrixStructureSquare,Distributer>& dest)
{
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
  if (static_cast<U>(destVectorData.size()) < numElems)
  {
    assembleFinder = true;
    destVectorData.resize(numElems);
  }
  if (static_cast<U>(destMatrixData.size()) < srcNumRows)
  {
    assembleFinder = true;
    destMatrixData.resize(srcNumRows);
  }

  U counter{srcNumColumns};
  U srcOffset{0};
  U destOffset{0};
  U zeroOffset{0};
  U counter2{srcNumColumns+1};
  for (U i=0; i<srcNumRows; i++)
  {
    U fillZeros = srcNumColumns-counter;
    fillZerosContig(&destVectorData[zeroOffset], fillZeros);
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], counter*sizeof(T));
    srcOffset += counter;
    destOffset += counter2;
    zeroOffset += srcNumColumns;
    counter--;
  }

  if (assembleFinder)
  {
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    dest.setNumRowsLocal(srcNumRows);
    dest.setNumColumnsLocal(srcNumColumns);
    dest.setNumElems(numElems);
    MatrixStructureUpperTriangular<T,U,Distributer>::AssembleMatrix(destVectorData, destMatrixData, srcNumColumns, srcNumRows);
  }
  return;
}

template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureSquare>::Serialize(Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& big,
  Matrix<T,U,MatrixStructureSquare,Distributer>& small, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir)
{
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
  assert(rangeX == rangeY);

  U bigNumRows = big.getNumRowsLocal();
  U bigNumColumns = big.getNumColumnsLocal();
  U smallNumRows = small.getNumRowsLocal();
  U smallNumColumns = small.getNumColumnsLocal();

  std::vector<T>& bigVectorData = big.getVectorData();
  std::vector<T*>& bigMatrixData = big.getMatrixData();
  std::vector<T>& smallVectorData = small.getVectorData();
  std::vector<T*>& smallMatrixData = small.getMatrixData();

  U numElems = (dir ? ((bigNumColumns*(bigNumColumns+1))>>1) : rangeX*rangeX);
  U numRows = (dir ? bigNumRows : rangeY);
  bool assembleFinder = false;
  if (static_cast<U>((dir ? bigVectorData.size() : smallVectorData.size())) < numElems)
  {
    assembleFinder = true;
    dir ? bigVectorData.resize(numElems) : smallVectorData.resize(numElems);
  }
  if (static_cast<U>((dir ? bigMatrixData.size() : smallVectorData.size())) < numRows)
  {
    assembleFinder = true;
    dir ? bigMatrixData.resize(numRows) : smallMatrixData.resize(numRows);
  }

  U bigMatrixCounter = bigNumColumns-cutDimensionYstart-1;
  U counter = (dir ? rangeX : bigMatrixCounter);
  U bigMatrixOffset = ((bigNumColumns*(bigNumColumns+1))>>1);
  U helper = bigNumColumns - cutDimensionYstart;
  bigMatrixOffset -= ((helper*(helper+1))>>1);		// Watch out for 64-bit rvalue implicit cast problems!
  bigMatrixOffset += (cutDimensionXstart-cutDimensionYstart);
  U srcOffset = (dir ? 0 : bigMatrixOffset);
  U destOffset = (dir ? bigMatrixOffset : 0);
  auto& destVectorData = dir ? bigVectorData : smallVectorData;
  auto& srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeY; i++)
  {
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*rangeX);
    destOffset += (dir ? bigMatrixCounter : rangeX);
    srcOffset += counter;
    counter = (dir ? counter : counter-1);
    bigMatrixCounter--;					// Just do this regardless of dir
  }

  if (assembleFinder)
  {
    if (dir) {abort();}		// weird case that I want to check against
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    small.setNumRowsLocal(numRows);
    small.setNumColumnsLocal(rangeX);
    small.setNumElems(numElems);
    // I am only providing Square here, not UT, because if UT, it would have aborted
    MatrixStructureSquare<T,U,Distributer>::AssembleMatrix(destVectorData, smallMatrixData, rangeX, numRows);
  }
}

/* No reason for this method. Just use UT to little square
template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureSquare>::Serialize(Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& src,
  Matrix<T,U,MatrixStructureSquare,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros, bool dir)
{
  std::cout << "Not updated. Only MatrixStructureSquare is. Makes no sense to change the implementation of all of these Structures until we think that the Square is right.\n";
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


template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureUpperTriangular>::Serialize(Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& src,
  Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& dest)
{
  U srcNumRows = src.getNumRowsLocal();
  U srcNumColumns = src.getNumColumnsLocal();
  U srcNumElems = srcNumRows*srcNumColumns;

  std::vector<T>& srcVectorData = src.getVectorData();
  std::vector<T*>& srcMatrixData = src.getMatrixData();
  std::vector<T>& destVectorData = dest.getVectorData();
  std::vector<T*>& destMatrixData = dest.getMatrixData();

  bool assembleFinder = false;
  if (static_cast<U>(destVectorData.size()) < srcNumElems)
  {
    assembleFinder = true;
    destVectorData.resize(srcNumElems);
  }
  if (static_cast<U>(destMatrixData.size()) < srcNumRows)
  {
    assembleFinder = true;
    destMatrixData.resize(srcNumRows);
  }

  // direction doesn't matter here since no indexing here
  memcpy(&destVectorData[0], &srcVectorData[0], sizeof(T)*srcNumElems);

  if (assembleFinder)
  {
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    // User can assume that everything except for global dimensions are set. If he needs global dimensions too, he can set them himself.
    dest.setNumRowsLocal(srcNumRows);
    dest.setNumColumnsLocal(srcNumColumns);
    dest.setNumElems(srcNumElems);
    MatrixStructureUpperTriangular<T,U,Distributer>::AssembleMatrix(destVectorData, destMatrixData, srcNumColumns, srcNumRows);
  }
  return;
}



template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureUpperTriangular>::Serialize(Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& big,
  Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& small, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir)
{
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
  assert(rangeX == rangeY);

  U bigNumRows = big.getNumRowsLocal();
  U bigNumColumns = big.getNumColumnsLocal();
  U smallNumRows = small.getNumRowsLocal();
  U smallNumColumns = small.getNumColumnsLocal();

  std::vector<T>& bigVectorData = big.getVectorData();
  std::vector<T*>& bigMatrixData = big.getMatrixData();
  std::vector<T>& smallVectorData = small.getVectorData();
  std::vector<T*>& smallMatrixData = small.getMatrixData();

  U numElems = (dir ? ((bigNumColumns*(bigNumColumns+1))>>1) : ((rangeX*(rangeX+1))>>1));
  U numRows = (dir ? bigNumRows : rangeY);
  bool assembleFinder = false;
  if (static_cast<U>((dir ? bigVectorData.size() : smallVectorData.size())) < numElems)
  {
    assembleFinder = true;
    dir ? bigVectorData.resize(numElems) : smallVectorData.resize(numElems);
  }
  if (static_cast<U>((dir ? bigMatrixData.size() : smallMatrixData.size())) < numRows)
  {
    assembleFinder = true;
    dir ? bigMatrixData.resize(numRows) : smallMatrixData.resize(numRows);
  }

  U bigMatrixCounter = bigNumColumns -cutDimensionYstart;		// should be right
  U smallMatrixCounter{rangeX};
  U smallMatrixOffset = 0;
  U bigMatrixOffset = ((bigNumColumns*(bigNumColumns+1))>>1);
  U helper = bigNumColumns  - cutDimensionYstart;
  bigMatrixOffset -= ((helper*(helper+1))>>1);		// Watch out for 64-bit rvalue implicit cast problems!
  bigMatrixOffset += (cutDimensionXstart-cutDimensionYstart);
  U srcOffset = (dir ? smallMatrixOffset : bigMatrixOffset);
  U destOffset = (dir ? bigMatrixOffset : smallMatrixOffset);
  auto& destVectorData = dir ? bigVectorData : smallVectorData;
  auto& srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeY; i++)
  {
    std::cout << "check  - " << destOffset << std::endl;
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*smallMatrixCounter);
    destOffset += (dir ? bigMatrixCounter : smallMatrixCounter);
    srcOffset += (dir ? smallMatrixCounter : bigMatrixCounter);
    bigMatrixCounter--;
    smallMatrixCounter--;
  }

  if (assembleFinder)
  {
    if (dir) {abort();}
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    // User can assume that everything except for global dimensions are set. If he needs global dimensions too, he can set them himself.
    small.setNumRowsLocal(numRows);
    small.setNumColumnsLocal(rangeX);	// no dir needed here due to abort above
    small.setNumElems(numElems);
    MatrixStructureUpperTriangular<T,U,Distributer>::AssembleMatrix(destVectorData, smallMatrixData, rangeX, numRows);	// again, no dir ? needed here
  }
  return;
}


template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureSquare>::Serialize(Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& src,
  Matrix<T,U,MatrixStructureSquare,Distributer>& dest)
{
  std::cout << "I am in LT to Square\n";
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
  if (static_cast<U>(destVectorData.size()) < numElems)
  {
    assembleFinder = true;
    destVectorData.resize(numElems);
  }
  if (static_cast<U>(destMatrixData.size()) < srcNumRows)
  {
    assembleFinder = true;
    destMatrixData.resize(srcNumRows);
  }

  U counter{1};
  U srcOffset{0};
  U destOffset{0};
  U zeroOffset{1};
  U counter2{srcNumColumns};
  for (U i=0; i<srcNumRows; i++)
  {
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], counter*sizeof(T));
    U zeroIter = srcNumColumns-counter;
    for (U j=0; j<zeroIter; j++)
    {
      destVectorData[zeroOffset+j] = 0;
    }
    srcOffset += counter;
    destOffset += counter2;
    zeroOffset += (srcNumColumns+1);
    counter++;
  }

  if (assembleFinder)
  {
    std::cout << "I am in AssembleFinder\n";
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    // User can assume that everything except for global dimensions are set. If he needs global dimensions too, he can set them himself.
    dest.setNumRowsLocal(srcNumRows);
    dest.setNumColumnsLocal(srcNumColumns);	// no dir needed here due to abort above
    dest.setNumElems(numElems);
    MatrixStructureSquare<T,U,Distributer>::AssembleMatrix(destVectorData, destMatrixData, srcNumColumns, srcNumRows);	// again, no dir ? needed here
  }
  return;
}



template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureSquare>::Serialize(Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& big,
  Matrix<T,U,MatrixStructureSquare,Distributer>& small, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir)
{
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
  assert(rangeX == rangeY);

  U bigNumRows = big.getNumRowsLocal();
  U bigNumColumns = big.getNumColumnsLocal();
  U smallNumRows = small.getNumRowsLocal();
  U smallNumColumns = small.getNumColumnsLocal();

  std::vector<T>& bigVectorData = big.getVectorData();
  std::vector<T*>& bigMatrixData = big.getMatrixData();
  std::vector<T>& smallVectorData = small.getVectorData();
  std::vector<T*>& smallMatrixData = small.getMatrixData();

  U numElems = (dir ? ((bigNumColumns*(bigNumColumns+1))>>1) : rangeX*rangeX);
  U numRows = (dir ? bigNumRows : rangeY);
  bool assembleFinder = false;
  if (static_cast<U>((dir ? bigVectorData.size() : smallVectorData.size())) < numElems)
  {
    assembleFinder = true;
    dir ? bigVectorData.resize(numElems) : smallVectorData.resize(numElems);
  }
  if (static_cast<U>((dir ? bigMatrixData.size() : smallMatrixData.size())) < numRows)
  {
    assembleFinder = true;
    dir ? bigMatrixData.resize(numRows) : smallMatrixData.resize(numRows);
  }

  U smallMatrixOffset = 0;
  U smallMatrixCounter = rangeX;
  U bigMatrixCounter = cutDimensionYstart;
  U bigMatrixOffset = ((bigMatrixCounter*(bigMatrixCounter+1))>>1);
  bigMatrixOffset += cutDimensionXstart;
  U srcOffset = (dir ? smallMatrixOffset : bigMatrixOffset);
  U destOffset = (dir ? bigMatrixOffset : 0);
  auto& destVectorData = dir ? bigVectorData : smallVectorData;
  auto& srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeY; i++)
  {
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*rangeX);
    destOffset += (dir ? bigMatrixCounter+1 : smallMatrixCounter);
    srcOffset += (dir ? smallMatrixCounter : bigMatrixCounter+1);
    bigMatrixCounter++;
  }
  if (assembleFinder)
  {
    if (dir) {abort();}
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    // User can assume that everything except for global dimensions are set. If he needs global dimensions too, he can set them himself.
    small.setNumRowsLocal(numRows);
    small.setNumColumnsLocal(rangeX);	// no dir needed here due to abort above
    small.setNumElems(numElems);
    MatrixStructureSquare<T,U,Distributer>::AssembleMatrix(destVectorData, smallMatrixData, rangeX, numRows);	// again, no dir ? needed here
  }
  return;
}


/*
template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureSquare>::Serialize(Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& src,
  Matrix<T,U,MatrixStructureSquare,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros, bool dir)
{
  std::cout << "Not updated. Only MatrixStructureSquare is. Makes no sense to change the implementation of all of these Structures until we think that the Square is right.\n";
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


template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureLowerTriangular>::Serialize(Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& src,
  Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& dest)
{
  U srcNumRows = src.getNumRowsLocal();
  U srcNumColumns = src.getNumColumnsLocal();
  U srcNumElems = srcNumRows*srcNumColumns;

  std::vector<T>& srcVectorData = src.getVectorData();
  std::vector<T*>& srcMatrixData = src.getMatrixData();
  std::vector<T>& destVectorData = dest.getVectorData();
  std::vector<T*>& destMatrixData = dest.getMatrixData();

  bool assembleFinder = false;
  if (static_cast<U>(destVectorData.size()) < srcNumElems)
  {
    assembleFinder = true;
    destVectorData.resize(srcNumElems);
  }
  if (static_cast<U>(destMatrixData.size()) < srcNumRows)
  {
    assembleFinder = true;
    destMatrixData.resize(srcNumRows);
  }

  // direction doesn't matter here since no indexing here
  memcpy(&destVectorData[0], &srcVectorData[0], sizeof(T)*srcNumElems);

  if (assembleFinder)
  {
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    // User can assume that everything except for global dimensions are set. If he needs global dimensions too, he can set them himself.
    dest.setNumRowsLocal(srcNumRows);
    dest.setNumColumnsLocal(srcNumColumns);
    dest.setNumElems(srcNumElems);
    MatrixStructureLowerTriangular<T,U,Distributer>::AssembleMatrix(destVectorData, destMatrixData, srcNumColumns, srcNumRows);
  }
  return;
}

template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureLowerTriangular>::Serialize(Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& big,
  Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& small, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir)
{
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
  assert(rangeX == rangeY);

  U bigNumRows = big.getNumRowsLocal();
  U bigNumColumns = big.getNumColumnsLocal();
  U smallNumRows = small.getNumRowsLocal();
  U smallNumColumns = small.getNumColumnsLocal();

  std::vector<T>& bigVectorData = big.getVectorData();
  std::vector<T*>& bigMatrixData = big.getMatrixData();
  std::vector<T>& smallVectorData = small.getVectorData();
  std::vector<T*>& smallMatrixData = small.getMatrixData();

  U numElems = (dir ? ((bigNumColumns*(bigNumColumns+1))>>1) : ((rangeX*(rangeX+1))>>1));
  U numRows = (dir ? bigNumRows : rangeY);
  bool assembleFinder = false;
  if (static_cast<U>((dir ? bigVectorData.size() : smallVectorData.size())) < numElems)
  {
    assembleFinder = true;
    dir ? bigVectorData.resize(numElems) : smallVectorData.resize(numElems);
  }
  if (static_cast<U>((dir ? bigMatrixData.size() : smallMatrixData.size())) < numRows)
  {
    assembleFinder = true;
    dir ? bigMatrixData.resize(numRows) : smallMatrixData.resize(numRows);
  }

  U smallMatrixCounter = 1;
  U bigMatrixCounter = cutDimensionYstart;
  U smallMatrixOffset = 0;
  U bigMatrixOffset = ((bigMatrixCounter*(bigMatrixCounter+1))>>1);
  bigMatrixOffset += cutDimensionXstart;
  U srcOffset = (dir ? smallMatrixOffset : bigMatrixOffset);
  U destOffset = (dir ? bigMatrixOffset : smallMatrixOffset);
  auto& destVectorData = dir ? bigVectorData : smallVectorData;
  auto& srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeY; i++)
  {
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*smallMatrixCounter);
    destOffset += (dir ? bigMatrixCounter+1 : smallMatrixCounter);
    srcOffset += (dir ? smallMatrixCounter : bigMatrixCounter+1);
    bigMatrixCounter++;
    smallMatrixCounter++;
  }

  if (assembleFinder)
  {
    if (dir) {abort();}
    // We won't always have to reassemble the offset vector. Only necessary when the destination matrix was being assembled in here.
    // User can assume that everything except for global dimensions are set. If he needs global dimensions too, he can set them himself.
    small.setNumRowsLocal(numRows);
    small.setNumColumnsLocal(rangeX);	// no dir needed here due to abort above
    small.setNumElems(numElems);
    MatrixStructureLowerTriangular<T,U,Distributer>::AssembleMatrix(destVectorData, smallMatrixData, rangeX, numRows);	// again, no dir ? needed here
  }
  return;
}
