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
  Matrix<T,U,MatrixStructureSquare,Distributer>& dest, bool dir)
{
  U srcNumRows = src.getNumRowsLocal();
  U srcNumColumns = src.getNumColumnsLocal();
  U srcNumElems = srcNumRows*srcNumColumns;

  std::vector<T>& srcVectorData = src.getVectorData();
  std::vector<T*>& srcMatrixData = src.getMatrixData();
  std::vector<T>& destVectorData = dest.getVectorData();
  std::vector<T*>& destMatrixData = dest.getMatrixData();

  if (destVectorData.size() < srcNumElems)
  {
    destVectorData.resize(srcNumElems);
  }
  if (destMatrixData.size() < srcNumRows)
  {
    destMatrixData.resize(srcNumRows);
  }

  dest.setNumRowsLocal(srcNumRows);
  dest.setNumColumnsLocal(srcNumColumns);
  dest.setNumRowsGlobal(src.getNumRowsGlobal());
  dest.setNumColumnsGlobal(src.getNumColumnsGlobal());
  dest.setNumElems(srcNumElems);

  // direction doesn't matter here since no indexing here
  memcpy(&destVectorData[0], &srcVectorData[0], sizeof(T)*srcNumElems);
  MatrixStructureSquare<T,U,Distributer>::AssembleMatrix(destVectorData, destMatrixData, srcNumColumns, srcNumElems); 
  return;
}
  
template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureSquare>::Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src,
  Matrix<T,U,MatrixStructureSquare,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir)
{
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
  assert(rangeX == rangeY);

  U srcNumRows = src.getNumRowsLocal();
  U srcNumColumns = src.getNumColumnsLocal();
  U destNumRows = dest.getNumRowsLocal();
  U destNumColumns = dest.getNumColumnsLocal();

  std::vector<T>& srcVectorData = src.getVectorData();
  std::vector<T*>& srcMatrixData = src.getMatrixData();
  std::vector<T>& destVectorData = dest.getVectorData();
  std::vector<T*>& destMatrixData = dest.getMatrixData();

  U numElems = (dir ? destNumRows*destNumColumns : rangeX*rangeY);
  U numRows = (dir ? destNumRows : rangeY);
  if (destVectorData.size() < numElems)
  {
    destVectorData.resize(numElems);
  }
  if (destMatrixData.size() < numRows)
  {
    destMatrixData.resize(numRows);
  }

  U destIndex = (dir ? cutDimensionYstart*destNumColumns+cutDimensionXstart : 0);
  U srcIndex = (dir ? 0 : cutDimensionYstart*srcNumColumns+cutDimensionXstart);
  U srcCounter = (dir ? rangeX : srcNumColumns);
  U destCounter = (dir ? destNumColumns : rangeX);
  for (U i=0; i<rangeY; i++)					// rangeY is fine.
  {
    memcpy(&destVectorData[destIndex], &srcVectorData[srcIndex], sizeof(T)*rangeX);		// rangeX is fine. blocks of size rangeX are still being copied.
    destIndex += destCounter;
    srcIndex += srcCounter;
  }

  MatrixStructureSquare<T,U,Distributer>::AssembleMatrix(destVectorData, destMatrixData, (dir ? destNumColumns : rangeX), numRows);
}

/*
template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureUpperTriangular>::Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src,
  Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& dest, bool dir)
{
  std::cout << "Not updated. Only MatrixStructureSquare is. Makes no sense to change the implementation of all of these Structures until we think that the Square is right.\n";
  abort();

  U numElems = (dir ? dimensionX*dimensionX : ((dimensionX*(dimensionX+1))>>1));
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
    memcpy(&dest[destOffset], &src[srcOffset], counter*sizeof(T));
    srcOffset += (dir ? counter : counter2);
    destOffset += (dir ? counter2 : counter);
    counter--;
  }
  return;
}

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
    fillZerosContig(&dest[destOffset], fillSize);
    destOffset += fillSize;
    memcpy(&dest[destOffset], &src[srcOffset], counter*sizeof(T));
    srcOffset += counter2;
    destOffset += counter;
    counter--;
  }
  return;
}

template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureUpperTriangular>::Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src,
  Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir)
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

  U numElems = ((rangeX*(rangeX+1))>>1);
  if (dest.size() < numElems)
  {
    dest.resize(numElems);
  }

  U destIndex = 0;
  U counter{rangeX};
  U srcIndexSave = cutDimensionYstart*dimensionX+cutDimensionXstart;
  for (U i=0; i<rangeY; i++)
  {
    memcpy(&dest[destIndex], &src[srcIndexSave], sizeof(T)*counter);
    destIndex += counter;
    srcIndexSave += (dimensionX+1);
    counter--;
  }
}

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
    fillZerosContig(&dest[destIndex], fillSize);
    destIndex += fillSize;
    memcpy(&dest[destIndex], &src[srcIndexSave], sizeof(T)*counter);
    destIndex += counter;
    srcIndexSave += dimensionX;
    counter--;
  }
}


template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureLowerTriangular>::Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src,
  Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& dest, bool dir)
{
  std::cout << "Not updated. Only MatrixStructureSquare is. Makes no sense to change the implementation of all of these Structures until we think that the Square is right.\n";
  abort();
  if (dir == true)
  {
    std::cout << "Not finished yet. Complete when necessary\n";
    abort();
  }

  U numElems = ((dimensionX*(dimensionX+1))>>1);
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
    memcpy(&dest[destOffset], &src[srcOffset], counter*sizeof(T));
    srcOffset += counter2;
    destOffset += counter;
    counter++;
  }
  return;
}

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
    memcpy(&dest[destOffset], &src[srcOffset], counter*sizeof(T));
    srcOffset += counter2;
    destOffset += counter;
    U fillSize = dimensionX - counter;
    fillZerosContig(&dest[destOffset], fillSize);
    destOffset += fillSize;
    counter++;
  }
  return;
}

template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureLowerTriangular>::Serialize(Matrix<T,U,MatrixStructureSquare,Distributer>& src,
  Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir)
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

  U numElems = ((rangeX*(rangeX+1))>>1);
  if (dest.size() < numElems)
  {
    dest.resize(numElems);
  }

  U counter{1};
  U srcOffset{cutDimensionYstart*dimensionX+cutDimensionXstart};
  U destOffset{0};
  for (U i=0; i<rangeY; i++)
  {
    memcpy(&dest[destOffset], &src[srcOffset], sizeof(T)*counter);
    destOffset += counter;
    srcOffset += dimensionX;
    counter++;
  }
}

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
    memcpy(&dest[destOffset], &src[srcOffset], sizeof(T)*counter);
    destOffset += counter;
    srcOffset += dimensionX;
    U fillSize = rangeX - counter;
    fillZerosContig(&dest[destOffset], fillSize);
    destOffset += fillSize;
    counter++;
  }
}


template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureSquare>::Serialize(Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& src,
  Matrix<T,U,MatrixStructureSquare,Distributer>& dest, U dimensionX, U dimensionY, bool dir)
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
  U zeroOffset{0};
  U counter2{dimensionX+1};
  for (U i=0; i<dimensionY; i++)
  {
    U fillZeros = dimensionX-counter;
    fillZerosContig(&dest[zeroOffset], fillZeros);
    memcpy(&dest[destOffset], &src[srcOffset], counter*sizeof(T));
    srcOffset += counter;
    destOffset += counter2;
    zeroOffset += dimensionX;
    counter--;
  }
  return;
}

template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureSquare>::Serialize(Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& src,
  Matrix<T,U,MatrixStructureSquare,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir)
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
  U srcOffset = ((dimensionX*(dimensionX+1))>>1);
  U helper = dimensionX - cutDimensionYstart;
  srcOffset -= ((helper*(helper+1))>>1);		// Watch out for 64-bit rvalue implicit cast problems!
  srcOffset += (cutDimensionXstart-cutDimensionYstart);
  U destOffset{0};
  for (U i=0; i<rangeY; i++)
  {
    memcpy(&dest[destOffset], &src[srcOffset], sizeof(T)*rangeX);
    destOffset += rangeX;
    srcOffset += counter;
    counter--;
  }
}

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
    fillZerosContig(&dest[destOffset], fillZeros);
    destOffset += fillZeros;
    memcpy(&dest[destOffset], &src[srcOffset], sizeof(T)*counter2);
    destOffset += counter2;
    srcOffset += (counter+1);
    counter--;
    counter2--;
  }
}


template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureUpperTriangular>::Serialize(Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& src,
  Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& dest, bool dir)
{
  std::cout << "Not updated. Only MatrixStructureSquare is. Makes no sense to change the implementation of all of these Structures until we think that the Square is right.\n";
  abort();
  if (dir == true)
  {
    std::cout << "Not finished yet. Complete when necessary\n";
    abort();
  }

  U numElems = ((dimensionX*(dimensionX+1))>>1);
  if (dest.size() < numElems)
  {
    dest.resize(numElems);
  }

  memcpy(&dest[0], &src[0], sizeof(T)*numElems);
  return;
}

template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureUpperTriangular>::Serialize(Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& src,
  Matrix<T,U,MatrixStructureUpperTriangular,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir)
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

  U numElems = ((rangeX*(rangeX+1))>>1);
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
    memcpy(&dest[destOffset], &src[srcOffset], sizeof(T)*counter2);
    destOffset += counter2;
    srcOffset += (counter+1);
    counter--;
    counter2--;
  }
}


template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureSquare>::Serialize(Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& src,
  Matrix<T,U,MatrixStructureSquare,Distributer>& dest, bool dir)
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
  U zeroOffset{1};
  U counter2{dimensionX};
  for (U i=0; i<dimensionY; i++)
  {
    memcpy(&dest[destOffset], &src[srcOffset], counter*sizeof(T));
    U zeroIter = dimensionX-counter;
    for (U j=0; j<zeroIter; j++)
    {
      dest[zeroOffset+j] = 0;
    }
    srcOffset += counter;
    destOffset += counter2;
    zeroOffset += (dimensionX+1);
    counter++;
  }
  return;
}

template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureSquare>::Serialize(Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& src,
  Matrix<T,U,MatrixStructureSquare,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir)
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
  U srcOffset = ((counter*(counter+1))>>1);
  srcOffset += cutDimensionXstart;
  U destOffset{0};
  for (U i=0; i<rangeY; i++)
  {
    memcpy(&dest[destOffset], &src[srcOffset], sizeof(T)*rangeX);
    destOffset += rangeX;
    srcOffset += (counter+1);
    counter++;
  }
}

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
    memcpy(&dest[destOffset], &src[srcOffset], sizeof(T)*counter2);
    destOffset += counter2;
    U fillSize = rangeX - counter2;
    fillZerosContig(&dest[destOffset], fillSize);
    destOffset += fillSize;
    srcOffset += (counter+1);
    counter++;
    counter2++;
  }
}

template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureLowerTriangular>::Serialize(Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& src,
  Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& dest, bool dir)
{
  std::cout << "Not updated. Only MatrixStructureSquare is. Makes no sense to change the implementation of all of these Structures until we think that the Square is right.\n";
  abort();
  if (dir == true)
  {
    std::cout << "Not finished yet. Complete when necessary\n";
    abort();
  }

  U numElems = ((dimensionX*(dimensionX+1))>>1);
  if (dest.size() < numElems)
  {
    dest.resize(numElems);
  } 

  return;
}

template<typename T, typename U>
template<template<typename, typename,int> class Distributer>
void Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureLowerTriangular>::Serialize(Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& src,
  Matrix<T,U,MatrixStructureLowerTriangular,Distributer>& dest, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool dir)
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

  U numElems = ((rangeX*(rangeX+1))>>1);
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
    memcpy(&dest[destOffset], &src[srcOffset], sizeof(T)*counter2);
    destOffset += counter2;
    srcOffset += (counter+1);
    counter++;
    counter2++;
  }
}
*/
