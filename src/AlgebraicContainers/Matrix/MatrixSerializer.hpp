/* Author: Edward Hutter */

/*
  Note: info 1 -> no memory was allocated for dest
        info 2 -> memory was allocated for dest
        info 3 -> memory was not allocated, but dest was changed to point to src
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
void Serializer<T,U,MatrixStructureSquare, MatrixStructureSquare>::Serialize(std::vector<T>& src, std::vector<T>& dest, U dimensionX, U dimensionY)
{
  assert(dimensionX == dimensionY);

  // This extra check could be expensive.
  U numElems = dimensionX*dimensionY;
  if (dest.size() < numElems)
  {
    dest.resize(numElems);
  }
  memcpy(&dest[0], &src[0], sizeof(T)*numElems);
  return;
}
  
template<typename T, typename U>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureSquare>::Serialize(std::vector<T>& src, std::vector<T>& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend)
{
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  assert(rangeX == rangeY);
  assert(dimensionX == dimensionY);

  U numElems = rangeX*rangeY;
  if (dest.size() < numElems)
  {
    dest.resize(numElems);
  }

  U destIndex = 0;
  U srcIndexSave = cutDimensionYstart*dimensionX+cutDimensionXstart;
  for (U i=0; i<rangeY; i++)
  {
    memcpy(&dest[destIndex], &src[srcIndexSave], sizeof(T)*rangeX);
    destIndex += rangeX;
    srcIndexSave += dimensionX;
  }
}

template<typename T, typename U>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureUpperTriangular>::Serialize(std::vector<T>& src, std::vector<T>& dest, U dimensionX, U dimensionY)
{
  assert(dimensionX == dimensionY);

  U numElems = ((dimensionX*(dimensionX+1))>>1);
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
    srcOffset += counter2;
    destOffset += counter;
    counter--;
  }
  return;
}

template<typename T, typename U>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureUpperTriangular>::Serialize(std::vector<T>& src, std::vector<T>& dest, U dimensionX, U dimensionY, bool fillZeros)
{
  assert(dimensionX == dimensionY);

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
void Serializer<T,U,MatrixStructureSquare, MatrixStructureUpperTriangular>::Serialize(std::vector<T>& src, std::vector<T>& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend)
{
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  assert(rangeX == rangeY);
  assert(dimensionX == dimensionY);

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
void Serializer<T,U,MatrixStructureSquare, MatrixStructureUpperTriangular>::Serialize(std::vector<T>& src, std::vector<T>& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros)
{
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  assert(rangeX == rangeY);
  assert(dimensionX == dimensionY);

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
void Serializer<T,U,MatrixStructureSquare, MatrixStructureLowerTriangular>::Serialize(std::vector<T>& src, std::vector<T>& dest, U dimensionX, U dimensionY)
{
  assert(dimensionX == dimensionY);

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
void Serializer<T,U,MatrixStructureSquare, MatrixStructureLowerTriangular>::Serialize(std::vector<T>& src, std::vector<T>& dest, U dimensionX, U dimensionY, bool fillZeros)
{
  assert(dimensionX == dimensionY);

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
void Serializer<T,U,MatrixStructureSquare, MatrixStructureLowerTriangular>::Serialize(std::vector<T>& src, std::vector<T>& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend)
{
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  assert(rangeX == rangeY);
  assert(dimensionX == dimensionY);

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
void Serializer<T,U,MatrixStructureSquare, MatrixStructureLowerTriangular>::Serialize(std::vector<T>& src, std::vector<T>& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros)
{
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  assert(rangeX == rangeY);
  assert(dimensionX == dimensionY);

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
void Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureSquare>::Serialize(std::vector<T>& src, std::vector<T>& dest, U dimensionX, U dimensionY)
{
  assert(dimensionX == dimensionY);

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
void Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureSquare>::Serialize(std::vector<T>& src, std::vector<T>& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend)
{
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  assert(rangeX == rangeY);
  assert(dimensionX == dimensionY);

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
void Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureSquare>::Serialize(std::vector<T>& src, std::vector<T>& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros)
{
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  assert(rangeX == rangeY);
  assert(dimensionX == dimensionY);

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
void Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureUpperTriangular>::Serialize(std::vector<T>& src, std::vector<T>& dest, U dimensionX, U dimensionY)
{
  assert(dimensionX == dimensionY);

  U numElems = ((dimensionX*(dimensionX+1))>>1);
  if (dest.size() < numElems)
  {
    dest.resize(numElems);
  }

  memcpy(&dest[0], &src[0], sizeof(T)*numElems);
  return;
}

template<typename T, typename U>
void Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureUpperTriangular>::Serialize(std::vector<T>& src, std::vector<T>& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend)
{
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  assert(rangeX == rangeY);
  assert(dimensionX == dimensionY);

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
void Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureSquare>::Serialize(std::vector<T>& src, std::vector<T>& dest, U dimensionX, U dimensionY)
{
  assert(dimensionX == dimensionY);

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
void Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureSquare>::Serialize(std::vector<T>& src, std::vector<T>& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend)
{
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  assert(rangeX == rangeY);
  assert(dimensionX == dimensionY);

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
void Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureSquare>::Serialize(std::vector<T>& src, std::vector<T>& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend, bool fillZeros)
{
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  assert(rangeX == rangeY);
  assert(dimensionX == dimensionY);

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
void Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureLowerTriangular>::Serialize(std::vector<T>& src, std::vector<T>& dest, U dimensionX, U dimensionY)
{
  assert(dimensionX == dimensionY);

  U numElems = ((dimensionX*(dimensionX+1))>>1);
  if (dest.size() < numElems)
  {
    dest.resize(numElems);
  } 

  return;
}

template<typename T, typename U>
void Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureLowerTriangular>::Serialize(std::vector<T>& src, std::vector<T>& dest, U dimensionX, U dimensionY, U cutDimensionXstart, U cutDimensionXend, U cutDimensionYstart, U cutDimensionYend)
{
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  assert(rangeX == rangeY);
  assert(dimensionX == dimensionY);

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
