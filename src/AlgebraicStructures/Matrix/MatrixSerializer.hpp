/* Author: Edward Hutter */

template<typename T, typename U>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureSquare>::Serialize(T* src, T*& dest, U dimensionX, U dimensionY)
{
  // Do nothing but must be defined
  if (dest == nullptr)
  {
    dest = src;
  }
  return;
}

template<typename T, typename U>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureUpperTriangular>::Serialize(const T* src, T*& dest, U dimensionX, U dimensionY)
{
  if (dest == nullptr)
  {
    U numElems = ((dimensionX*(dimensionX+1))>>1);
    dest = new T[numElems];
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
void Serializer<T,U,MatrixStructureSquare, MatrixStructureLowerTriangular>::Serialize(const T* src, T*& dest, U dimensionX, U dimensionY)
{
  if (dest == nullptr)
  {
    U numElems = ((dimensionX*(dimensionX+1))>>1);
    dest = new T[numElems];
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
void Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureSquare>::Serialize(const T* src, T*& dest, U dimensionX, U dimensionY)
{
  if (dest == nullptr)
  {
    U numElems = dimensionX*dimensionX;
    dest = new T[numElems];
  }

  U counter{dimensionX};
  U srcOffset{0};
  U destOffset{0};
  U zeroOffset{0};
  U counter2{dimensionX+1};
  for (U i=0; i<dimensionY; i++)
  {
    U zeroIter = dimensionX-counter;
    for (U j=0; j<zeroIter; j++)
    {
      dest[zeroOffset+j] = 0;
    }
    memcpy(&dest[destOffset], &src[srcOffset], counter*sizeof(T));
    srcOffset += counter;
    destOffset += counter2;
    zeroOffset += dimensionX;
    counter--;
  }
  return;
}

template<typename T, typename U>
void Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureUpperTriangular>::Serialize(T* src, T*& dest, U dimensionX, U dimensionY)
{
  // Do nothing but must be defined
  if (dest == nullptr)
  {
    dest = src;
  }
  return;
}

template<typename T, typename U>
void Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureSquare>::Serialize(const T* src, T*& dest, U dimensionX, U dimensionY)
{
  if (dest == nullptr)
  {
    U numElems = dimensionX*dimensionX;
    dest = new T[numElems];
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
void Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureLowerTriangular>::Serialize(T* src, T*& dest, U dimensionX, U dimensionY)
{
  // Do nothing but must be defined
  if (dest == nullptr)
  {
    dest = src;
  }
  return;
}
