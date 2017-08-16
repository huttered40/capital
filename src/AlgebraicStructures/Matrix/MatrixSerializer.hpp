/* Author: Edward Hutter */

template<typename T, typename U>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureUpperTriangular>::Serialize(const std::vector<T*>& src, std::vector<T*>& dest, U dimensionX, U dimensionY)
{
  U counter{dimensionX};
  U srcOffset{0};
  U destOffset{0};
  U counter2{dimensionX+1};
  for (U i=0; i<dimensionY; i++)
  {
    memcpy(&dest[0][destOffset], &src[0][srcOffset], counter*sizeof(T));
    srcOffset += counter2;
    destOffset += counter;
    counter--;
  }
  return;
}

template<typename T, typename U>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureLowerTriangular>::Serialize(const std::vector<T*>& src, std::vector<T*>& dest, U dimensionX, U dimensionY)
{
  U counter{1};
  U srcOffset{0};
  U destOffset{0};
  U counter2{dimensionX};
  for (U i=0; i<dimensionY; i++)
  {
    memcpy(&dest[0][destOffset], &src[0][srcOffset], counter*sizeof(T));
    srcOffset += counter2;
    destOffset += counter;
    counter++;
  }
  return;
}

template<typename T, typename U>
void Serializer<T,U,MatrixStructureUpperTriangular, MatrixStructureSquare>::Serialize(const std::vector<T*>& src, std::vector<T*>& dest, U dimensionX, U dimensionY)
{
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
      dest[0][zeroOffset+j] = 0;
    }
    memcpy(&dest[0][destOffset], &src[0][srcOffset], counter*sizeof(T));
    srcOffset += counter;
    destOffset += counter2;
    zeroOffset += dimensionX;
    counter--;
  }
  return;
}

template<typename T, typename U>
void Serializer<T,U,MatrixStructureLowerTriangular, MatrixStructureSquare>::Serialize(const std::vector<T*>& src, std::vector<T*>& dest, U dimensionX, U dimensionY)
{
  U counter{1};
  U srcOffset{0};
  U destOffset{0};
  U zeroOffset{1};
  U counter2{dimensionX};
  for (U i=0; i<dimensionY; i++)
  {
    memcpy(&dest[0][destOffset], &src[0][srcOffset], counter*sizeof(T));
    U zeroIter = dimensionX-counter;
    for (U j=0; j<zeroIter; j++)
    {
      dest[0][zeroOffset+j] = 0;
    }
    srcOffset += counter;
    destOffset += counter2;
    zeroOffset += (dimensionX+1);
    counter++;
  }
  return;
}
