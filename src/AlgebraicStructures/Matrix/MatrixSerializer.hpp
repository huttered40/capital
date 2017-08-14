/* Author: Edward Hutter */

template<typename T, typename U>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureUpperTriangular>::Serialize(const std::vector<T*>& src, std::vector<T*>& dest, U dimensionX, U dimensionY)
{
  U counter{dimensionY};
  U srcOffset{0};
  U destOffset{0};
  U counter2{dimensionY+1};
  for (U i=0; i<dimensionX; i++)
  {
    memcpy(&dest[0][destOffset], &src[0][srcOffset], counter*sizeof(T));
    srcOffset += counter2;
    destOffset += counter;
    counter--;
  }
  return;
}
