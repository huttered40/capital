/* Author: Edward Hutter */

template<typename T, typename U>
void Serializer<T,U,MatrixStructureSquare, MatrixStructureUpperTriangular>::Serialize(const std::vector<T*>& src, std::vector<T*>& dest, U dimensionX, U dimensionY)
{
  std::cout << "I am serializing a Square into a Upper Triangular\n";
  return;
}
