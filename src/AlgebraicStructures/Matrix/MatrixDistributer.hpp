/* Author: Edward Hutter */

// Try this to get it to compile: provide a declaration of a templated constructor,
//   and then define/implement a specialized method.

template<typename T, typename U>
void MatrixDistributerCyclic<T,U>::Distribute
                (
		   std::vector<T*>& matrix,
		   U dimensionX,
		   U dimensionY,
		   U localPgridX,
		   U localPgridY,
		   U globalPgridX,
		   U globalPgridY
		 )
{
  std::cout << "I am in the Distribute method!!\n";
  std::cout << "Note that the innefficiency here is that I am allocating the memory for matrix twice and constructing twice, incuring two loops over the data, when I should be allocating once and constructing once, with only a single loop to construct/distribute. This will necessitate usage of MatrixAllocator, which I will experiment with later, and enforcing that before we use this matrix, the elements have been constructed and not just allocated." << std::endl;
  for (int i=0; i<dimensionX; i++)
  {
    for (int j=0; j<dimensionY; j++)
    {
      srand(i*dimensionY+j);
      matrix[i][j] = drand48();			// Change this later.
    }
  }
  return;
}
