/* Author: Edward Hutter */

// Try this to get it to compile: provide a declaration of a templated constructor,
//   and then define/implement a specialized method.

template<typename T, typename U>
void MatrixDistributerCyclic<T,U,std::vector<T*>>::Distribute
		(
		  std::vector<T*>& matrix,
		  U dimensionX,
		  U dimensionY,
		  U globalDimensionX,
		  U globalDimensionY
		)
{
  std::cout << "I am in the Distribute method!!\n";
}
