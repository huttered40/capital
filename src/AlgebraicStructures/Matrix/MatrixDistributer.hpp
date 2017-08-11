/* Author: Edward Hutter */

template<typename T, typename U>
static void MatrixDistributerCyclic<T,U,std::vector<T*>>::Distribute
		(
		  std::vector<T*> matrix,
		  U dimensionX,
		  U dimensionY
		)
{
  std::cout << "I am in the Distribute method!!\n";
}
