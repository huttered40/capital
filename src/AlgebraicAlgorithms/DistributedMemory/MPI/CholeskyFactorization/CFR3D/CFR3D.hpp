/* Author: Edward Hutter */


template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
void CFR3D<T,U,MatrixStructureSquare,MatrixStructureSquare>::Factor(
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixL,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixLI,
  U dimension,
  MPI_Comm commWorld )
{
  // Need to split up the commWorld communicator into a 3D grid similar to Summa3D
  int rank,size;
  MPI_Comm_rank(commWorld, &rank);
  MPI_Comm_size(commWorld, &size);

  int pGridDimensionSize = ceil(pow(size,1./3.));
  int helper = pGridDimensionSize;
  helper *= helper;
  U bcDimension = dimension/helper;
  U globalDimension = dimension*pGridDimensionSize;

  rFactor(matrixA, matrixL, matrixLI, dimension, bcDimension, globalDimension,
    0, dimension, 0, dimension, 0, dimension, 0, dimension, 0, dimension, 0, dimension, commWorld);
}

template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
void CFR3D<T,U,MatrixStructureSquare,MatrixStructureSquare>::rFactor(
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixL,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixLI,
  U localDimension,
  U bcDimension,
  U globalDimension,
  U matAstartX,
  U matAendX,
  U matAstartY,
  U matAendY,
  U matLstartX,
  U matLendX,
  U matLstartY,
  U matLendY,
  U matLIstartX,
  U matLIendX,
  U matLIstartY,
  U matLIendY,
  MPI_Comm commWorld )
{
  if (globalDimension == bcDimension)
  {
    std::cout << "Base case has been reached with " << globalDimension << " " << localDimension << " " << bcDimension << "\n";
    return;
  }

  U shift = (localDimension>>1);
  rFactor(matrixA, matrixL, matrixLI, shift, bcDimension, (globalDimension>>1),
    matAstartX, matAstartX+shift, matAstartY, matAstartY+shift,
    matLstartX, matLstartX+shift, matLstartY, matLstartY+shift,
    matLIstartX, matLIstartX+shift, matLIstartY, matLIstartY+shift, commWorld);

  // Perform transpose via some static method since it is something we will use alot

  // Perform first matrix multiplication
  // Note that we aim to "fill up" the top-left part of L and L^{-1} when this returns. 
  //Summa3D<...>::Multiply(...);

}
