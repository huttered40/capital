/* Author: Edward Hutter */


template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
void CFR3D<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular>::Factor(
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
  Matrix<T,U,MatrixStructureLowerTriangular,Distribution>& matrixL,
  Matrix<T,U,MatrixStructureLowerTriangular,Distribution>& matrixLI,
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
void CFR3D<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular>::rFactor(
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
  Matrix<T,U,MatrixStructureLowerTriangular,Distribution>& matrixL,
  Matrix<T,U,MatrixStructureLowerTriangular,Distribution>& matrixLI,
  U dimension,
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
    std::cout << "Base case has been reached!\n";
    return;
  }

  std::cout << "In recursive function\n";
}
