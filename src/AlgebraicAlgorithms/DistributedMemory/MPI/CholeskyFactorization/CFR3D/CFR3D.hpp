/* Author: Edward Hutter */


template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
void CFR3D<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular>::Factor(
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
  Matrix<T,U,MatrixStructureLowerTriangular,Distribution>& matrixL,
  U dimension,
  MPI_Comm commWorld )
{
  std::cout << "Ed Hutter\n";
}
