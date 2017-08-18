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
  // Need to write a base case
  // Need to decide whether this routine should be a wrapper for the recursive routine (I think it should)
  // Need to write the base case
  // Need to write the recursive function with the calls to Summa3D_cut
}
