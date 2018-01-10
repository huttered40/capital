/*
  Author: Edward Hutter
*/

template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void RTI3D<T,U,blasEngine>::Invert(
              Matrix<T,U,MatrixStructureSquare,Distribution>& matrixT,
              Matrix<T,U,MatrixStructureSquare,Distribution>& matrixTI,
              U dimension,
              char dir,
              MPI_Comm commWorld
            )
{
  // blah
}

template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void RTI3D<T,U,blasEngine>::InvertLower(
                  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixL,
                  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixLI,
                  U dimension,
                  MPI_Comm commWorld
                )
{
  // blah
}

template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void RTI3D<T,U,blasEngine>::InvertUpper(
                  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixU,
                  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixUI,
                  U dimension,
                  MPI_Comm commWorld
                )
{
  // blah
}
