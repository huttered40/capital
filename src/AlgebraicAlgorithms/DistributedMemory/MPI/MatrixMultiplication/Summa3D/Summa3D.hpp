/* Author: Edward Hutter */


template<typename T, typename U,
  template<typename,typename, template<typename,typename,int> class> class StructureA,
  template<typename,typename, template<typename,typename,int> class> class StructureB,
  template<typename,typename, template<typename,typename,int> class> class StructureC>
template<template<typename,typename,int> class Distribution>
void Summa3D<T,U,StructureA,StructureB,StructureC>::Multiply(
                                                              const Matrix<T,U,StructureA,Distribution>& matrixA,
                                                              const Matrix<T,U,StructureB,Distribution>& matrixB,
                                                                    Matrix<T,U,StructureC,Distribution>& matrixC,
                                                              U dimensionX,
                                                              U dimensionY,
                                                              U dimensionZ,
                                                              int pGridCoordX,
                                                              int pGridCoordY,
                                                              int pGridCoordZ,
                                                              MPI_Comm commWorld
                                                            )
{
  // Broadcast first
  // Broadcast next
  // Call the right cblas routine
  // broadcast again


}

template<typename T, typename U,
  template<typename,typename, template<typename,typename,int> class> class StructureA,
  template<typename,typename, template<typename,typename,int> class> class StructureB,
  template<typename,typename, template<typename,typename,int> class> class StructureC>
template<template<typename,typename,int> class Distribution>
void Summa3D<T,U,StructureA,StructureB,StructureC>::Multiply(
                                                              const Matrix<T,U,StructureA,Distribution>& matrixA,
                                                              const Matrix<T,U,StructureB,Distribution>& matrixB,
                                                                    Matrix<T,U,StructureC,Distribution>& matrixC,
                                                              U matrixAcutXstart,
                                                              U matrixAcutXend,
                                                              U matrixAcutYstart,
                                                              U matrixAcutYend,
                                                              U matrixBcutYstart,
                                                              U matrixBcutYend,
                                                              U matrixBcutZstart,
                                                              U matrixBcutZend,
                                                              U matrixCcutXstart,
                                                              U matrixCcutXend,
                                                              U matrixCcutZstart,
                                                              U matrixCcutZend,
                                                              int pGridCoordX,
                                                              int pGridCoordY,
                                                              int pGridCoordZ,
                                                              MPI_Comm commWorld
                                                            )
{
}
