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
                                                              U dimensionZ
                                                            )
{
  std::cout << "Ed is here\n";
}
