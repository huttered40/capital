/* Author: Edward Hutter */

#ifndef SUMMA3D_H_
#define SUMMA3D_H_

// System includes
#include <iostream>
#include <vector>

// Local includes
#include "./../../../../../AlgebraicStructures/Matrix/Matrix.h"
#include "./../MatrixMultiplicationEngine/MatrixMultiplicationEngine.h"

/*
  We can implement square MM for now, but soon, we will need triangular MM
    and triangular matrices, as well as Square-Triangular Multiplication and Triangular-Square Multiplication
  Also, we need to figure out what to do with Rectangular.
*/

template<typename T, typename U,
  template<typename,typename, template<typename,typename,int> class> class StructureA,
  template<typename,typename, template<typename,typename,int> class> class StructureB,
  template<typename,typename, template<typename,typename,int> class> class StructureC>
class Summa3D
{

public:
  // Prevent any instantiation of this class.
  Summa3D() = delete;
  Summa3D(const Summa3D& rhs) = delete;
  Summa3D(Summa3D&& rhs) = delete;
  Summa3D& operator=(const Summa3D& rhs) = delete;
  Summa3D& operator=(Summa3D&& rhs) = delete;
  ~Summa3D() = delete;

  // For now, lets just use 1 method.
  template<template<typename,typename,int> class Distribution>
  static void Multiply(
                        const Matrix<T,U,StructureA,Distribution>& matrixA,
                        const Matrix<T,U,StructureB,Distribution>& matrixB,
                              Matrix<T,U,StructureC,Distribution>& matrixC,
                        U dimensionX,
                        U dimensionY,
                        U dimensionZ
                      );

private:
/*
  There is no reason to store state here. All we need this to do is to act as an interface. Nothing more.
  MatrixMultiplicationEngine will take over from here. Storing state here would make no sense.
*/
};

#include "Summa3D.hpp"

#endif /* SUMMA3D_H_ */
