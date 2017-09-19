/* Author: Edward Hutter */

#ifndef SQUAREMM3D_H_
#define SQUAREMM3D_H_

// System includes
#include <iostream>
#include <complex>
#include <vector>
#include <mpi.h>

// Local includes
#include "./../../../../../AlgebraicContainers/Matrix/Matrix.h"
#include "./../../../../../AlgebraicContainers/Matrix/MatrixSerializer.h"
#include "./../../../../../AlgebraicBLAS/blasEngine.h"
#include "./../MMvalidate/MMvalidate.h"

/*
  We can implement square MM for now, but soon, we will need triangular MM
    and triangular matrices, as well as Square-Triangular Multiplication and Triangular-Square Multiplication
  Also, we need to figure out what to do with Rectangular.
*/

template<typename T, typename U,
  template<typename,typename, template<typename,typename,int> class> class StructureA,
  template<typename,typename, template<typename,typename,int> class> class StructureB,
  template<typename,typename, template<typename,typename,int> class> class StructureC,
  template<typename,typename> class blasEngine>
class SquareMM3D
{

public:
  // Prevent any instantiation of this class.
  SquareMM3D() = delete;
  SquareMM3D(const SquareMM3D& rhs) = delete;
  SquareMM3D(SquareMM3D&& rhs) = delete;
  SquareMM3D& operator=(const SquareMM3D& rhs) = delete;
  SquareMM3D& operator=(SquareMM3D&& rhs) = delete;
  ~SquareMM3D() = delete;

  // Format: matrixA is X x Y
  //         matrixB is Y x Z
  //         matrixC is X x Z


  template<template<typename,typename,int> class Distribution>
  static void Multiply(
                        Matrix<T,U,StructureA,Distribution>& matrixA,
                        Matrix<T,U,StructureB,Distribution>& matrixB,
                        Matrix<T,U,StructureC,Distribution>& matrixC,
                        U dimensionX,
                        U dimensionY,
                        U dimensionZ,
                        MPI_Comm commWorld,
                        const blasEngineArgumentPackage& srcPackage
                      );

  template<template<typename,typename,int> class Distribution>
  static void Multiply(
                        Matrix<T,U,StructureA,Distribution>& matrixA,
                        Matrix<T,U,StructureB,Distribution>& matrixB,
                        Matrix<T,U,StructureC,Distribution>& matrixC,
                        U matrixAcutXstart,
                        U matrixAcutXend,
                        U matrixAcutYstart,
                        U matrixAcutYend,
                        U matrixBcutZstart,
                        U matrixBcutZend,
                        U matrixBcutXstart,
                        U matrixBcutXend,
                        U matrixCcutZstart,
                        U matrixCcutZend,
                        U matrixCcutYstart,
                        U matrixCcutYend,
                        MPI_Comm commWorld,
                        const blasEngineArgumentPackage& srcPackage,
                        bool cutA = true,
                        bool cutB = true,
                        bool cutC = true
                      );

  // Later on, I'd like an overloaded Multiply() method that took parameters to "cut up" the current 3 matrices (so 8 extra arguments)
  // This will require more Serialize methods, but the engine should remain the same, so I don't want that to change. Only at this 1st
  // level should anything be different.

private:
/*
  There is no reason to store state here. All we need this to do is to act as an interface. Nothing more.
  MatrixMultiplicationEngine will take over from here. Storing state here would make no sense.
*/
};

#include "SquareMM3D.hpp"

#endif /* SQUAREMM3D_H_ */
