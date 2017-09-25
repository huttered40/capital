/* Author: Edward Hutter */

#ifndef SQUAREMM3D_H_
#define SQUAREMM3D_H_

// System includes
#include <iostream>
#include <complex>
#include <vector>
#include <utility>
#include <tuple>
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
  template<typename,typename, template<typename,typename,int> class> class StructureC = MatrixStructureSquare,
  template<typename,typename> class blasEngine = cblasEngine>
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

  // New design: user will specify via an argument to the overloaded Multiply() method what underlying BLAS routine he wants called.
  //             I think this is a reasonable assumption to make and will allow me to optimize each routine.

  template<template<typename,typename,int> class Distribution>
  static void Multiply(
                        Matrix<T,U,StructureA,Distribution>& matrixA,
                        Matrix<T,U,StructureB,Distribution>& matrixB,
                        Matrix<T,U,StructureC,Distribution>& matrixC,
                        U dimensionX,
                        U dimensionY,
                        U dimensionZ,
                        MPI_Comm commWorld,
                        const blasEngineArgumentPackage_gemm<T>& srcPackage
                      );

  template<template<typename,typename,int> class Distribution>
  static void Multiply(
                        Matrix<T,U,StructureA,Distribution>& matrixA,
                        Matrix<T,U,StructureB,Distribution>& matrixB,
                        U dimensionX,
                        U dimensionY,
                        U dimensionZ,
                        MPI_Comm commWorld,
                        const blasEngineArgumentPackage_trmm<T>& srcPackage
                      );

  template<template<typename,typename,int> class Distribution>
  static void Multiply(
                        Matrix<T,U,StructureA,Distribution>& matrixA,
                        Matrix<T,U,StructureB,Distribution>& matrixB,		// MatrixB represents MatrixC in a typical SYRK routine. matrixB will hold the output
                        U dimensionX,
                        U dimensionY,
                        U dimensionZ,
                        MPI_Comm commWorld,
                        const blasEngineArgumentPackage_syrk<T>& srcPackage
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
                        const blasEngineArgumentPackage_gemm<T>& srcPackage,
                        bool cutA = true,
                        bool cutB = true,
                        bool cutC = true
                      );

  template<template<typename,typename,int> class Distribution>
  static void Multiply(
                        Matrix<T,U,StructureA,Distribution>& matrixA,
                        Matrix<T,U,StructureB,Distribution>& matrixB,
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
                        const blasEngineArgumentPackage_trmm<T>& srcPackage,
                        bool cutA = true,
                        bool cutB = true,
                        bool cutC = true
                      );

  template<template<typename,typename,int> class Distribution>
  static void Multiply(
                        Matrix<T,U,StructureA,Distribution>& matrixA,
                        Matrix<T,U,StructureB,Distribution>& matrixB,		// MatrixB represents MatrixC in a typical SYRK routine. matrixB will hold the output
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
                        const blasEngineArgumentPackage_syrk<T>& srcPackage,
                        bool cutA = true,
                        bool cutB = true,
                        bool cutC = true
                      );

private:

  static void BroadcastPanels(
				std::vector<T>& data,
				U size,
				bool isRoot,
				int pGridCoordZ,
				MPI_Comm panelComm
			     );

  template<template<typename,typename, template<typename,typename,int> class> class StructureArg,
    template<typename,typename,int> class Distribution>					// Added additional template parameters just for this method
  static T* getEnginePtr(
				Matrix<T,U,StructureArg, Distribution>& matrixArg,
				std::vector<T>& data,
				bool isRoot
                          );

  template<template<typename,typename, template<typename,typename,int> class> class StructureArg,
    template<typename,typename,int> class Distribution>					// Added additional template parameters just for this method
  static T* getSubMatrix(
				Matrix<T,U,StructureArg, Distribution>& matrixArg,
				U matrixArgColumnStart,
				U matrixArgColumnEnd,
				U matrixArgRowStart,
				U matrixArgRowEnd,
				U globalDiff,
				bool getSub
			  );

/*
  There is no reason to store state here. All we need this to do is to act as an interface. Nothing more.
  MatrixMultiplicationEngine will take over from here. Storing state here would make no sense.
*/
};

#include "SquareMM3D.hpp"

#endif /* SQUAREMM3D_H_ */
