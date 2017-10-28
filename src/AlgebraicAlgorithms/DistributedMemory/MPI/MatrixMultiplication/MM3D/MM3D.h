/* Author: Edward Hutter */

#ifndef MM3D_H_
#define MM3D_H_

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
class MM3D
{

public:
  // Prevent any instantiation of this class.
  MM3D() = delete;
  MM3D(const MM3D& rhs) = delete;
  MM3D(MM3D&& rhs) = delete;
  MM3D& operator=(const MM3D& rhs) = delete;
  MM3D& operator=(MM3D&& rhs) = delete;
  ~MM3D() = delete;

  // Format: matrixA is M x K
  //         matrixB is K x N
  //         matrixC is M x N

  // New design: user will specify via an argument to the overloaded Multiply() method what underlying BLAS routine he wants called.
  //             I think this is a reasonable assumption to make and will allow me to optimize each routine.

  template<template<typename,typename,int> class Distribution>
  static void Multiply(
                        Matrix<T,U,StructureA,Distribution>& matrixA,
                        Matrix<T,U,StructureB,Distribution>& matrixB,
                        Matrix<T,U,StructureC,Distribution>& matrixC,
                        U localDimensionM,
                        U localDimensionN,
                        U localDimensionK,
                        MPI_Comm commWorld,
                        const blasEngineArgumentPackage_gemm<T>& srcPackage
                      );

  template<template<typename,typename,int> class Distribution>
  static void Multiply(
                        Matrix<T,U,StructureA,Distribution>& matrixA,
                        Matrix<T,U,StructureB,Distribution>& matrixB,
                        U localDimensionM,
                        U localDimensionN,
                        MPI_Comm commWorld,
                        const blasEngineArgumentPackage_trmm<T>& srcPackage
                      );

  template<template<typename,typename,int> class Distribution>
  static void Multiply(
                        Matrix<T,U,StructureA,Distribution>& matrixA,
                        Matrix<T,U,StructureB,Distribution>& matrixB,		// MatrixB represents MatrixC in a typical SYRK routine. matrixB will hold the output
                        U localDimensionN,
                        U localDimensionK,
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
                        MPI_Comm commWorld,
                        const blasEngineArgumentPackage_trmm<T>& srcPackage,
                        bool cutA = true,
                        bool cutB = true
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
                        MPI_Comm commWorld,
                        const blasEngineArgumentPackage_syrk<T>& srcPackage,
                        bool cutA = true,
                        bool cutB = true
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
				Matrix<T,U,MatrixStructureSquare, Distribution>& matrixDest,
				std::vector<T>& data,
				bool isRoot
                          );

  template<template<typename,typename, template<typename,typename,int> class> class StructureArg,
    template<typename,typename,int> class Distribution>					// Added additional template parameters just for this method
  static Matrix<T,U,StructureArg,Distribution>& getSubMatrix(
				Matrix<T,U,StructureArg, Distribution>& srcMatrix,
				Matrix<T,U,StructureArg, Distribution>& fillMatrix,
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

#include "MM3D.hpp"

#endif /* MM3D_H_ */