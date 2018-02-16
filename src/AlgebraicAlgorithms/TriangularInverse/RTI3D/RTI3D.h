/*
  Author: Edward Hutter
*/

#ifndef RTI3D_H_
#define RTI3D_H_

// System includes
#include <iostream>
#include <complex>
#include <mpi.h>
#include "/home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/lapack-netlib/LAPACKE/include/lapacke.h"

// Local includes
#include "./../../../Util/shared.h"
#include "./../../../Matrix/Matrix.h"
#include "./../../../Timer/Timer.h"
#include "./../../../Matrix/MatrixSerializer.h"
#include "./../../../AlgebraicBLAS/blasEngine.h"
#include "./../../MatrixMultiplication/MM3D/MM3D.h"

template<typename T, typename U, template<typename, typename> class blasEngine>
class RTI3D
{
public:
  // Prevent instantiation of this class
  RTI3D() = delete;
  RTI3D(const RTI3D& rhs) = delete;
  RTI3D(RTI3D&& rhs) = delete;
  RTI3D<T,U,blasEngine>& operator=(const RTI3D& rhs) = delete;
  RTI3D<T,U,blasEngine>& operator=(RTI3D&& rhs) = delete;

  template<template<typename,typename,int> class Distribution>
  static void Invert(
                      Matrix<T,U,MatrixStructureSquare,Distribution>& matrixT,
                      Matrix<T,U,MatrixStructureSquare,Distribution>& matrixTI,
                      char dir,
                      MPI_Comm commWorld
                    );
private:
  template<template<typename,typename,int> class Distribution>
  static void InvertLower(
                      Matrix<T,U,MatrixStructureSquare,Distribution>& matrixL,
                      Matrix<T,U,MatrixStructureSquare,Distribution>& matrixLI,
                      U localDimension,
                      U startX,
                      U endX,
                      U startY,
                      U endY,
                      int key,
                      MPI_Comm commWorld
                    );

  template<template<typename,typename,int> class Distribution>
  static void InvertUpper(
                      Matrix<T,U,MatrixStructureSquare,Distribution>& matrixU,
                      Matrix<T,U,MatrixStructureSquare,Distribution>& matrixUI,
                      U localDimension,
                      int key,
                      MPI_Comm commWorld
                    );

  template<template<typename,typename,int> class Distribution>
  static void sliceExchangeBase(
                  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixT,
                  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixTI,
                  U localDimension,
                  U startX,
                  U endX,
                  U startY,
                  U endY,
                  MPI_Comm commWorld,
                  char dir
                );

  template<template<typename,typename,int> class Distribution>
  static std::vector<T> blockedToCyclicTransformation(
									Matrix<T,U,MatrixStructureSquare,Distribution>& matT,
									U localDimension,
									U globalDimension,
									U matTstartX,
									U matTendX,
									U matTstartY,
									U matTendY,
									int pGridDimensionSize,
									MPI_Comm slice2Dcomm
								);

  static void cyclicToLocalTransformation(
								std::vector<T>& storeT,
								U localDimension,
								U globalDimension,
								int pGridDimensionSize,
								int rankSlice,
								char dir
							);
};

#include "RTI3D.hpp"
#endif /* RTI3D_H_ */
