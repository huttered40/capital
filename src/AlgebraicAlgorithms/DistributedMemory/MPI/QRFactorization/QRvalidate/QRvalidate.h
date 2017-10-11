/* Author: Edward Hutter */

#ifndef QRVALIDATE_H_
#define QRVALIDATE_H_

// System includes
#include <iostream>
#include <tuple>
#include <cmath>
#include <mpi.h>
#include "/home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/LAPACKE/include/lapacke.h"

// Local includes
#include "./../../../../../AlgebraicContainers/Matrix/Matrix.h"
#include "./../../../../../AlgebraicContainers/Matrix/MatrixSerializer.h"
#include "./../../../../../AlgebraicBLAS/blasEngine.h"

// These static methods will take the matrix in question, distributed in some fashion across the processors
//   and use them to calculate the residual or error.

template<typename T, typename U>
class QRvalidate
{
public:
  QRvalidate() = delete;
  ~QRvalidate() = delete;
  QRvalidate(const QRvalidate& rhs) = delete;
  QRvalidate(QRvalidate&& rhs) = delete;
  QRvalidate& operator=(const QRvalidate& rhs) = delete;
  QRvalidate& operator=(QRvalidate&& rhs) = delete;

  // We require that for a 1D algorithm, Q is rectangular and R is square
  template<template<typename,typename,int> class Distribution>
  static void validateLocal1D(
                        Matrix<T,U,MatrixStructureRectangle,Distribution>& matrixA,
                        Matrix<T,U,MatrixStructureRectangle,Distribution>& matrixSol_Q,
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixSol_R,
                        U globalDimensionX,
                        U globalDimensionY,
                        MPI_Comm commWorld
                      );

  // We require that for a 3D algorithm, Q is square and R is square
  template<template<typename,typename,int> class Distribution>
  static void validateLocal3D(
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixSol_Q,
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixSol_R,
                        U globalDimensionX,
                        U globalDimensionY,
                        MPI_Comm commWorld
                      );

private:

  static T getResidual1D_Q(std::vector<T>& myQ, std::vector<T>& solQ, U globalDimensionX, U globalDimensionY, MPI_Comm commWorld);
  static T getResidual1D_R(std::vector<T>& myR, std::vector<T>& solR, U globalDimensionX, U globalDimensionY, MPI_Comm commWorld);
  static T testOrthgonality(std::vector<T>& myQ, std::vector<T>& solQ, U globalDimensionX, U globalDimensionY, MPI_Comm commWorld);
  static T testResidual(std::vector<T>& myR, std::vector<T>& solR, U globalDimensionX, U globalDimensionY, MPI_Comm commWorld);
  static T testComputedQR(std::vector<T>& myR, std::vector<T>& solR, U globalDimensionX, U globalDimensionY, MPI_Comm commWorld);

};

// Templated classes require method definition within the same unit as method declarations (correct wording?)
#include "QRvalidate.hpp"

#endif /* QRVALIDATE_H_ */
