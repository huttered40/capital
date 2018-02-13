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
#include "./../../MatrixMultiplication/MM3D/MM3D.h"
#include "./../../../Matrix/Matrix.h"
#include "./../../../Matrix/MatrixSerializer.h"
#include "./../../../AlgebraicBLAS/blasEngine.h"

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
                        MPI_Comm commWorld
                      );

  // We require that for a 3D algorithm, Q is square and R is square
  template<template<typename,typename,int> class Distribution>
  static void validateLocal3D(
                        Matrix<T,U,MatrixStructureRectangle,Distribution>& matrixA,
                        Matrix<T,U,MatrixStructureRectangle,Distribution>& myQ,
                        Matrix<T,U,MatrixStructureSquare,Distribution>& myR,
                        MPI_Comm commWorld
                      );

  template<template<typename,typename,int> class Distribution>
  static void validateLocalTunable(
                        Matrix<T,U,MatrixStructureRectangle,Distribution>& matrixA,
                        Matrix<T,U,MatrixStructureRectangle,Distribution>& myQ,
                        Matrix<T,U,MatrixStructureSquare,Distribution>& myR,
                        int gridDimensionD,
                        int gridDimensionC,
                        MPI_Comm commWorld
		      );
private:
  // 1D helper routines
  static T getResidual1D_RowCyclic(std::vector<T>& myMatrix, std::vector<T>& solutionMatrix, U globalDimensionX, U globalDimensionY, U localDimensionY, MPI_Comm commWorld);
  static T getResidual1D_Full(std::vector<T>& myMatrix, std::vector<T>& solutionMatrix, U globalDimensionX, U globalDimensionY, MPI_Comm commWorld);
  static T testOrthogonality1D(std::vector<T>& myQ, U globalDimensionX, U globalDimensionY, U localDimensionY, MPI_Comm commWorld);
  static T getResidual1D(std::vector<T>& myA, std::vector<T>& myQ, std::vector<T>& myR, U globalDimensionX, U globalDimensionY, U localDimensionY, MPI_Comm commWorld);

  // 3D helper routines
  template<template<typename,typename,int> class Distribution>
  static T getResidual3D(Matrix<T,U,MatrixStructureRectangle,Distribution>& myA,
                         Matrix<T,U,MatrixStructureRectangle,Distribution>& myQ,
                         Matrix<T,U,MatrixStructureSquare,Distribution>& myR,
                         U globalDimensionM, U globalDimensionN, MPI_Comm commWorld);

  template<template<typename,typename,int> class Distribution>
  static T testOrthogonality3D(Matrix<T,U,MatrixStructureRectangle,Distribution>& myQ,
                               U globalDimensionM, U globalDimensionN, MPI_Comm commWorld);

  template<template<typename,typename,int> class Distribution>
  static T getResidualTunable(Matrix<T,U,MatrixStructureRectangle,Distribution>& myA,
                         Matrix<T,U,MatrixStructureRectangle,Distribution>& myQ,
                         Matrix<T,U,MatrixStructureSquare,Distribution>& myR,
                         U globalDimensionM, U globalDimensionN, int gridDimensionD, int gridDimensionC, MPI_Comm commWorld,
                         std::tuple<MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm> tunableCommunicators);

  template<template<typename,typename,int> class Distribution>
  static std::vector<T> getReferenceMatrix1D(
                        			Matrix<T,U,MatrixStructureRectangle,Distribution>& myMatrix,
						U globalDimensionX,
						U globalDimensionY,
						U localDimensionY,
						U key,
						MPI_Comm commWorld
					    );
};

// Templated classes require method definition within the same unit as method declarations (correct wording?)
#include "QRvalidate.hpp"

#endif /* QRVALIDATE_H_ */
