/* Author: Edward Hutter */

#ifndef QRVALIDATE_H_
#define QRVALIDATE_H_

#include "./../../Algorithms.h"
#include "./../../../Util/validation.h"

// These static methods will take the matrix in question, distributed in some fashion across the processors
//   and use them to calculate the residual or error.

namespace qr{
class QRvalidate{
public:
  // We require that for a 1D algorithm, Q is rectangular and R is square
  template<typename MatrixAType, typename MatrixQType, typename MatrixRType>
  static void validateLocal1D(MatrixAType& matrixA, MatrixQType& matrixQ, MatrixRType& matrixR, MPI_Comm commWorld);

  // We require that for a 3D algorithm, Q is square and R is square
  template<typename MatrixAType, typename MatrixQType, typename MatrixRType>
  static std::pair<typename MatrixAType::ScalarType,typename MatrixAType::ScalarType>
         validateParallel3D(MatrixAType& matrixA, MatrixQType& matrixQ, MatrixRType& matrixR, MPI_Comm commWorld,
                            std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D);

  template<typename MatrixAType, typename MatrixQType, typename MatrixRType>
  static std::pair<typename MatrixAType::ScalarType,typename MatrixAType::ScalarType>
         validateParallelTunable(MatrixAType& matrixA, MatrixQType& matrixQ, MatrixRType& myR,
                                 size_t gridDimensionD, size_t gridDimensionC, MPI_Comm commWorld, std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm>& commInfoTunable);
private:
  // 1D helper routines
  template<typename T, typename U>
  static T getResidual1D_RowCyclic(std::vector<T>& myMatrix, std::vector<T>& solutionMatrix, U globalDimensionX, U globalDimensionY, U localDimensionY, MPI_Comm commWorld);
  template<typename T, typename U>
  static T getResidual1D_Full(std::vector<T>& myMatrix, std::vector<T>& solutionMatrix, U globalDimensionX, U globalDimensionY, MPI_Comm commWorld);
  template<typename T, typename U>
  static T testOrthogonality1D(std::vector<T>& myQ, U globalDimensionX, U globalDimensionY, U localDimensionY, MPI_Comm commWorld);
  template<typename T, typename U>
  static T getResidual1D(std::vector<T>& myA, std::vector<T>& myQ, std::vector<T>& myR, U globalDimensionX, U globalDimensionY, U localDimensionY, MPI_Comm commWorld);

/*
  template<template<typename,typename,size_t> class Distribution>
  static T testOrthogonality3D(Matrix<T,U,MatrixStructureRectangle,Distribution>& myQ,
                               U globalDimensionM, U globalDimensionN, MPI_Comm commWorld);
*/

  template<typename MatrixType>
  static std::vector<typename MatrixType::ScalarType>
         getReferenceMatrix1D(MatrixType& myMatrix, typename MatrixType::DimensionType globalDimensionX, typename MatrixType::DimensionType globalDimensionY,
                              typename MatrixType::DimensionType localDimensionY, size_t key, MPI_Comm commWorld);
};
}

// Templated classes require method definition within the same unit as method declarations (correct wording?)
#include "QRvalidate.hpp"

#endif /* QRVALIDATE_H_ */
