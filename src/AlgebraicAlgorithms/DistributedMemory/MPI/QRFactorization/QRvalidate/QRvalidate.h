/* Author: Edward Hutter */

#ifndef QRVALIDATE_H_
#define QRVALIDATE_H_

// System includes
#include <iostream>
#include <tuple>
#include <cmath>
#include "/home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/LAPACKE/include/lapacke.h"

// Local includes
#include "./../../../../../AlgebraicContainers/Matrix/Matrix.h"
#include "./../../../../../AlgebraicContainers/Matrix/MatrixSerializer.h"

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

  template<template<typename,typename,int> class Distribution>
  static std::pair<T,T> validateQR_Local(
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixSol_Q,
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixSol_R,
                        U localDimension,
                        U globalDimension,
                        MPI_Comm commWorld
                      );

private:

  static T getResidualTriangle(
				std::vector<T>& myValues,
				std::vector<T>& lapackValues,
				U localDimension,
				U globalDimension,
		                std::tuple<MPI_Comm, int, int, int, int> commInfo
			    );
};

// Templated classes require method definition within the same unit as method declarations (correct wording?)
#include "QRvalidate.hpp"

#endif /* QRVALIDATE_H_ */
