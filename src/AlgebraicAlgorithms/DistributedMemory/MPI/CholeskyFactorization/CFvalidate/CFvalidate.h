/* Author: Edward Hutter */

#ifndef CFVALIDATE_H_
#define CFVALIDATE_H_

// System includes
#include <iostream>
#include <tuple>
#include <cmath>
#include <string>
#include "/home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/LAPACKE/include/lapacke.h"

// Local includes
#include "./../../../../../AlgebraicContainers/Matrix/Matrix.h"
#include "./../../../../../AlgebraicContainers/Matrix/MatrixSerializer.h"
#include "./../../../../../Timer/Timer.h"

// These static methods will take the matrix in question, distributed in some fashion across the processors
//   and use them to calculate the residual or error.

template<typename T, typename U>
class CFvalidate
{
public:
  CFvalidate() = delete;
  ~CFvalidate() = delete;
  CFvalidate(const CFvalidate& rhs) = delete;
  CFvalidate(CFvalidate&& rhs) = delete;
  CFvalidate& operator=(const CFvalidate& rhs) = delete;
  CFvalidate& operator=(CFvalidate&& rhs) = delete;

  template<template<typename,typename,int> class Distribution>
  static std::pair<T,T> validateCF_Local(
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixSol_CF,
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixSol_TI,
                        U localDimension,
                        U globalDimension,
			char dir,
                        MPI_Comm commWorld
                      );

private:

  static T getResidualTriangleLower(
				std::vector<T>& myValues,
				std::vector<T>& lapackValues,
				U localDimension,
				U globalDimension,
		                std::tuple<MPI_Comm, int, int, int, int> commInfo
			    );

  static T getResidualTriangleUpper(
				std::vector<T>& myValues,
				std::vector<T>& lapackValues,
				U localDimension,
				U globalDimension,
		                std::tuple<MPI_Comm, int, int, int, int> commInfo
			    );
};

// Templated classes require method definition within the same unit as method declarations (correct wording?)
#include "CFvalidate.hpp"

#endif /* CFVALIDATE_H_ */
