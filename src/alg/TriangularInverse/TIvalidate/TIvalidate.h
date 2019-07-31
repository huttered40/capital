/* Author: Edward Hutter */

#ifndef TIVALIDATE_H_
#define TIVALIDATE_H_

// System includes
#include <iostream>
#include <tuple>
#include <cmath>
#include <string>

// Local includes
#include "./../../../Util/shared.h"
#include "./../../../Timer/CTFtimer.h"
#include "./../../../Matrix/Matrix.h"
#include "./../../../Matrix/MatrixSerializer.h"

// These static methods will take the matrix in question, distributed in some fashion across the processors
//   and use them to calculate the residual or error.

template<typename T, typename U>
class TIvalidate
{
public:
  TIvalidate() = delete;
  ~TIvalidate() = delete;
  TIvalidate(const TIvalidate& rhs) = delete;
  TIvalidate(TIvalidate&& rhs) = delete;
  TIvalidate& operator=(const TIvalidate& rhs) = delete;
  TIvalidate& operator=(TIvalidate&& rhs) = delete;

  template<template<typename,typename,int> class Distribution>
  static void validateTI_Local(
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixSol_TI,
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

  template<template<typename,typename,int> class Distribution>
  static std::vector<T> getReferenceMatrix(
                        			Matrix<T,U,MatrixStructureSquare,Distribution>& myMatrix,
						U localDimension,
						U globalDimension,
						U key,
						std::tuple<MPI_Comm, int, int, int, int> commInfo
					  );
};

// Templated classes require method definition within the same unit as method declarations (correct wording?)
#include "TIvalidate.hpp"

#endif /* TIVALIDATE_H_ */
