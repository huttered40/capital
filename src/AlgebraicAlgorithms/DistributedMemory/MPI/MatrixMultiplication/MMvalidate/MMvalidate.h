/* Author: Edward Hutter */

#ifndef MMVALIDATE_H_
#define MMVALIDATE_H_

// System includes
#include <iostream>
#include <tuple>
#include <cmath>

// Local includes
#include "./../../../../../AlgebraicContainers/Matrix/Matrix.h"
#include "./../../../../../AlgebraicContainers/Matrix/MatrixSerializer.h"
#include "./../../../../../AlgebraicBLAS/blasEngine.h"


// These static methods will take the matrix in question, distributed in some fashion across the processors
//   and use them to calculate the residual or error.

template<typename T, typename U, template<typename,typename> class blasEngine>
class MMvalidate
{
public:
  MMvalidate() = delete;
  ~MMvalidate() = delete;
  MMvalidate(const MMvalidate& rhs) = delete;
  MMvalidate(MMvalidate&& rhs) = delete;
  MMvalidate& operator=(const MMvalidate& rhs) = delete;
  MMvalidate& operator=(MMvalidate&& rhs) = delete;

  // This method does not depend on the structures. There are situations where have square structured matrices,
  //   but want to use trmm, for example, so disregard the structures in this entire class.
  
  // However, we still need to template this method because the method needs to be aware of the Matrix's template parameters
  //   in order to use it as a method argument.

  template<template<typename,typename,int> class Distribution>
  static void validateLocal(
                        Matrix<T,U,MatrixStructureSquare,Distribution>& myMatrix,
                        U localDimensionM,
                        U localDimensionN,
                        U localDimensionK,
                        U globalDimensionM,
                        U globalDimensionN,
                        U globalDimensionK,
                        MPI_Comm commWorld,
                        const blasEngineArgumentPackage_gemm<T>& srcPackage
                      );

  template<template<typename,typename,int> class Distribution>
  static void validateLocal(
                        Matrix<T,U,MatrixStructureSquare,Distribution>& myMatrix,
                        U localDimensionM,
                        U localDimensionN,
                        U globalDimensionM,
                        U globalDimensionN,
                        MPI_Comm commWorld,
                        const blasEngineArgumentPackage_trmm<T>& srcPackage
                      );

  template<template<typename,typename,int> class Distribution>
  static void validateLocal(
                        Matrix<T,U,MatrixStructureSquare,Distribution>& myMatrix,
                        U localDimensionN,
                        U localDimensionK,
                        U globalDimensionN,
                        U globalDimensionK,
                        MPI_Comm commWorld,
                        const blasEngineArgumentPackage_syrk<T>& srcPackage
                      );

private:

  static T getResidualSquare(
				std::vector<T>& myValues,
				std::vector<T>& blasValues,
				U localDimensionM,
				U localDimensionN,
				U globalDimensionM,
				U globalDimensionN,
		                std::tuple<MPI_Comm, int, int, int, int> commInfo
			    );

  template<template<typename,typename,int> class Distribution>
  static std::vector<T> getReferenceMatrix(
                        			Matrix<T,U,MatrixStructureSquare,Distribution>& myMatrix,
						U localNumColumns,
						U localNumRows,
						U globalNumColumns,
						U globalNumRows,
						U key,
						std::tuple<MPI_Comm, int, int, int, int> commInfo
					  );
};

// Templated classes require method definition within the same unit as method declarations (correct wording?)
#include "MMvalidate.hpp"

#endif /* MMVALIDATE_H_ */
