/* Author: Edward Hutter */

#ifndef MMVALIDATE_H_
#define MMVALIDATE_H_

#include "./../../Algorithms.h"
#include "./../../../Util/validation.h"

// These static methods will take the matrix in question, distributed in some fashion across the processors
//   and use them to calculate the residual or error.

class MMvalidate{
public:
  // This method does not depend on the structures. There are situations where have square structured matrices,
  //   but want to use trmm, for example, so disregard the structures in this entire class.
  
  template<typename MatrixAType, typename MatrixBType, typename MatrixCType>
  static void validateLocal(MatrixAType& matrixA, MatrixBType& matrixB, MatrixCType& matrixC, MPI_Comm commWorld, const blasEngineArgumentPackage_gemm<typename MatrixAType::ScalarType>& srcPackage);

  template<typename MatrixAType, typename MatrixBinType, typename MatrixBoutType>
  static void validateLocal(MatrixAType& matrixA, MatrixBinType& matrixBin, MatrixBoutType& matrixBout, MPI_Comm commWorld, const blasEngineArgumentPackage_trmm<typename MatrixAType::ScalarType>& srcPackage);

private:
  template<typename T, typename U>
  static T getResidual(std::vector<T>& myValues, std::vector<T>& blasValues,
                       U localDimensionM, U localDimensionN, U globalDimensionM, U globalDimensionN, std::tuple<MPI_Comm,size_t,size_t,size_t,size_t> commInfo);
};

// Templated classes require method definition within the same unit as method declarations (correct wording?)
#include "MMvalidate.hpp"

#endif /* MMVALIDATE_H_ */
