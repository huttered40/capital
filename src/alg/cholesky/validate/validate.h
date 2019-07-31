/* Author: Edward Hutter */

#ifndef CHOLESKY__VALIDATE_H_
#define CHOLESKY__VALIDATE_H_

#include "./../../alg.h"
#include "./../../../util/validation.h"

// These static methods will take the matrix in question, distributed in some fashion across the processors
//   and use them to calculate the residual or error.

namespace cholesky{
class CFvalidate{
public:
  template<typename MatrixAType, typename MatrixSolType>
  static void validateLocal(MatrixAType& matrixA, MatrixSolType& matrixSol, char dir, MPI_Comm commWorld);

  template<typename MatrixAType, typename MatrixTriType>
  static typename MatrixAType::ScalarType invoke(MatrixAType& matrixA, MatrixTriType& matrixTri,
                            char dir, MPI_Comm commWorld, std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D);

private:
  template<typename T, typename U>
  static T getResidualTriangleLower(std::vector<T>& myValues, std::vector<T>& lapackValues, U localDimension, U globalDimension, std::tuple<MPI_Comm,size_t,size_t,size_t,size_t> commInfo);

  template<typename T, typename U>
  static T getResidualTriangleUpper(std::vector<T>& myValues, std::vector<T>& lapackValues, U localDimension, U globalDimension, std::tuple<MPI_Comm,size_t,size_t,size_t,size_t> commInfo);
};
}

// Templated classes require method definition within the same unit as method declarations (correct wording?)
#include "validate.hpp"

#endif /* CHOLESKY__VALIDATE_H_ */
