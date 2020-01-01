/* Author: Edward Hutter */

#ifndef CHOLESKY__VALIDATE_H_
#define CHOLESKY__VALIDATE_H_

#include "../../../alg/alg.h"

// These static methods will take the matrix in question, distributed in some fashion across the processors
//   and use them to calculate the residual or error.

namespace cholesky{

template<typename AlgType>
class validate{
public:
  template<typename MatrixAType, typename MatrixTriType, typename CommType>
  static typename MatrixAType::ScalarType invoke(MatrixAType& A, MatrixTriType& Tri, char dir, CommType&& CommInfo);
};
}

// Templated classes require method definition within the same unit as method declarations (correct wording?)
#include "validate.hpp"

#endif /* CHOLESKY__VALIDATE_H_ */
