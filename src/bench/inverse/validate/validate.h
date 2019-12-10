/* Author: Edward Hutter */

#ifndef INVERSE__VALIDATE_H_
#define INVERSE__VALIDATE_H_

#include "../../../alg/alg.h"

// These static methods will take the matrix in question, distributed in some fashion across the processors
//   and use them to calculate the residual or error.

namespace inverse{

template<typename AlgType>
class validate{
public:
  template<typename MatrixType, typename CommType>
  static typename MatrixType::ScalarType invoke(MatrixType& matrixA, MatrixType& matrixB, CommType&& CommInfo);

private:
};
}

// Templated classes require method definition within the same unit as method declarations (correct wording?)
#include "validate.hpp"

#endif /* INVERSE__VALIDATE_H_ */
