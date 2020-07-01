/* Author: Edward Hutter */

#ifndef QR__VALIDATE_H_
#define QR__VALIDATE_H_

#include "../../src/alg/alg.h"

// These static methods will take the matrix in question, distributed in some fashion across the processors
//   and use them to calculate the residual or error.

namespace qr{
template<typename AlgType>
class validate{
public:

  template<typename MatrixType, typename ArgType, typename RectCommType>
  static typename MatrixType::ScalarType orthogonality(const MatrixType& A, ArgType& args, RectCommType&& RectTopo);
  
  template<typename MatrixType, typename ArgType, typename RectCommType>
  static typename MatrixType::ScalarType residual(const MatrixType& A, ArgType& args, RectCommType&& RectTopo);
};
}

// Templated classes require method definition within the same unit as method declarations (correct wording?)
#include "validate.hpp"

#endif /* QR__VALIDATE_H_ */
