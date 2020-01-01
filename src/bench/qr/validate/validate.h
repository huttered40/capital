/* Author: Edward Hutter */

#ifndef QR__VALIDATE_H_
#define QR__VALIDATE_H_

#include "../../../alg/alg.h"

// These static methods will take the matrix in question, distributed in some fashion across the processors
//   and use them to calculate the residual or error.

namespace qr{
template<typename AlgType>
class validate{
public:
  template<typename MatrixAType, typename MatrixQType, typename MatrixRType, typename CommType>
  static std::pair<typename MatrixAType::ScalarType,typename MatrixAType::ScalarType>
           invoke(MatrixAType& A, MatrixQType& Q, MatrixRType& R, CommType&& CommInfo);

private:
  template<typename MatrixType, typename RectCommType, typename SquareCommType>
  static typename MatrixType::ScalarType
  orth(MatrixType& Q, RectCommType&& RectCommInfo, SquareCommType&& SquareCommInfo);
  
  template<typename MatrixQType, typename MatrixRType, typename MatrixAType, typename RectCommType, typename SquareCommType>
  static typename MatrixAType::ScalarType
  residual(MatrixQType& Q, MatrixRType& R, MatrixAType& A, RectCommType&& RectCommInfo, SquareCommType&& SquareCommInfo);
};
}

// Templated classes require method definition within the same unit as method declarations (correct wording?)
#include "validate.hpp"

#endif /* QR__VALIDATE_H_ */
