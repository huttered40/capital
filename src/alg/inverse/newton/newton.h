/* Author: Edward Hutter */

#ifndef INVERSE__NEWTON_H_
#define INVERSE__NEWTON_H_

#include "./../../alg.h"
#include "./../../matmult/summa/summa.h"

namespace inverse{

class newton{
public:
  template<typename MatrixType, typename CommType>
  static void invoke(MatrixType& matrix, CommType&& CommInfo, typename MatrixType::ScalarType tol=1e-14, size_t max_iter=100);
protected:
};
}

#include "newton.hpp"

#endif /* INVERSE__NEWTON_H_ */
