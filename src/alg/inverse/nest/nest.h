/* Author: Edward Hutter */

#ifndef INVERSE__NEST_H_
#define INVERSE__NEST_H_

#include "./../../alg.h"
#include "./../../matmult/summa/summa.h"

namespace inverse{

class nest{
public:
  template<typename MatrixType, typename CommType>
  static void invoke(MatrixType& matrix, CommType&& CommInfo);
protected:
};
}

#include "nest.hpp"

#endif /* INVERSE__NEST_H_ */
