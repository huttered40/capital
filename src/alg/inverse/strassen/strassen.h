/* Author: Edward Hutter */

#ifndef INVERSE__STRASSEN_H_
#define INVERSE__STRASSEN_H_

#include "./../../alg.h"
#include "./../../matmult/summa/summa.h"

namespace inverse{

class strassen{
public:
  template<typename MatrixType, typename CommType>
  static void invoke(MatrixType& matrix, CommType&& CommInfo);
protected:
};
}

#include "strassen.hpp"

#endif /* INVERSE__STRASSEN_H_ */
