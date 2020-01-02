/* Author: Edward Hutter */

#ifndef INVERSE__NEWTON_H_
#define INVERSE__NEWTON_H_

#include "./../../alg.h"
#include "./../../matmult/summa/summa.h"

namespace inverse{

class newton{
public:
  // newton is not parameterized as its not dependent on any lower-level algorithmic type
  class pack{
  public:
    pack(const pack& p) : tol = p.tol, max_iter = p.max_iter {}
    pack(pack&& p) : tol = std::move(p.tol), max_iter = std::move(p.max_iter) {}
    pack(int64_t tol, char max_iter) : tol = tol, max_iter = max_iter }
    int64_t tol;
    char max_iter;
  };

  template<typename MatrixType, typename ArgType, typename CommType>
  static void invoke(MatrixType& matrix, ArgType& args, CommType&& CommInfo);
protected:
};
}

#include "newton.hpp"

#endif /* INVERSE__NEWTON_H_ */
