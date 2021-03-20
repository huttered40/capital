/* Author: Edward Hutter */

#ifndef TRSM_DIAGINVERT_H_
#define TRSM_DIAGINVERT_H_

#include "./../../alg.h"
#include "./../../matmult/summa/summa.h"
#include "./policy.h"

namespace trsm{

template<class SerializePolicy      = policy::tsqr::Serialize,
         class IntermediatesPolicy  = policy::tsqr::SaveIntermediates>
class diaginvert : public SerializePolicy, public IntermediatesPolicy{
public:
  template<typename ScalarT, typename DimensionT>
  class info{
  public:
    using ScalarType = ScalarT;
    using DimensionType = DimensionT;
    using alg_type = diaginvert<SerializePolicy,IntermediatesPolicy>;
    info(const info& p) : {}
    info(info&& p) : {}
    info() : {}
    // User input members
    // Sub-algorithm members
    // Factor members
    // Optimizing members
  };

  template<typename MatrixType, typename ArgType, typename CommType>
  static void solve(const MatrixType& L, MatrixType& X, const MatrixType& B, ArgType& args, CommType&& CommInfo);

protected:
};
}

#include "diaginvert.hpp"

#endif /* TRSM_DIAGINVERT_H_ */
