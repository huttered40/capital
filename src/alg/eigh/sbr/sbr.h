/* Author: Edward Hutter */

#ifndef EIGH__SBR_H_
#define EIGH__SBR_H_

#include "./../../alg.h"
#include "./../../matmult/summa/summa.h"
#include "./policy.h"

namespace eigh{

template<class SerializePolicy     = policy::caqr::Serialize,
         class IntermediatesPolicy = policy::caqr::SaveIntermediates>
class sbr : public SerializePolicy, public IntermediatesPolicy{
public:
  template<typename ScalarType, typename DimensionType>
  class info{
  public:
    using ScalarType = ScalarType;
    using DimensionType = DimensionType;
    using alg_type = sbr<SerializePolicy,IntermediatesPolicy>;
    info(const info& p) : {}
    info(info&& p) : {}
    info() : {}
    // User input members
    // Sub-algorithm members
    // Factor members
    // Optimizing members
  };

  template<typename MatrixType, typename ArgType, typename CommType>
  static void factor(const MatrixType& A, ArgType& args, CommType&& CommInfo);

protected:
};
}

#include "sbr.hpp"

#endif /* EIGH__SBR_H_ */
