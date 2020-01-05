/* Author: Edward Hutter */

#ifndef QR__CAQR_H_
#define QR__CAQR_H_

#include "./../../alg.h"
#include "./../../matmult/summa/summa.h"
#include "./../policies/caqr/policy.h"

namespace qr{

template<class PipelinePolicy      = policy::caqr::NoPipelining,
         class SerializePolicy     = policy::caqr::Serialize,
         class IntermediatesPolicy = policy::caqr::SaveIntermediates>
class caqr : public PipelinePolicy, public SerializePolicy, public IntermediatesPolicy{
public:
  // caqr is parameterized only by its cholesky-inverse factorization algorithm
  template<typename ScalarType, typename DimensionType>
  class info{
  public:
    using ScalarType = ScalarType;
    using DimensionType = DimensionType;
    using alg_type = caqr<PipelinePolicy,SerializePolicy,IntermediatesPolicy>;
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

  template<typename ArgType, typename CommType>
  static matrix<typename ArgType::ScalarType,typename ArgType::DimensionType,rect> construct_Q(ArgType& args, CommType&& CommInfo);

  template<typename ArgType, typename CommType>
  static matrix<typename ArgType::ScalarType,typename ArgType::DimensionType,rect> construct_R(ArgType& args, CommType&& CommInfo);

  template<typename MatrixType, typename ArgType, typename CommType>
  static void apply_Q(MatrixType& src, ArgType& args,CommType&& CommInfo);

  template<typename MatrixType, typename ArgType, typename CommType>
  static void apply_QT(MatrixType& src, ArgType& args,CommType&& CommInfo);

protected:
};
}

#include "caqr.hpp"

#endif /* QR__CAQR_H_ */
