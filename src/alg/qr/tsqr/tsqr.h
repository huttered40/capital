/* Author: Edward Hutter */

#ifndef QR__TSQR_H_
#define QR__TSQR_H_

#include "./../../alg.h"
#include "./../../matmult/summa/summa.h"
#include "./policy.h"

namespace qr{

template<class TraversalPolicy      = policy::tsqr::TreeSweep,
         class RepresentationPolicy = policy::tsqr::Tree,
         class SerializePolicy      = policy::tsqr::Serialize,
         class IntermediatesPolicy  = policy::tsqr::SaveIntermediates>
class tsqr : public TraversalPolicy, public RepresentationPolicy, public SerializePolicy, public IntermediatesPolicy{
public:
  template<typename ScalarType, typename DimensionType>
  class info{
  public:
    using ScalarType = ScalarType;
    using DimensionType = DimensionType;
    using alg_type = tsqr<TraversalPolicy,RepresentationPolicy,SerializePolicy,IntermediatesPolicy>;
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

#include "tsqr.hpp"

#endif /* QR__TSQR_H_ */
