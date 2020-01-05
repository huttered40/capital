/* Author: Edward Hutter */

#ifndef QR__CACQR_H_
#define QR__CACQR_H_

#include "./../../alg.h"
#include "./../../matmult/summa/summa.h"
#include "./../../cholesky/cholinv/cholinv.h"
#include "./../policies/cacqr/policy.h"

namespace qr{

template<class SerializePolicy     = policy::cacqr::Serialize,
         class IntermediatesPolicy = policy::cacqr::SaveIntermediates>
class cacqr : public SerializePolicy, public IntermediatesPolicy{
public:
  // cacqr is parameterized only by its cholesky-inverse factorization algorithm
  template<typename ScalarType, typename DimensionType, typename CholeskyInversionType>
  class info{
  public:
    using ScalarType = ScalarType;
    using DimensionType = DimensionType;
    using alg_type = cacqr<SerializePolicy,IntermediatesPolicy>;
    using cholesky_inverse_type = CholeskyInversionType;
    info(const info& p) : num_iter(p.num_iter),cholesky_inverse_args(p.cholesky_inverse_args),Q(p.Q),R(p.R) {}
    info(info&& p) : cholesky_inverse_args(std::move(p.cholesky_inverse_args)) {}
    template<typename CholeskyInversionArgType>
    info(size_t num_iter, CholeskyInversionArgType&& ci_args) : num_iter(num_iter),cholesky_inverse_args(std::forward<CholeskyInversionArgType>(ci_args)) {}
    // User input members
    const size_t num_iter;
    // Sub-algorithm members
    typename CholeskyInversionType::info<ScalarType,DimensionType> cholesky_inverse_args;
    // Factor members
    matrix<ScalarType,DimensionType,rect> Q;
    matrix<ScalarType,DimensionType,typename SerializePolicy::structure> R;
    // Optimizing members
    std::map<std::pair<DimensionType,DimensionType>,matrix<ScalarType,DimensionType,typename SerializePolicy::structure>> policy_table;
    std::map<std::pair<DimensionType,DimensionType>,matrix<ScalarType,DimensionType,rect>> rect_table1;
    std::map<std::pair<DimensionType,DimensionType>,matrix<ScalarType,DimensionType,rect>> rect_table2;
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
  template<typename ArgType, typename CommType>
  static void invoke_1d(ArgType& args, CommType&& CommInfo);

  template<typename ArgType, typename CommType>
  static void invoke_3d(ArgType& args, CommType&& CommInfo);

  template<typename ArgType, typename CommType>
  static void sweep_1d(ArgType& args, CommType&& CommInfo);

  template<typename ArgType, typename CommType>
  static void sweep_3d(ArgType& args, CommType&& CommInfo);

  template<typename ArgType, typename RectCommType, typename SquareCommType>
  static void sweep_tune(ArgType& args, RectCommType&& RectCommInfo, SquareCommType&& SquareCommInfo);

  template<typename ArgType, typename CommType>
  static void solve(ArgType& args, CommType&& CommInfo);

  template<typename ArgType, typename CommType>
  static void simulate_solve(ArgType& args, CommType&& CommInfo);
};
}

#include "cacqr.hpp"

#endif /* QR__CACQR_H_ */
