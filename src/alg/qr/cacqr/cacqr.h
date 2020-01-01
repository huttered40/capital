/* Author: Edward Hutter */

#ifndef QR__CACQR_H_
#define QR__CACQR_H_

#include "./../../alg.h"
#include "./../../matmult/summa/summa.h"
#include "./../../cholesky/cholinv/cholinv.h"
#include "./../policies/cacqr/policy.h"
#include "./../../cholesky/policies/cholinv/policy.h"

namespace qr{

template<class SerializePolicy = policy::cacqr::Serialize>
class cacqr{
public:
  // cacqr is parameterized only by its cholesky-inverse factorization algorithm
  template<typename T, typename U, typename CholeskyInversionType>
  class pack{
  public:
    using alg_type = cacqr<SerializePolicy>;
    using cholesky_inverse_type = CholeskyInversionType;
    pack(const pack& p) : cholesky_inverse_args(p.cholesky_inverse_args) {}
    pack(pack&& p) : cholesky_inverse_args(std::move(p.cholesky_inverse_args)) {}
    template<typename CholeskyInversionArgType>
    pack(CholeskyInversionArgType&& ci_args) : cholesky_inverse_args(std::forward<CholeskyInversionArgType>(ci_args)) {}
    typename CholeskyInversionType::pack<T,U> cholesky_inverse_args;
    // cacqr takes no local parameters
  };

  template<typename MatrixAType, typename MatrixRType, typename ArgType, typename CommType>
  static void invoke(MatrixAType& A, MatrixRType& R, ArgType&& args, CommType&& CommInfo);

  template<typename T, typename U, typename ArgType, typename CommType>
  static std::pair<T*,T*> invoke(T* A, T* R, U localNumRows, U localNumColumns, U globalNumRows, U globalNumColumns, ArgType&& args, CommType&& CommInfo);

protected:
  // Special overload to avoid recreating MPI communicator topologies
  template<typename MatrixAType, typename MatrixRType, typename ArgType, typename RectCommType, typename SquareCommType>
  static void invoke(MatrixAType& A, MatrixRType& R, MatrixRType& RI, ArgType&& args, RectCommType&& RectCommInfo, SquareCommType&& SquareCommInfo);

  template<typename MatrixAType, typename MatrixRType, typename ArgType, typename CommType>
  static void invoke_1d(MatrixAType& A, MatrixRType& R, typename MatrixRType::ScalarType* RI, ArgType&& args, CommType&& CommInfo);

  template<typename MatrixAType, typename MatrixRType, typename ArgType, typename CommType>
  static void invoke_3d(MatrixAType& A, MatrixRType& R, MatrixRType& RI, ArgType&& args, CommType&& CommInfo);

  template<typename MatrixAType, typename MatrixUType, typename MatrixUIType, typename CommType>
  static void solve(MatrixAType& A, MatrixUType& U, MatrixUIType& UI, CommType&& CommInfo, blas::ArgPack_gemm<typename MatrixAType::ScalarType>& gemmPackage);
};

template<class SerializePolicy = policy::cacqr::Serialize>
class cacqr2 : public cacqr<SerializePolicy>{
public:
  // cacqr2 is parameterized only by its cholesky-inverse factorization algorithm
  template<typename T, typename U, typename CholeskyInversionType>
  class pack{
  public:
    using alg_type = cacqr<SerializePolicy>;
    using cholesky_inverse_type = CholeskyInversionType;
    pack(const pack& p) : cholesky_inverse_args(p.cholesky_inverse_args) {}
    pack(pack&& p) : cholesky_inverse_args(std::move(p.cholesky_inverse_args)) {}
    template<typename CholeskyInversionArgType>
    pack(CholeskyInversionArgType&& ci_args) : cholesky_inverse_args(std::forward<CholeskyInversionArgType>(ci_args)) {}
    typename CholeskyInversionType::pack<T,U> cholesky_inverse_args;
    // cacqr2 takes no local parameters
  };

  template<typename MatrixAType, typename MatrixRType, typename ArgType, typename CommType>
  static void invoke(MatrixAType& A, MatrixRType& R, ArgType&& args, CommType&& CommInfo);

  template<typename T, typename U, typename ArgType, typename CommType>
  static std::pair<T*,T*> invoke(T* A, T* R, U localNumRows, U localNumColumns, U globalNumRows, U globalNumColumns, ArgType&& args, CommType&& CommInfo);

protected:
  template<typename MatrixAType, typename MatrixRType, typename ArgType, typename CommType>
  static void invoke_1d(MatrixAType& A, MatrixRType& R, ArgType&& args, CommType&& CommInfo);

  template<typename MatrixAType, typename MatrixRType, typename ArgType, typename CommType>
  static void invoke_3d(MatrixAType& A, MatrixRType& R, ArgType&& args, CommType&& CommInfo);
};
}

#include "cacqr.hpp"

#endif /* QR__CACQR_H_ */
