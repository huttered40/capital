/* Author: Edward Hutter */

#ifndef QR__CACQR_H_
#define QR__CACQR_H_

#include "./../../alg.h"
#include "./../../matmult/summa/summa.h"
#include "./../../cholesky/cholinv/cholinv.h"

namespace qr{

class cacqr{
public:
  // cacqr is parameterized only by its cholesky-inverse factorization algorithm
  template<typename CholeskyInversionType>
  class pack{
  public:
    using cholesky_inverse_type = CholeskyInversionType;
    pack(const pack& p) : cholesky_inverse_pack(p.cholesky_inverse_pack) {}
    pack(pack&& p) : cholesky_inverse_pack(std::move(p.cholesky_inverse_pack)) {}
    template<typename CholeskyInversionArgType>
    pack(CholeskyInversionArgType&& ci_args) : cholesky_inverse_args(std::forward<CholeskyInversionArgType>(ci_args)) {}
    typename CholeskyInversionType::pack cholesky_inverse_args;
    // cacqr takes no local parameters
  };

  template<typename MatrixAType, typename MatrixRType, typename ArgType, typename CommType>
  static void invoke(MatrixAType& A, MatrixRType& R, ArgType&& args, CommType&& CommInfo);

protected:
  // Special overload to avoid recreating MPI communicator topologies
  template<typename MatrixAType, typename MatrixRType, typename ArgType, typename RectCommType, typename SquareCommType>
  static void invoke(MatrixAType& A, MatrixRType& R, MatrixRType& RI, ArgType&& args, RectCommType&& RectCommInfo, SquareCommType&& SquareCommInfo);

  template<typename MatrixAType, typename MatrixRType, typename ArgType, typename CommType>
  static void invoke_1d(MatrixAType& A, MatrixRType& R, typename MatrixRType::ScalarType* RI, ArgType&& args, CommType&& CommInfo);

  template<typename MatrixAType, typename MatrixRType, typename ArgType, typename CommType>
  static void invoke_3d(MatrixAType& A, MatrixRType& R, MatrixRType& RI, ArgType&& args, CommType&& CommInfo);

  template<typename MatrixAType, typename MatrixUType, typename MatrixUIType, typename CommType>
  static void solve_upper_left(MatrixAType& matrixA, MatrixUType& matrixU, MatrixUIType& matrixUI, CommType&& CommInfo,
                               std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                               blas::ArgPack_gemm<typename MatrixAType::ScalarType>& gemmPackage);
};

class cacqr2 : public cacqr{
public:
  // cacqr2 is parameterized only by its cholesky-inverse factorization algorithm
  template<typename CholeskyInversionType>
  class pack{
  public:
    using cholesky_inverse_type = CholeskyInversionType;
    pack(const pack& p) : cholesky_inverse_pack = p.cholesky_inverse_pack {}
    pack(pack&& p) : cholesky_inverse_pack = std::move(p.cholesky_inverse_pack) {}
    template<typename CholeskyInversionArgType>
    pack(CholeskyInversionArgType&& ci_args) : cholesky_inverse_args(std::forward<CholeskyInversionArgType>(ci_args)) {}
    typename CholeskyInversionType::pack cholesky_inverse_args;
    // cacqr2 takes no local parameters
  };

  template<typename MatrixAType, typename MatrixRType, typename ArgType, typename CommType>
  static void invoke(MatrixAType& A, MatrixRType& R, ArgType&& args, CommType&& CommInfo);

protected:
  template<typename MatrixAType, typename MatrixRType, typename ArgType, typename CommType>
  static void invoke_1d(MatrixAType& A, MatrixRType& R, ArgType&& args, CommType&& CommInfo);

  template<typename MatrixAType, typename MatrixRType, typename ArgType, typename CommType>
  static void invoke_3d(MatrixAType& A, MatrixRType& R, ArgType&& args, CommType&& CommInfo);
};
}

#include "cacqr.hpp"

#endif /* QR__CACQR_H_ */
