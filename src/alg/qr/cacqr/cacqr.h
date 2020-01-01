/* Author: Edward Hutter */

#ifndef QR__CACQR_H_
#define QR__CACQR_H_

#include "./../../alg.h"
#include "./../../matmult/summa/summa.h"
#include "./../../cholesky/cholinv/cholinv.h"
#include "./../policies/cacqr/policy.h"
#include "./../../cholesky/policies/cholinv/policy.h"

namespace qr{

template<class SerializePolicy = policy::cacqr::Serialize,
         class IntermediatesPolicy = policy::cacqr::SaveIntermediates>
class cacqr{
public:
  // cacqr is parameterized only by its cholesky-inverse factorization algorithm
  template<typename ScalarType, typename DimensionType, typename CholeskyInversionType>
  class pack{
  public:
    using alg_type = cacqr<SerializePolicy>;
    using cholesky_inverse_type = CholeskyInversionType;
    pack(const pack& p) : cholesky_inverse_args(p.cholesky_inverse_args) {}
    pack(pack&& p) : cholesky_inverse_args(std::move(p.cholesky_inverse_args)) {}
    template<typename CholeskyInversionArgType>
    pack(CholeskyInversionArgType&& ci_args) : cholesky_inverse_args(std::forward<CholeskyInversionArgType>(ci_args)) {}
    typename CholeskyInversionType::pack<ScalarType,DimensionType> cholesky_inverse_args;
    std::map<std::pair<DimensionType,DimensionType>,matrix<ScalarType,DimensionType,typename SerializePolicy::structure>> policy_table1;
    std::map<std::pair<DimensionType,DimensionType>,matrix<ScalarType,DimensionType,typename SerializePolicy::structure>> policy_table2;
    std::map<std::pair<DimensionType,DimensionType>,matrix<ScalarType,DimensionType,rect>> square_table1;
    std::map<std::pair<DimensionType,DimensionType>,matrix<ScalarType,DimensionType,rect>> square_table2;
  };

  template<typename MatrixType, typename ArgType, typename CommType>
  static void invoke(MatrixType& A, MatrixType& R, ArgType&& args, CommType&& CommInfo);

  template<typename ScalarType, typename DimensionType, typename ArgType, typename CommType>
  static std::pair<ScalarType*,ScalarType*> invoke(ScalarType* A, ScalarType* R, DimensionType localNumRows, DimensionType localNumColumns, DimensionType globalNumRows, DimensionType globalNumColumns, ArgType&& args, CommType&& CommInfo);

protected:
  // Special overload to avoid recreating MPI communicator topologies
  template<typename MatrixType, typename ArgType, typename RectCommType, typename SquareCommType>
  static void invoke(MatrixType& A, MatrixType& R, MatrixType& RI, ArgType&& args, RectCommType&& RectCommInfo, SquareCommType&& SquareCommInfo);

  template<typename MatrixType, typename ArgType, typename CommType>
  static void invoke_1d(MatrixType& A, MatrixType& R, typename MatrixType::ScalarType* RI, ArgType&& args, CommType&& CommInfo);

  template<typename MatrixType, typename ArgType, typename CommType>
  static void invoke_3d(MatrixType& A, MatrixType& R, MatrixType& RI, ArgType&& args, CommType&& CommInfo);
};

template<class SerializePolicy = policy::cacqr::Serialize,
         class IntermediatesPolicy = policy::cacqr::SaveIntermediates>
class cacqr2 : public cacqr<SerializePolicy,IntermediatesPolicy>{
public:
  // cacqr2 is parameterized only by its cholesky-inverse factorization algorithm
  template<typename ScalarType, typename DimensionType, typename CholeskyInversionType>
  class pack{
  public:
    using alg_type = cacqr<SerializePolicy>;
    using cholesky_inverse_type = CholeskyInversionType;
    pack(const pack& p) : cholesky_inverse_args(p.cholesky_inverse_args) {}
    pack(pack&& p) : cholesky_inverse_args(std::move(p.cholesky_inverse_args)) {}
    template<typename CholeskyInversionArgType>
    pack(CholeskyInversionArgType&& ci_args) : cholesky_inverse_args(std::forward<CholeskyInversionArgType>(ci_args)) {}
    typename CholeskyInversionType::pack<ScalarType,DimensionType> cholesky_inverse_args;
    std::map<std::pair<DimensionType,DimensionType>,matrix<ScalarType,DimensionType,typename SerializePolicy::structure>> policy_table1;
    std::map<std::pair<DimensionType,DimensionType>,matrix<ScalarType,DimensionType,typename SerializePolicy::structure>> policy_table2;
    std::map<std::pair<DimensionType,DimensionType>,matrix<ScalarType,DimensionType,rect>> square_table1;
    std::map<std::pair<DimensionType,DimensionType>,matrix<ScalarType,DimensionType,rect>> square_table2;
  };

  template<typename MatrixType, typename ArgType, typename CommType>
  static void invoke(MatrixType& A, MatrixType& R, ArgType&& args, CommType&& CommInfo);

  template<typename ScalarType, typename DimensionType, typename ArgType, typename CommType>
  static std::pair<ScalarType*,ScalarType*> invoke(ScalarType* A, ScalarType* R, DimensionType localNumRows, DimensionType localNumColumns, DimensionType globalNumRows, DimensionType globalNumColumns, ArgType&& args, CommType&& CommInfo);

protected:
  template<typename MatrixType, typename ArgType, typename CommType>
  static void invoke_1d(MatrixType& A, MatrixType& R, ArgType&& args, CommType&& CommInfo);

  template<typename MatrixType, typename ArgType, typename CommType>
  static void invoke_3d(MatrixType& A, MatrixType& R, ArgType&& args, CommType&& CommInfo);
};
}

#include "cacqr.hpp"

#endif /* QR__CACQR_H_ */
