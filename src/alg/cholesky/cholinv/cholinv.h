/* Author: Edward Hutter */

#ifndef CHOLESKY__CHOLINV_H_
#define CHOLESKY__CHOLINV_H_

#include "./../../alg.h"
#include "./../../matmult/summa/summa.h"
#include "./../policies/cholinv/policy.h"

namespace cholesky{
template<class SerializePolicy     = policy::cholinv::Serialize,
         class IntermediatesPolicy = policy::cholinv::SaveIntermediates,
         class OverlapPolicy       = policy::cholinv::NoOverlap>
class cholinv{
public:
  template<typename T, typename U>
  class pack{
  public:
    using alg_type = cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>;
    pack(const pack& p) : complete_inv(p.complete_inv), bc_mult_dim(p.bc_mult_dim), dir(p.dir) {}
    pack(pack&& p) : complete_inv(p.complete_inv), bc_mult_dim(p.bc_mult_dim), dir(p.dir) {}
    pack(int64_t complete_inv, int64_t bc_mult_dim, char dir) : complete_inv(complete_inv), bc_mult_dim(bc_mult_dim), dir(dir) {}
    const int64_t complete_inv;
    const int64_t bc_mult_dim;
    const char dir;
    std::map<std::pair<U,U>,matrix<T,U,typename SerializePolicy::structure>> policy_table;
    std::map<std::pair<U,U>,matrix<T,U,typename SerializePolicy::structure>> policy_table_diaginv;
    std::map<std::pair<U,U>,matrix<T,U,rect>> square_table1;
    std::map<std::pair<U,U>,matrix<T,U,rect>> square_table2;
    std::map<std::pair<U,U>,matrix<T,U,typename SerializePolicy::structure>> base_case_table;
    std::map<std::pair<U,U>,std::vector<T>> base_case_blocked_table;
    std::map<std::pair<U,U>,matrix<T,U,rect>> base_case_cyclic_table;
  };

  template<typename MatrixType, typename ArgType, typename CommType>
  static void invoke(MatrixType& A, MatrixType& TI, ArgType&& args, CommType&& CommInfo);

  template<typename T, typename U, typename ArgType, typename CommType>
  static std::pair<T*,T*> invoke(T* A, T* TI, U localDim, U globalDim, ArgType&& args, CommType&& CommInfo);

private:
  template<typename MatrixType, typename ArgType, typename CommType>
  static void factor(MatrixType& A, MatrixType& RI, ArgType&& args, typename MatrixType::DimensionType localDimension, typename MatrixType::DimensionType trueLocalDimension,
                     typename MatrixType::DimensionType bcDimension, typename MatrixType::DimensionType globalDimension, typename MatrixType::DimensionType trueGlobalDimension,
                     typename MatrixType::DimensionType AstartX, typename MatrixType::DimensionType AendX, typename MatrixType::DimensionType AstartY,
                     typename MatrixType::DimensionType AendY, typename MatrixType::DimensionType RIstartX, typename MatrixType::DimensionType RIendX,
                     typename MatrixType::DimensionType RIstartY, typename MatrixType::DimensionType RIendY, bool complete_inv, CommType&& CommInfo);

  template<typename MatrixType, typename ArgType, typename CommType>
  static void base_case(MatrixType& A, MatrixType& I, ArgType&& args, typename MatrixType::DimensionType localDimension, typename MatrixType::DimensionType trueLocalDimension,
                        typename MatrixType::DimensionType bcDimension, typename MatrixType::DimensionType globalDimension, typename MatrixType::DimensionType trueGlobalDimension,
                        typename MatrixType::DimensionType AstartX, typename MatrixType::DimensionType AendX, typename MatrixType::DimensionType AstartY,
                        typename MatrixType::DimensionType AendY, typename MatrixType::DimensionType matIstartX, typename MatrixType::DimensionType matIendX,
                        typename MatrixType::DimensionType matIstartY, typename MatrixType::DimensionType matIendY, CommType&& CommInfo);

  template<typename ArgType, typename CommType>
  static void simulate(ArgType&& args, int64_t localDimension, int64_t trueLocalDimension, int64_t bcDimension, int64_t globalDimension, int64_t trueGlobalDimension,
                       int64_t AstartX, int64_t AendX, int64_t AstartY, int64_t AendY, int64_t RIstartX, int64_t RIendX, int64_t RIstartY, int64_t RIendY, bool complete_inv, CommType&& CommInfo);

  template<typename ArgType, typename CommType>
  static void simulate_basecase(ArgType&& args, int64_t localDimension, int64_t trueLocalDimension, int64_t bcDimension, int64_t globalDimension, int64_t trueGlobalDimension,
                                int64_t AstartX, int64_t AendX, int64_t AstartY, int64_t AendY, int64_t matIstartX, int64_t matIendX, int64_t matIstartY, int64_t matIendY, CommType&& CommInfo);
};
}

#include "cholinv.hpp"

#endif /* CHOLESKY__CHOLINV_H_ */
