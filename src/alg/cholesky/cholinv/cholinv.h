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
    int64_t complete_inv;
    int64_t bc_mult_dim;
    char dir;
    std::map<std::pair<U,U>,matrix<T,U,typename SerializePolicy::structure>> policy_table;
    std::map<std::pair<U,U>,matrix<T,U,typename SerializePolicy::structure>> policy_table_diaginv;
    std::map<std::pair<U,U>,matrix<T,U,rect>> square_table1;
    std::map<std::pair<U,U>,matrix<T,U,rect>> square_table2;
    std::map<std::pair<U,U>,matrix<T,U,typename SerializePolicy::structure>> base_case_table;
    std::map<std::pair<U,U>,std::vector<T>> base_case_blocked_table;
    std::map<std::pair<U,U>,matrix<T,U,rect>> base_case_cyclic_table;
  };

  template<typename MatrixAType, typename MatrixTIType, typename ArgType, typename CommType>
  static void invoke(MatrixAType& A, MatrixTIType& TI, ArgType&& args, CommType&& CommInfo);

  template<typename T, typename U, typename ArgType, typename CommType>
  static std::pair<T*,T*> invoke(T* A, T* TI, U localDim, U globalDim, ArgType&& args, CommType&& CommInfo);

private:
  template<typename MatrixAType, typename MatrixRIType, typename PolicyTableType, typename SquareTableType, typename BaseCaseTableType, typename BaseCaseBlockedTableType, typename BaseCaseCyclicTableType, typename CommType>
  static void factor(MatrixAType& A, MatrixRIType& RI, PolicyTableType& policy_table, PolicyTableType& policy_table_diaginvert,
                     SquareTableType& square_table1, SquareTableType& square_table2, BaseCaseTableType& base_case_table, BaseCaseBlockedTableType& base_case_blocked_table, BaseCaseCyclicTableType& base_case_cyclic_table,
                     typename MatrixAType::DimensionType localDimension, typename MatrixAType::DimensionType trueLocalDimension,
                     typename MatrixAType::DimensionType bcDimension, typename MatrixAType::DimensionType globalDimension, typename MatrixAType::DimensionType trueGlobalDimension,
                     typename MatrixAType::DimensionType AstartX, typename MatrixAType::DimensionType AendX, typename MatrixAType::DimensionType AstartY,
                     typename MatrixAType::DimensionType AendY, typename MatrixAType::DimensionType RIstartX, typename MatrixAType::DimensionType RIendX,
                     typename MatrixAType::DimensionType RIstartY, typename MatrixAType::DimensionType RIendY, bool complete_inv, CommType&& CommInfo);

  template<typename MatrixAType, typename MatrixIType, typename BaseCaseTableType, typename BaseCaseBlockedTableType, typename BaseCaseCyclicTableType, typename CommType>
  static void base_case(MatrixAType& A, MatrixIType& I, BaseCaseTableType& base_case_table, BaseCaseBlockedTableType& base_case_blocked_table, BaseCaseCyclicTableType& base_case_cyclic_table,
                 typename MatrixAType::DimensionType localDimension, typename MatrixAType::DimensionType trueLocalDimension,
                 typename MatrixAType::DimensionType bcDimension, typename MatrixAType::DimensionType globalDimension, typename MatrixAType::DimensionType trueGlobalDimension,
                 typename MatrixAType::DimensionType AstartX, typename MatrixAType::DimensionType AendX, typename MatrixAType::DimensionType AstartY,
                 typename MatrixAType::DimensionType AendY, typename MatrixAType::DimensionType matIstartX, typename MatrixAType::DimensionType matIendX,
                 typename MatrixAType::DimensionType matIstartY, typename MatrixAType::DimensionType matIendY, CommType&& CommInfo);

  template<typename PolicyTableType, typename SquareTableType, typename BaseCaseTableType, typename BaseCaseBlockedTableType, typename BaseCaseCyclicTableType, typename CommType>
  static void simulate(PolicyTableType& policy_table, PolicyTableType& policy_table_diaginvert, SquareTableType& square_table1, SquareTableType& square_table2,
                       BaseCaseTableType& base_case_table, BaseCaseBlockedTableType& base_case_blocked_table, BaseCaseCyclicTableType& base_case_cyclic_table,
                       int64_t localDimension, int64_t trueLocalDimension, int64_t bcDimension, int64_t globalDimension, int64_t trueGlobalDimension,
                       int64_t AstartX, int64_t AendX, int64_t AstartY, int64_t AendY, int64_t RIstartX, int64_t RIendX, int64_t RIstartY, int64_t RIendY, bool complete_inv, CommType&& CommInfo);

  template<typename BaseCaseTableType, typename BaseCaseBlockedTableType, typename BaseCaseCyclicTableType, typename CommType>
  static void simulate_basecase(BaseCaseTableType& base_case_table, BaseCaseBlockedTableType& base_case_blocked_table, BaseCaseCyclicTableType& base_case_cyclic_table,
                                int64_t localDimension, int64_t trueLocalDimension, int64_t bcDimension, int64_t globalDimension, int64_t trueGlobalDimension,
                                int64_t AstartX, int64_t AendX, int64_t AstartY, int64_t AendY, int64_t matIstartX, int64_t matIendX, int64_t matIstartY, int64_t matIendY, CommType&& CommInfo);

  template<typename T, typename U>
  static void cyclic_to_local(T* storeT, T* storeTI, U localDimension, U globalDimension, U bcDimension, int64_t sliceDim, int64_t rankSlice);
};
}

#include "cholinv.hpp"

#endif /* CHOLESKY__CHOLINV_H_ */
