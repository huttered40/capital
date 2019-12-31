/* Author: Edward Hutter */

#ifndef CHOLESKY__CHOLINV_H_
#define CHOLESKY__CHOLINV_H_

#include "./../../alg.h"
#include "./../../matmult/summa/summa.h"
#include "./../policies/cholinv/policy.h"

namespace cholesky{
template<class SerializePolicy = policy::cholinv::SerializeAvoidComm,
         class OverlapRecursivePolicy = policy::cholinv::NoIntermediateOverlap>
class cholinv{
public:
  // cholinv is not parameterized as its not dependent on any lower-level algorithmic type
  class pack{
  public:
    using alg_type = cholinv<SerializePolicy,OverlapRecursivePolicy>;
    pack(const pack& p) : inv_cut_off_dim(p.inv_cut_off_dim), bc_mult_dim(p.bc_mult_dim), dir(p.dir) {}
    pack(pack&& p) : inv_cut_off_dim(p.inv_cut_off_dim), bc_mult_dim(p.bc_mult_dim), dir(p.dir) {}
    pack(int64_t inv_cut_off_dim, int64_t bc_mult_dim, char dir) : inv_cut_off_dim(inv_cut_off_dim), bc_mult_dim(bc_mult_dim), dir(dir) {}
    int64_t inv_cut_off_dim;
    int64_t bc_mult_dim;
    char dir;
  };

  template<typename MatrixAType, typename MatrixTIType, typename ArgType, typename CommType>
  static std::pair<bool,std::vector<typename MatrixAType::DimensionType>>
         invoke(MatrixAType& A, MatrixTIType& TI, ArgType&& args, CommType&& CommInfo);

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
                     typename MatrixAType::DimensionType RIstartY, typename MatrixAType::DimensionType RIendY,
                     CommType&& CommInfo, bool& isInversePath, std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                     typename MatrixAType::DimensionType inverseCutoffGlobalDimension);

  template<typename MatrixAType, typename MatrixIType, typename BaseCaseTableType, typename BaseCaseBlockedTableType, typename BaseCaseCyclicTableType, typename CommType>
  static void base_case(MatrixAType& A, MatrixIType& I, BaseCaseTableType& base_case_table, BaseCaseBlockedTableType& base_case_blocked_table, BaseCaseCyclicTableType& base_case_cyclic_table,
                 typename MatrixAType::DimensionType localDimension, typename MatrixAType::DimensionType trueLocalDimension,
                 typename MatrixAType::DimensionType bcDimension, typename MatrixAType::DimensionType globalDimension, typename MatrixAType::DimensionType trueGlobalDimension,
                 typename MatrixAType::DimensionType AstartX, typename MatrixAType::DimensionType AendX, typename MatrixAType::DimensionType AstartY,
                 typename MatrixAType::DimensionType AendY, typename MatrixAType::DimensionType matIstartX, typename MatrixAType::DimensionType matIendX,
                 typename MatrixAType::DimensionType matIstartY, typename MatrixAType::DimensionType matIendY,
                 CommType&& CommInfo, bool& isInversePath, std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                 typename MatrixAType::DimensionType inverseCutoffGlobalDimension, char dir);

  template<typename PolicyTableType, typename SquareTableType, typename BaseCaseTableType, typename BaseCaseBlockedTableType, typename BaseCaseCyclicTableType, typename CommType>
  static void simulate(PolicyTableType& policy_table, PolicyTableType& policy_table_diaginvert, SquareTableType& square_table1, SquareTableType& square_table2,
                       BaseCaseTableType& base_case_table, BaseCaseBlockedTableType& base_case_blocked_table, BaseCaseCyclicTableType& base_case_cyclic_table,
                       int64_t localDimension, int64_t trueLocalDimension, int64_t bcDimension, int64_t globalDimension, int64_t trueGlobalDimension,
                       int64_t AstartX, int64_t AendX, int64_t AstartY, int64_t AendY, int64_t RIstartX, int64_t RIendX, int64_t RIstartY, int64_t RIendY,
                       CommType&& CommInfo, bool& isInversePath, std::vector<int64_t>& baseCaseDimList, int64_t inverseCutoffGlobalDimension);

  template<typename BaseCaseTableType, typename BaseCaseBlockedTableType, typename BaseCaseCyclicTableType, typename CommType>
  static void simulate_basecase(BaseCaseTableType& base_case_table, BaseCaseBlockedTableType& base_case_blocked_table, BaseCaseCyclicTableType& base_case_cyclic_table,
                                int64_t localDimension, int64_t trueLocalDimension, int64_t bcDimension, int64_t globalDimension, int64_t trueGlobalDimension,
                                int64_t AstartX, int64_t AendX, int64_t AstartY, int64_t AendY, int64_t matIstartX, int64_t matIendX, int64_t matIstartY, int64_t matIendY,
                                CommType&& CommInfo, bool& isInversePath, std::vector<int64_t>& baseCaseDimList, int64_t inverseCutoffGlobalDimension, char dir);

  template<typename SquareTableType, typename CommType>
  static void simulate_solve(SquareTableType& square_table1, SquareTableType& square_table2, SquareTableType& square_table3, CommType&& CommInfo, int64_t num_cols_A,
                             int64_t num_rows_A, int64_t num_cols_L, std::vector<int64_t>& baseCaseDimList);

  template<typename MatrixLType, typename MatrixLIType, typename MatrixAType, typename SquareTableType, typename CommType>
  static void solve(MatrixLType& L, MatrixLIType& LI, MatrixAType& A, SquareTableType& square_table1, SquareTableType& square_table2, SquareTableType& square_table3, CommType&& CommInfo,
                    std::vector<typename MatrixAType::DimensionType>& baseCaseDimList, blas::ArgPack_gemm<typename MatrixAType::ScalarType>& gemmPackage);

  template<typename T, typename U>
  static void cyclic_to_local(T* storeT, T* storeTI, U localDimension, U globalDimension, U bcDimension, int64_t sliceDim, int64_t rankSlice, char dir);

  template<typename U>
  static inline void update_inverse_path(U inverseCutoffGlobalDimension, U globalDimension, bool& isInversePath, std::vector<U>& baseCaseDimList, U localDimension);

  template<typename U>
  static inline void update_inverse_path_simulate(U inverseCutoffGlobalDimension, U globalDimension, bool& isInversePath, U localDimension);
};
}

#include "cholinv.hpp"

#endif /* CHOLESKY__CHOLINV_H_ */
