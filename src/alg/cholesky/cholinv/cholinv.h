/* Author: Edward Hutter */

#ifndef CHOLESKY__CHOLINV_H_
#define CHOLESKY__CHOLINV_H_

#include "./../../alg.h"
#include "./../../matmult/summa/summa.h"
#include "./policy.h"

namespace cholesky{
template<class SerializePolicy     = policy::cholinv::Serialize,
         class IntermediatesPolicy = policy::cholinv::SaveIntermediates,
         class BaseCasePolicy      = policy::cholinv::NoReplication>
class cholinv : public SerializePolicy, public IntermediatesPolicy, public BaseCasePolicy{
public:
  template<typename ScalarT, typename DimensionT>
  class info{
  public:
    using ScalarType = ScalarT;
    using DimensionType = DimensionT;
    using alg_type = cholinv<SerializePolicy,IntermediatesPolicy,BaseCasePolicy>;
    using SP = SerializePolicy; using IP = IntermediatesPolicy; using BP = BaseCasePolicy;
    info(const info& p) : complete_inv(p.complete_inv), split(p.split), bc_mult_dim(p.bc_mult_dim), dir(p.dir) {}
    info(info&& p) : complete_inv(p.complete_inv), split(p.split), bc_mult_dim(p.bc_mult_dim), dir(p.dir) {}
    info(DimensionType complete_inv, DimensionType split, DimensionType bc_mult_dim, char dir) : complete_inv(complete_inv), split(split), bc_mult_dim(bc_mult_dim), dir(dir) {}
    // User input members
    const DimensionType complete_inv;
    const DimensionType split;
    const DimensionType bc_mult_dim;
    const char dir;
    // Factor members
    matrix<ScalarType,DimensionType,typename SerializePolicy::structure> R;
    matrix<ScalarType,DimensionType,typename SerializePolicy::structure> Rinv;
    // Optimizing members
    std::map<std::pair<DimensionType,DimensionType>,matrix<ScalarType,DimensionType,typename SerializePolicy::structure>> policy_table;
    std::map<std::pair<DimensionType,DimensionType>,matrix<ScalarType,DimensionType,rect>> rect_table1;
    std::map<std::pair<DimensionType,DimensionType>,matrix<ScalarType,DimensionType,rect>> rect_table2;
    std::map<std::pair<DimensionType,DimensionType>,matrix<ScalarType,DimensionType,typename SerializePolicy::structure>> base_case_table;
    std::map<std::pair<DimensionType,DimensionType>,std::vector<ScalarType>> base_case_blocked_table;
    std::map<std::pair<DimensionType,DimensionType>,matrix<ScalarType,DimensionType,rect>> base_case_cyclic_table;
    DimensionType localDimension,globalDimension,trueLocalDimension,trueGlobalDimension,bcDimension;
    DimensionType AstartX,AendX,AstartY,AendY,TIstartX,TIendX,TIstartY,TIendY;
    MPI_Request req;
  };

  template<typename MatrixType, typename ArgType, typename CommType>
  static void factor(const MatrixType& A, ArgType& args, CommType&& CommInfo);

  template<typename ArgType, typename CommType>
  static matrix<typename ArgType::ScalarType,typename ArgType::DimensionType,rect> construct_R(ArgType& args, CommType&& CommInfo);

  template<typename ArgType, typename CommType>
  static matrix<typename ArgType::ScalarType,typename ArgType::DimensionType,rect> construct_Rinv(ArgType& args, CommType&& CommInfo);

  using SP = SerializePolicy; using IP = IntermediatesPolicy; using BP = BaseCasePolicy;

private:
  template<typename ArgType, typename CommType>
  static void invoke(ArgType& args, CommType&& CommInfo);

  template<typename ArgType, typename CommType>
  static void base_case(ArgType& args, CommType&& CommInfo);

  template<typename ArgType, typename CommType>
  static void simulate(ArgType& args, CommType&& CommInfo);

  template<typename ArgType, typename CommType>
  static void simulate_basecase(ArgType& args, CommType&& CommInfo);
};
}

#include "cholinv.hpp"

#endif /* CHOLESKY__CHOLINV_H_ */
