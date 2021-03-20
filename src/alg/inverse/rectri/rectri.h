/* Author: Edward Hutter */

#ifndef INVERSE__RECTRI_H_
#define INVERSE__RECTRI_H_

#include "./../../alg.h"
#include "./../../matmult/summa/summa.h"
#include "./policy.h"

namespace inverse{
template<class SerializePolicy     = policy::rectri::Serialize,
         class IntermediatesPolicy = policy::rectri::SaveIntermediates>
class rectri : public SerializePolicy, public IntermediatesPolicy{
public:
  template<typename ScalarT, typename DimensionT>
  class info{
  public:
    using ScalarType = ScalarT;
    using DimensionType = DimensionT;
    using alg_type = rectri<SerializePolicy,IntermediatesPolicy>;
    using SP = SerializePolicy; using IP = IntermediatesPolicy;
    info(const info& p) : dir(p.dir) {}
    info(info&& p) : dir(p.dir) {}
    info(char dir) : dir(dir) {}
    // User input members
    const char dir;
    // Factor members
    matrix<ScalarType,DimensionType,typename SerializePolicy::structure> L;
    matrix<ScalarType,DimensionType,typename SerializePolicy::structure> Linv;
    // Optimizing members
    std::map<int,matrix<ScalarType,DimensionType,rect>> L_panel_table;
    std::map<int,matrix<ScalarType,DimensionType,rect>> L_block_table;
    std::map<int,matrix<ScalarType,DimensionType,rect>> Linv_panel_table;
    std::map<int,matrix<ScalarType,DimensionType,rect>> Linv_block_table;
    int num_levels;
    std::vector<topo::square> process_grids;
    std::vector<MPI_Comm> swap_communicators;
  };

  template<typename MatrixType, typename ArgType, typename CommType>
  static void invoke(const MatrixType& A, ArgType& args, CommType&& CommInfo);

  template<typename ArgType, typename CommType>
  static matrix<typename ArgType::ScalarType,typename ArgType::DimensionType,rect> construct_Linv(ArgType& args, CommType&& CommInfo);

  using SP = SerializePolicy; using IP = IntermediatesPolicy;

private:
  template<typename ArgType, typename CommType>
  static void invert(ArgType& args, CommType&& CommInfo);

  template<typename ArgType, typename CommType>
  static void simulate(ArgType& args, CommType&& CommInfo);
};
}

#include "rectri.hpp"

#endif /* INVERSE__RECTRI_H_ */
