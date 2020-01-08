/* Author: Edward Hutter */

namespace trsm{

template<class SerializePolicy, class IntermediatesPolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void diaginvert<SerializePolicy,IntermediatesPolicy>::solve(const MatrixType& A, MatrixType& X, const MatrixType& B, ArgType& args, CommType&& CommInfo){
  using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType; using SP = SerializePolicy; using IP = IntermediatesPolicy;
  static_assert(0,"not implemented");
}

}
