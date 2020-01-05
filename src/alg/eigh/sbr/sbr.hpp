/* Author: Edward Hutter */

namespace eigh{

template<class SerializePolicy, class IntermediatesPolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void sbr<SerializePolicy,IntermediatesPolicy>::factor(const MatrixType& A, ArgType& args, CommType&& CommInfo){
  using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType; using SP = SerializePolicy; using IP = IntermediatesPolicy;
  static_assert(0,"not implemented");
}

}
