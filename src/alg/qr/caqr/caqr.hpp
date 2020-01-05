/* Author: Edward Hutter */

namespace qr{

template<class PipelinePolicy, class SerializePolicy, class IntermediatesPolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void caqr<PipelinePolicy,SerializePolicy,IntermediatesPolicy>::factor(const MatrixType& A, ArgType& args, CommType&& CommInfo){
  using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType; using SP = SerializePolicy; using IP = IntermediatesPolicy;
  static_assert(0,"not implemented");
}

template<class PipelinePolicy, class SerializePolicy, class IntermediatesPolicy>
template<typename ArgType, typename CommType>
matrix<typename ArgType::ScalarType,typename ArgType::DimensionType,rect> caqr<PipelinePolicy,SerializePolicy,IntermediatesPolicy>::construct_Q(ArgType& args, CommType&& CommInfo){
  static_assert(0,"not implemented");
}

template<class PipelinePolicy, class SerializePolicy, class IntermediatesPolicy>
template<typename ArgType, typename CommType>
matrix<typename ArgType::ScalarType,typename ArgType::DimensionType,rect> caqr<PipelinePolicy,SerializePolicy,IntermediatesPolicy>::construct_R(ArgType& args, CommType&& CommInfo){
  static_assert(0,"not implemented");
}

template<class PipelinePolicy, class SerializePolicy, class IntermediatesPolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void caqr<PipelinePolicy,SerializePolicy,IntermediatesPolicy>::apply_Q(MatrixType& src, ArgType& args,CommType&& CommInfo) { static_assert(0,"not implemented"); }

template<class PipelinePolicy, class SerializePolicy, class IntermediatesPolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void caqr<PipelinePolicy,SerializePolicy,IntermediatesPolicy>::apply_QT(MatrixType& src, ArgType& args,CommType&& CommInfo) { static_assert(0,"not implemented"); }

}
