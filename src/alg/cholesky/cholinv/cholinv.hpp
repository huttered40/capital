/* Author: Edward Hutter */

namespace cholesky{
template<class SerializePolicy, class IntermediatesPolicy, class PipelinePolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,PipelinePolicy>::factor(const MatrixType& A, ArgType& args, CommType&& CommInfo){

  using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType;
  assert(args.split>0); assert(args.dir == 'U');	// Removed support for 'L'. Necessary future support for this case can be handled via a final transpose.
  U localDimension = A.num_rows_local(); U globalDimension = A.num_rows_global(); U minDimLocal = 1;
  args.R._register_(A.num_columns_global(),A.num_rows_global(),CommInfo.d,CommInfo.d);
  args.Rinv._register_(A.num_columns_global(),A.num_rows_global(),CommInfo.d,CommInfo.d);
  serialize<typename MatrixType::StructureType,typename SP::structure>::invoke(A,args.R,0,localDimension,0,localDimension,0,localDimension,0,localDimension);
  U bcDimLocal = util::get_next_power2(localDimension/(CommInfo.c*CommInfo.d));
  auto bcMult = args.bc_mult_dim;
  if (bcMult<0){ bcMult *= (-1); for (int i=0;i<bcMult; i++) bcDimLocal/=2;} else {for (int i=0;i<bcMult; i++) bcDimLocal*=2;}
  bcDimLocal  = std::max(minDimLocal,bcDimLocal); bcDimLocal  = std::min(localDimension,bcDimLocal); U bcDimension = CommInfo.d*bcDimLocal;

  args.localDimension=localDimension; args.trueLocalDimension=localDimension; args.globalDimension=globalDimension; args.trueGlobalDimension=globalDimension; args.bcDimension=bcDimension;
  args.AstartX=0; args.AendX=localDimension; args.AstartY=0; args.AendY=localDimension; args.TIstartX=0; args.TIendX=localDimension; args.TIstartY=0; args.TIendY=localDimension;
  simulate(args, std::forward<CommType>(CommInfo));

  args.localDimension=localDimension; args.trueLocalDimension=localDimension; args.globalDimension=globalDimension; args.trueGlobalDimension=globalDimension; args.bcDimension=bcDimension;
  args.AstartX=0; args.AendX=localDimension; args.AstartY=0; args.AendY=localDimension; args.TIstartX=0; args.TIendX=localDimension; args.TIstartY=0; args.TIendY=localDimension;
  invoke(args, std::forward<CommType>(CommInfo));
}

template<class SerializePolicy, class IntermediatesPolicy, class PipelinePolicy>
template<typename ArgType, typename CommType>
matrix<typename ArgType::ScalarType,typename ArgType::DimensionType,rect> cholinv<SerializePolicy,IntermediatesPolicy,PipelinePolicy>::construct_R(ArgType& args, CommType&& CommInfo){
  auto localDimension = args.R.num_rows_local();
  matrix<typename ArgType::ScalarType,typename ArgType::DimensionType,rect> ret(args.R.num_columns_global(),args.R.num_rows_global(),CommInfo.c, CommInfo.c);
  serialize<typename SerializePolicy::structure,rect>::invoke(args.R, ret,0,localDimension,0,localDimension,0,localDimension,0,localDimension);
  return ret;
}

template<class SerializePolicy, class IntermediatesPolicy, class PipelinePolicy>
template<typename ArgType, typename CommType>
matrix<typename ArgType::ScalarType,typename ArgType::DimensionType,rect> cholinv<SerializePolicy,IntermediatesPolicy,PipelinePolicy>::construct_Rinv(ArgType& args, CommType&& CommInfo){
  auto localDimension = args.R.num_rows_local();
  matrix<typename ArgType::ScalarType,typename ArgType::DimensionType,rect> ret(args.Rinv.num_columns_global(),args.Rinv.num_rows_global(),CommInfo.c, CommInfo.c);
  serialize<typename SerializePolicy::structure,rect>::invoke(args.Rinv, ret,0,localDimension,0,localDimension,0,localDimension,0,localDimension);
  return ret;
}


template<class SerializePolicy, class IntermediatesPolicy, class PipelinePolicy>
template<typename ArgType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,PipelinePolicy>::simulate(ArgType& args, CommType&& CommInfo){

  using U = typename ArgType::DimensionType;
  U split1 = (args.localDimension>>args.split); split1 = util::get_next_power2(split1);
  if (((args.localDimension*CommInfo.d) <= args.bcDimension) || (split1<args.split)){
    simulate_basecase(args, std::forward<CommType>(CommInfo)); return;
  }

  split1 = (args.localDimension>>args.split); split1 = util::get_next_power2(split1); U split2 = args.localDimension-split1;
  U save1 = args.localDimension; U save2 = args.globalDimension; U save3=args.AendX; U save4=args.AendY; U save5=args.TIendX; U save6=args.TIendY;
  args.localDimension=split1; args.globalDimension=(args.globalDimension>>1); args.AendX=args.AstartX+split1; args.AendY=args.AstartY+split1; args.TIendX=args.TIstartX+split1; args.TIendY=args.TIstartY+split1;
  simulate(args, std::forward<CommType>(CommInfo));
  args.localDimension=save1; args.globalDimension=save2; args.AendX=save3; args.AendY=save4; args.TIendX=save5; args.TIendY=save6;

  IP::init(args.policy_table,std::make_pair(split1,split1),nullptr,split1,split1,CommInfo.d,CommInfo.d);
  IP::init(args.rect_table1,std::make_pair(split2,split1),nullptr,split2,split1,CommInfo.d,CommInfo.d);
  IP::init(args.rect_table2,std::make_pair(split2,split1),nullptr,split2,split1,CommInfo.d,CommInfo.d);
  IP::init(args.policy_table,std::make_pair(split2,split2),nullptr,split2,split2,CommInfo.d,CommInfo.d);

  save1 = args.localDimension; save2 = args.globalDimension; save3=args.AstartX; save4=args.AstartY; save5=args.TIstartX; save6=args.TIstartY;
  args.localDimension=split2; args.globalDimension=split2*CommInfo.d; args.AstartX=args.AstartX+split1; args.AstartY=args.AstartY+split1; args.TIstartX=args.TIstartX+split1; args.TIstartY=args.TIstartY+split1;
  simulate(args, std::forward<CommType>(CommInfo));
  args.localDimension=save1; args.globalDimension=save2; args.AstartX=save3; args.AstartY=save4; args.TIstartX=save5; args.TIstartY=save6;

  if (!(!args.complete_inv && (args.globalDimension==args.trueGlobalDimension))){
    IP::init(args.policy_table,std::make_pair(split1,split1),nullptr,split1,split1,CommInfo.d,CommInfo.d);
    IP::init(args.policy_table,std::make_pair(split2,split2),nullptr,split2,split2,CommInfo.d,CommInfo.d);
  }
}

template<class SerializePolicy, class IntermediatesPolicy, class PipelinePolicy>
template<typename ArgType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,PipelinePolicy>::simulate_basecase(ArgType& args, CommType&& CommInfo){

  using U = typename ArgType::DimensionType;
  assert(args.localDimension>0); assert((args.AendX-args.AstartX)==(args.AendY-args.AstartY));
  auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY);
  IP::init(args.base_case_table, index_pair, nullptr,args.AendX-args.AstartX,args.AendY-args.AstartY,CommInfo.d,CommInfo.d);
  U num_elems = args.base_case_table[index_pair].num_elems()*CommInfo.d*CommInfo.d;
  IP::init(args.base_case_blocked_table,index_pair, num_elems); U aggregDim = (args.AendX-args.AstartX)*CommInfo.d;
  IP::init(args.base_case_cyclic_table, index_pair, nullptr,aggregDim,aggregDim,CommInfo.d,CommInfo.d);
}

template<class SerializePolicy, class IntermediatesPolicy, class PipelinePolicy>
template<typename ArgType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,PipelinePolicy>::invoke(ArgType& args, CommType&& CommInfo){

  using ArgTypeRR = typename std::remove_reference<ArgType>::type; using T = typename ArgTypeRR::ScalarType; using U = typename ArgTypeRR::DimensionType;
  U split1 = (args.localDimension>>args.split); split1 = util::get_next_power2(split1);
  if (((args.localDimension*CommInfo.d) <= args.bcDimension) || (split1<args.split)){
    base_case(args, std::forward<CommType>(CommInfo));
    return;
  }

  split1 = (args.localDimension>>args.split); split1 = util::get_next_power2(split1); U split2 = args.localDimension-split1;
  U save1 = args.localDimension; U save2 = args.globalDimension; U save3=args.AendX; U save4=args.AendY; U save5=args.TIendX; U save6=args.TIendY;
  args.localDimension=split1; args.globalDimension=(args.globalDimension>>1); args.AendX=args.AstartX+split1; args.AendY=args.AstartY+split1; args.TIendX=args.TIstartX+split1; args.TIendY=args.TIstartY+split1;
  invoke(args, std::forward<CommType>(CommInfo));
  args.localDimension=save1; args.globalDimension=save2; args.AendX=save3; args.AendY=save4; args.TIendX=save5; args.TIendY=save6;

  serialize<uppertri,uppertri>::invoke(args.Rinv, IP::invoke(args.policy_table,std::make_pair(split1,split1)), args.TIstartX, args.TIstartX+split1, args.TIstartY, args.TIstartY+split1,0,split1,0,split1);
  util::transpose(IP::invoke(args.policy_table,std::make_pair(split1,split1)), std::forward<CommType>(CommInfo));
  blas::ArgPack_trmm<T> trmmArgs(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper, blas::Transpose::AblasTrans, blas::Diag::AblasNonUnit, 1.);

  serialize<rect,rect>::invoke(args.R, IP::invoke(args.rect_table1,std::make_pair(split2,split1)), args.AstartX+split1, args.AendX, args.AstartY, args.AstartY+split1,0,split2,0,split1);
  matmult::summa::invoke(IP::invoke(args.policy_table,std::make_pair(split1,split1)), IP::invoke(args.rect_table1,std::make_pair(split2,split1)), std::forward<CommType>(CommInfo), trmmArgs);
  serialize<rect,rect>::invoke(IP::invoke(args.rect_table1,std::make_pair(split2,split1)), args.R, 0,split2,0,split1,args.AstartX+split1, args.AendX, args.AstartY, args.AstartY+split1);
  serialize<rect,rect>::invoke(IP::invoke(args.rect_table1,std::make_pair(split2,split1)), IP::invoke(args.rect_table2,std::make_pair(split2,split1)),0,split2,0,split1,0,split2,0,split1);

  blas::ArgPack_syrk<T> syrkArgs(blas::Order::AblasColumnMajor, blas::UpLo::AblasUpper, blas::Transpose::AblasTrans, -1., 1.);
  serialize<uppertri,uppertri>::invoke(args.R, IP::invoke(args.policy_table,std::make_pair(split2,split2)), args.AstartX+split1, args.AendX, args.AstartY+split1, args.AendY,0,split2,0,split2);
  matmult::summa::invoke(IP::invoke(args.rect_table1,std::make_pair(split2,split1)), IP::invoke(args.rect_table2,std::make_pair(split2,split1)), IP::invoke(args.policy_table,std::make_pair(split2,split2)), std::forward<CommType>(CommInfo), syrkArgs);
  serialize<uppertri,uppertri>::invoke(IP::invoke(args.policy_table,std::make_pair(split2,split2)), args.R, 0,split2,0,split2,args.AstartX+split1, args.AendX, args.AstartY+split1, args.AendY);

  save1 = args.localDimension; save2 = args.globalDimension; save3=args.AstartX; save4=args.AstartY; save5=args.TIstartX; save6=args.TIstartY;
  args.localDimension=split2; args.globalDimension=split2*CommInfo.d; args.AstartX=args.AstartX+split1; args.AstartY=args.AstartY+split1; args.TIstartX=args.TIstartX+split1; args.TIstartY=args.TIstartY+split1;
  invoke(args, std::forward<CommType>(CommInfo));
  args.localDimension=save1; args.globalDimension=save2; args.AstartX=save3; args.AstartY=save4; args.TIstartX=save5; args.TIstartY=save6;

  if (!(!args.complete_inv && (args.globalDimension==args.trueGlobalDimension))){
    serialize<rect,rect>::invoke(args.R, IP::invoke(args.rect_table1,std::make_pair(split2,split1)), args.AstartX+split1, args.AendX, args.AstartY, args.AstartY+split1,0,split2,0,split1);
    serialize<uppertri,uppertri>::invoke(args.Rinv, IP::invoke(args.policy_table,std::make_pair(split1,split1)), args.TIstartX, args.TIstartX+split1, args.TIstartY, args.TIstartY+split1,0,split1,0,split1);
    blas::ArgPack_trmm<T> invPackage1(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper, blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    matmult::summa::invoke(IP::invoke(args.policy_table,std::make_pair(split1,split1)), IP::invoke(args.rect_table1,std::make_pair(split2,split1)), std::forward<CommType>(CommInfo), invPackage1);
    invPackage1.alpha = -1.; invPackage1.side = blas::Side::AblasRight;
    serialize<uppertri,uppertri>::invoke(args.Rinv, IP::invoke(args.policy_table,std::make_pair(split2,split2)), args.TIstartX+split1, args.TIendX, args.TIstartY+split1, args.TIendY,0,split2,0,split2);
    matmult::summa::invoke(IP::invoke(args.policy_table,std::make_pair(split2,split2)), IP::invoke(args.rect_table1,std::make_pair(split2,split1)), std::forward<CommType>(CommInfo), invPackage1);
    serialize<rect,rect>::invoke(IP::invoke(args.rect_table1,std::make_pair(split2,split1)), args.Rinv,0,split2,0,split1,args.TIstartX+split1, args.TIendX, args.TIstartY, args.TIstartY+split1);
  }
  IP::flush(args.rect_table1[std::make_pair(split2,split1)]); IP::flush(args.rect_table2[std::make_pair(split2,split1)]);
  IP::flush(args.policy_table[std::make_pair(split1,split1)]); IP::flush(args.policy_table[std::make_pair(split2,split2)]);
}


template<class SerializePolicy, class IntermediatesPolicy, class PipelinePolicy>
template<typename ArgType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,PipelinePolicy>::base_case(ArgType& args, CommType&& CommInfo){

  using ArgTypeRR = typename std::remove_reference<ArgType>::type; using T = typename ArgTypeRR::ScalarType; using U = typename ArgTypeRR::DimensionType;

  // No matter what path we are on, if we get into the base case, we will do regular Cholesky + Triangular inverse
  // First: AllGather matrix A so that every processor has the same replicated diagonal square partition of matrix A of dimension bcDimension
  //          Note that processors only want to communicate with those on their same 2D slice, since the matrices are replicated on every slice
  //          Note that before the AllGather, we need to serialize the matrix A into the small square matrix
  // Second: Data will be received in a blocked order due to AllGather semantics, which is not what we want. We need to get back to cyclic again
  //           This is an ugly process, as it was in the last code.
  // Third: Once data is in cyclic format, we call call sequential Cholesky Factorization and Triangular Inverse.
  // Fourth: Save the data that each processor owns according to the cyclic rule.

  int rankSlice; MPI_Comm_rank(CommInfo.slice, &rankSlice); U aggregDim = (args.AendX-args.AstartX)*CommInfo.d;
  U span = (args.AendX!=args.trueLocalDimension ? aggregDim :aggregDim-(args.trueLocalDimension*CommInfo.d-args.trueGlobalDimension)); auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY);
  serialize<uppertri,uppertri>::invoke(args.R, IP::invoke(args.base_case_table,index_pair), args.AstartX, args.AendX, args.AstartY, args.AendY,0,args.AendX-args.AstartX,0,args.AendY-args.AstartY);
  SP::invoke(IP::invoke(args.base_case_table,index_pair),args.base_case_blocked_table[index_pair],IP::invoke(args.base_case_cyclic_table,index_pair),std::forward<CommType>(CommInfo));
  lapack::ArgPack_potrf potrfArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper);
  lapack::ArgPack_trtri trtriArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper, lapack::Diag::AlapackNonUnit);
  lapack::engine::_potrf(IP::invoke(args.base_case_cyclic_table,index_pair).data(),span,aggregDim,potrfArgs);
  std::memcpy(IP::invoke(args.base_case_cyclic_table,index_pair).scratch(),IP::invoke(args.base_case_cyclic_table,index_pair).data(),sizeof(T)*IP::invoke(args.base_case_cyclic_table,index_pair).num_elems());
  lapack::engine::_trtri(IP::invoke(args.base_case_cyclic_table,index_pair).scratch(),span,aggregDim,trtriArgs);
  util::cyclic_to_local(IP::invoke(args.base_case_cyclic_table,index_pair).data(),IP::invoke(args.base_case_cyclic_table,index_pair).scratch(), args.localDimension, args.globalDimension, aggregDim, CommInfo.d,rankSlice);
  serialize<uppertri,uppertri>::invoke(IP::invoke(args.base_case_cyclic_table,index_pair), args.R, 0,args.AendX-args.AstartX,0,args.AendY-args.AstartY,args.AstartY, args.AendY, args.AstartY, args.AendY);
  IP::invoke(args.base_case_cyclic_table,index_pair).swap();	// puts the inverse buffer into the `data` member before final serialization
  serialize<uppertri,uppertri>::invoke(IP::invoke(args.base_case_cyclic_table,index_pair), args.Rinv,0,args.AendX-args.AstartX,0,args.AendY-args.AstartY,args.TIstartX, args.TIendX, args.TIstartY, args.TIendY);
  IP::invoke(args.base_case_cyclic_table,index_pair).swap();	// puts the inverse buffer into the `data` member before final serialization
  IP::flush(args.base_case_table[index_pair]); IP::flush(args.base_case_cyclic_table[index_pair]);
  return;
}
}
