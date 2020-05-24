/* Author: Edward Hutter */

namespace cholesky{
template<class SerializePolicy, class IntermediatesPolicy, class BaseCasePolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,BaseCasePolicy>::factor(const MatrixType& A, ArgType& args, CommType&& CommInfo){
  CRITTER_START(CI::factor);
  using T = typename MatrixType::ScalarType;
  assert(args.split>0); assert(args.dir == 'U');	// Removed support for 'L'. Necessary future support for this case can be handled via a final transpose.
  auto localDimension = A.num_rows_local(); auto globalDimension = A.num_rows_global(); typename ArgType::DimensionType minDimLocal = 1;
  args.R._register_(A.num_columns_global(),A.num_rows_global(),CommInfo.d,CommInfo.d);
  args.Rinv._register_(A.num_columns_global(),A.num_rows_global(),CommInfo.d,CommInfo.d);
  serialize<uppertri,uppertri>::invoke(A,args.R,0,localDimension,0,localDimension,0,localDimension,0,localDimension);

  typename ArgType::DimensionType bcDimLocal = CommInfo.c*CommInfo.d; auto bcMult = args.bc_mult_dim;
  if (bcMult<0){ bcMult *= (-1); for (int i=0;i<bcMult; i++) bcDimLocal*=2;} else {for (int i=0;i<bcMult; i++) bcDimLocal/=2;}
  bcDimLocal  = std::max(minDimLocal,bcDimLocal); bcDimLocal  = std::min(localDimension,bcDimLocal);
  bcDimLocal = localDimension/bcDimLocal; auto bcDimension = CommInfo.d*bcDimLocal;

  args.localDimension=localDimension; args.trueLocalDimension=localDimension; args.globalDimension=globalDimension; args.trueGlobalDimension=globalDimension; args.bcDimension=bcDimension;
  args.AstartX=0; args.AendX=localDimension; args.AstartY=0; args.AendY=localDimension; args.TIstartX=0; args.TIendX=localDimension; args.TIstartY=0; args.TIendY=localDimension;
  simulate(args, std::forward<CommType>(CommInfo));

  args.localDimension=localDimension; args.trueLocalDimension=localDimension; args.globalDimension=globalDimension; args.trueGlobalDimension=globalDimension; args.bcDimension=bcDimension;
  args.AstartX=0; args.AendX=localDimension; args.AstartY=0; args.AendY=localDimension; args.TIstartX=0; args.TIendX=localDimension; args.TIstartY=0; args.TIendY=localDimension;
  invoke(args, std::forward<CommType>(CommInfo));
  CRITTER_STOP(CI::factor);
}

template<class SerializePolicy, class IntermediatesPolicy, class BaseCasePolicy>
template<typename ArgType, typename CommType>
matrix<typename ArgType::ScalarType,typename ArgType::DimensionType,rect> cholinv<SerializePolicy,IntermediatesPolicy,BaseCasePolicy>::construct_R(ArgType& args, CommType&& CommInfo){
  auto localDimension = args.R.num_rows_local();
  matrix<typename ArgType::ScalarType,typename ArgType::DimensionType,rect> ret(args.R.num_columns_global(),args.R.num_rows_global(),CommInfo.c, CommInfo.c);
  serialize<typename SerializePolicy::structure,rect>::invoke(args.R, ret,0,localDimension,0,localDimension,0,localDimension,0,localDimension);
  return ret;
}

template<class SerializePolicy, class IntermediatesPolicy, class BaseCasePolicy>
template<typename ArgType, typename CommType>
matrix<typename ArgType::ScalarType,typename ArgType::DimensionType,rect> cholinv<SerializePolicy,IntermediatesPolicy,BaseCasePolicy>::construct_Rinv(ArgType& args, CommType&& CommInfo){
  auto localDimension = args.R.num_rows_local();
  matrix<typename ArgType::ScalarType,typename ArgType::DimensionType,rect> ret(args.Rinv.num_columns_global(),args.Rinv.num_rows_global(),CommInfo.c, CommInfo.c);
  serialize<typename SerializePolicy::structure,rect>::invoke(args.Rinv, ret,0,localDimension,0,localDimension,0,localDimension,0,localDimension);
  return ret;
}

template<class SerializePolicy, class IntermediatesPolicy, class BaseCasePolicy>
template<typename ArgType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,BaseCasePolicy>::simulate(ArgType& args, CommType&& CommInfo){
  auto split1 = (args.localDimension>>args.split); split1 = split1;
  if (((args.localDimension*CommInfo.d) <= args.bcDimension) || (split1<args.split)){
    simulate_basecase(args, std::forward<CommType>(CommInfo)); return;
  }

  split1 = (args.localDimension>>args.split); split1 = split1; auto split2 = args.localDimension-split1;
  auto save1 = args.localDimension; auto save2 = args.globalDimension; auto save3=args.AendX; auto save4=args.AendY; auto save5=args.TIendX; auto save6=args.TIendY;
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

template<class SerializePolicy, class IntermediatesPolicy, class BaseCasePolicy>
template<typename ArgType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,BaseCasePolicy>::simulate_basecase(ArgType& args, CommType&& CommInfo){
  assert(args.localDimension>0); assert((args.AendX-args.AstartX)==(args.AendY-args.AstartY));
  IP::create_buffers(BP::get_id(),args,std::forward<CommType>(CommInfo));
}

template<class SerializePolicy, class IntermediatesPolicy, class BaseCasePolicy>
template<typename ArgType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,BaseCasePolicy>::invoke(ArgType& args, CommType&& CommInfo){
#ifdef FUNCTION_SYMBOLS
  CRITTER_START(CI::invoke);
#endif
  using ArgTypeRR = typename std::remove_reference<ArgType>::type; using T = typename ArgTypeRR::ScalarType;
  auto split1 = (args.localDimension>>args.split); split1 = split1;
  if (((args.localDimension*CommInfo.d) <= args.bcDimension) || (split1<args.split)){
#ifdef ALGORITHMIC_SYMBOLS
    CRITTER_START(CI::factor_diag);
#endif
    base_case(args, std::forward<CommType>(CommInfo));
#ifdef ALGORITHMIC_SYMBOLS
    CRITTER_STOP(CI::factor_diag);
#endif
#ifdef FUNCTION_SYMBOLS
    CRITTER_STOP(CI::invoke);
#endif
    return;
  }

  split1 = (args.localDimension>>args.split); split1 = split1; auto split2 = args.localDimension-split1;
  auto save1 = args.localDimension; auto save2 = args.globalDimension; auto save3=args.AendX; auto save4=args.AendY; auto save5=args.TIendX; auto save6=args.TIendY;
  args.localDimension=split1; args.globalDimension=(args.globalDimension>>1); args.AendX=args.AstartX+split1; args.AendY=args.AstartY+split1; args.TIendX=args.TIstartX+split1; args.TIendY=args.TIstartY+split1;
  invoke(args, std::forward<CommType>(CommInfo));
  args.localDimension=save1; args.globalDimension=save2; args.AendX=save3; args.AendY=save4; args.TIendX=save5; args.TIendY=save6;

#ifdef ALGORITHMIC_SYMBOLS
  CRITTER_START(CI::trsm);
#endif
  serialize<uppertri,uppertri>::invoke(args.Rinv, IP::invoke(args.policy_table,std::make_pair(split1,split1)), args.TIstartX, args.TIstartX+split1, args.TIstartY, args.TIstartY+split1,0,split1,0,split1);
  util::transpose(IP::invoke(args.policy_table,std::make_pair(split1,split1)), std::forward<CommType>(CommInfo));
  blas::ArgPack_trmm<T> trmmArgs(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper, blas::Transpose::AblasTrans, blas::Diag::AblasNonUnit, 1.);

  serialize<rect,rect>::invoke(args.R, IP::invoke(args.rect_table1,std::make_pair(split2,split1)), args.AstartX+split1, args.AendX, args.AstartY, args.AstartY+split1,0,split2,0,split1);
  matmult::summa::invoke(IP::invoke(args.policy_table,std::make_pair(split1,split1)), IP::invoke(args.rect_table1,std::make_pair(split2,split1)), std::forward<CommType>(CommInfo), trmmArgs);
  serialize<rect,rect>::invoke(IP::invoke(args.rect_table1,std::make_pair(split2,split1)), args.R, 0,split2,0,split1,args.AstartX+split1, args.AendX, args.AstartY, args.AstartY+split1);
  serialize<rect,rect>::invoke(IP::invoke(args.rect_table1,std::make_pair(split2,split1)), IP::invoke(args.rect_table2,std::make_pair(split2,split1)),0,split2,0,split1,0,split2,0,split1);
#ifdef ALGORITHMIC_SYMBOLS
  CRITTER_STOP(CI::trsm);
#endif

#ifdef ALGORITHMIC_SYMBOLS
  CRITTER_START(CI::tmu);
#endif
  blas::ArgPack_syrk<T> syrkArgs(blas::Order::AblasColumnMajor, blas::UpLo::AblasUpper, blas::Transpose::AblasTrans, -1., 1.);
  serialize<uppertri,uppertri>::invoke(args.R, IP::invoke(args.policy_table,std::make_pair(split2,split2)), args.AstartX+split1, args.AendX, args.AstartY+split1, args.AendY,0,split2,0,split2);
  matmult::summa::invoke(IP::invoke(args.rect_table1,std::make_pair(split2,split1)), IP::invoke(args.rect_table2,std::make_pair(split2,split1)), IP::invoke(args.policy_table,std::make_pair(split2,split2)), std::forward<CommType>(CommInfo), syrkArgs);
  serialize<uppertri,uppertri>::invoke(IP::invoke(args.policy_table,std::make_pair(split2,split2)), args.R, 0,split2,0,split2,args.AstartX+split1, args.AendX, args.AstartY+split1, args.AendY);
#ifdef ALGORITHMIC_SYMBOLS
  CRITTER_STOP(CI::tmu);
#endif

  save1 = args.localDimension; save2 = args.globalDimension; save3=args.AstartX; save4=args.AstartY; save5=args.TIstartX; save6=args.TIstartY;
  args.localDimension=split2; args.globalDimension=split2*CommInfo.d; args.AstartX=args.AstartX+split1; args.AstartY=args.AstartY+split1; args.TIstartX=args.TIstartX+split1; args.TIstartY=args.TIstartY+split1;
  invoke(args, std::forward<CommType>(CommInfo));
  args.localDimension=save1; args.globalDimension=save2; args.AstartX=save3; args.AstartY=save4; args.TIstartX=save5; args.TIstartY=save6;

#ifdef ALGORITHMIC_SYMBOLS
  CRITTER_START(CI::tmu);
#endif
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
#ifdef ALGORITHMIC_SYMBOLS
  CRITTER_STOP(CI::tmu);
#endif
  IP::flush(args.rect_table1[std::make_pair(split2,split1)]); IP::flush(args.rect_table2[std::make_pair(split2,split1)]);
  IP::flush(args.policy_table[std::make_pair(split1,split1)]); IP::flush(args.policy_table[std::make_pair(split2,split2)]);
#ifdef FUNCTION_SYMBOLS
  CRITTER_STOP(CI::invoke);
#endif
}


template<class SerializePolicy, class IntermediatesPolicy, class BaseCasePolicy>
template<typename ArgType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,BaseCasePolicy>::base_case(ArgType& args, CommType&& CommInfo){
#ifdef FUNCTION_SYMBOLS
  CRITTER_START(CI::base_case);
#endif
  auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY);
  IP::init_buffers(BP::get_id(),args,std::forward<CommType>(CommInfo));
  BP::initiate(args,std::forward<CommType>(CommInfo));
  BP::compute(args,std::forward<CommType>(CommInfo));
  BP::complete(args,std::forward<CommType>(CommInfo));
  IP::remove_buffers(BP::get_id(),args,std::forward<CommType>(CommInfo));
#ifdef FUNCTION_SYMBOLS
  CRITTER_STOP(CI::base_case);
#endif
}
}
