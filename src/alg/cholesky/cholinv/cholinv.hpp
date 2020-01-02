/* Author: Edward Hutter */

namespace cholesky{
template<class SerializePolicy, class IntermediatesPolicy, class OverlapPolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>::invoke(MatrixType& A, MatrixType& TI, ArgType&& args, CommType&& CommInfo){

  using SP = SerializePolicy; using IP = IntermediatesPolicy; using OP = OverlapPolicy;
  using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType;
  using Offload = typename MatrixType::OffloadType;
  static_assert(std::is_same<typename MatrixType::StructureType,rect>::value,"cholesky::cholinv requires non-packed matrices\n");
  assert(args.dir == 'U');	// Removed support for 'L'. Necessary future support for this case can be handled via a final transpose.
  U localDimension = A.num_rows_local(); U globalDimension = A.num_rows_global(); U minDimLocal = 1;
  U bcDimLocal = util::get_next_power2(localDimension/(CommInfo.c*CommInfo.d));
  auto bcMult = args.bc_mult_dim;
  if (bcMult<0){ bcMult *= (-1); for (int i=0;i<bcMult; i++) bcDimLocal/=2;} else {for (int i=0;i<bcMult; i++) bcDimLocal*=2;}
  bcDimLocal  = std::max(minDimLocal,bcDimLocal); bcDimLocal  = std::min(localDimension,bcDimLocal); U bcDimension = CommInfo.d*bcDimLocal;

  args.localDimension=localDimension; args.trueLocalDimension=localDimension; args.globalDimension=globalDimension; args.trueGlobalDimension=globalDimension; args.bcDimension=bcDimension;
  args.AstartX=0; args.AendX=localDimension; args.AstartY=0; args.AendY=localDimension; args.TIstartX=0; args.TIendX=localDimension; args.TIstartY=0; args.TIendY=localDimension;
  simulate(std::forward<ArgType>(args), std::forward<CommType>(CommInfo));

  args.localDimension=localDimension; args.trueLocalDimension=localDimension; args.globalDimension=globalDimension; args.trueGlobalDimension=globalDimension; args.bcDimension=bcDimension;
  args.AstartX=0; args.AendX=localDimension; args.AstartY=0; args.AendY=localDimension; args.TIstartX=0; args.TIendX=localDimension; args.TIstartY=0; args.TIendY=localDimension;
  factor(A, TI, std::forward<ArgType>(args), std::forward<CommType>(CommInfo));
}

template<class SerializePolicy, class IntermediatesPolicy, class OverlapPolicy>
template<typename ScalarType, typename DimensionType, typename ArgType, typename CommType>
std::pair<ScalarType*,ScalarType*> cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>::invoke(ScalarType* A, ScalarType* TI, DimensionType localDim, DimensionType globalDim, ArgType&& args, CommType&& CommInfo){
  //TODO: Test with non-power-of-2 global matrix dimensions
  matrix<ScalarType,DimensionType,rect,cyclic> mA(A,localDim,localDim,globalDim,globalDim,CommInfo.c,CommInfo.c);	// re-used in SquareTable
  matrix<ScalarType,DimensionType,rect,cyclic> mTI(R,localDim,localDim,globalDim,globalDim,CommInfo.c,CommInfo.c);
  invoke(mA,mTI,std::forward<ArgType>(args),std::forward<CommType>(CommInfo));
  return std::make_pair(mA.get_data(),mTI.get_data());
}

template<class SerializePolicy, class IntermediatesPolicy, class OverlapPolicy>
template<typename ArgType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>::simulate(ArgType&& args, CommType&& CommInfo){

  using SP = SerializePolicy; using IP = IntermediatesPolicy; using OP = OverlapPolicy;
  using U = int64_t;
  if ((args.localDimension*CommInfo.d) <= args.bcDimension){
    simulate_basecase(std::forward<ArgType>(args), std::forward<CommType>(CommInfo));
    return;
  }

  U split1 = (args.localDimension>>args.split); split1 = util::get_next_power2(split1); U split2 = args.localDimension-split1;
  U save1 = args.localDimension; U save2 = args.globalDimension; U save3=args.AendX; U save4=args.AendY; U save5=args.TIendX; U save6=args.TIendY;
  args.localDimension=split1; args.globalDimension=(args.globalDimension>>1); args.AendX=args.AstartX+split1; args.AendY=args.AstartY+split1; args.TIendX=args.TIstartX+split1; args.TIendY=args.TIstartY+split1;
  simulate(std::forward<ArgType>(args), std::forward<CommType>(CommInfo));
  args.localDimension=save1; args.globalDimension=save2; args.AendX=save3; args.AendY=save4; args.TIendX=save5; args.TIendY=save6;

  IP::init(args.policy_table,std::make_pair(split1,split1),nullptr,split1,split1,CommInfo.d,CommInfo.d);
  IP::init(args.rect_table,std::make_pair(split2,split1),nullptr,split2,split1,CommInfo.d,CommInfo.d);
  IP::init(args.policy_table,std::make_pair(split2,split2),nullptr,split2,split2,CommInfo.d,CommInfo.d);	// this might be a problem. It was coupled before

  save1 = args.localDimension; save2 = args.globalDimension; save3=args.AstartX; save4=args.AstartY; save5=args.TIstartX; save6=args.TIstartY;
  args.localDimension=split2; args.globalDimension=split2*CommInfo.d; args.AstartX=args.AstartX+split1; args.AstartY=args.AstartY+split1; args.TIstartX=args.TIstartX+split1; args.TIstartY=args.TIstartY+split1;
  simulate(std::forward<ArgType>(args), std::forward<CommType>(CommInfo));
  args.localDimension=save1; args.globalDimension=save2; args.AstartX=save3; args.AstartY=save4; args.TIstartX=save5; args.TIstartY=save6;

  if (!(!args.complete_inv && (args.globalDimension==args.trueGlobalDimension))){
    IP::init(args.policy_table,std::make_pair(split1,split1),nullptr,split1,split1,CommInfo.d,CommInfo.d);
    IP::init(args.policy_table,std::make_pair(split2,split2),nullptr,split2,split2,CommInfo.d,CommInfo.d);
  }
}

template<class SerializePolicy, class IntermediatesPolicy, class OverlapPolicy>
template<typename ArgType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>::simulate_basecase(ArgType&& args, CommType&& CommInfo){

  using SP = SerializePolicy; using IP = IntermediatesPolicy; using OP = OverlapPolicy; using U = int64_t;
  if (args.localDimension==0) return;
  auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY);
  IP::init(args.base_case_table, index_pair, nullptr,args.AendX-args.AstartX,args.AendY-args.AstartY,CommInfo.d,CommInfo.d);
  U num_elems = args.base_case_table[index_pair].num_elems()*CommInfo.d*CommInfo.d;
  IP::init(args.base_case_blocked_table,index_pair, num_elems);
  IP::init(args.base_case_cyclic_table, index_pair, nullptr,args.bcDimension,args.bcDimension,CommInfo.d,CommInfo.d);
}

template<class SerializePolicy, class IntermediatesPolicy, class OverlapPolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>::factor(MatrixType& A, MatrixType& TI, ArgType&& args, CommType&& CommInfo){

  using SP = SerializePolicy; using IP = IntermediatesPolicy; using OP = OverlapPolicy;
  using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType; using Offload = typename MatrixType::OffloadType;
  if ((args.localDimension*CommInfo.d) <= args.bcDimension){
    base_case(A, TI, std::forward<ArgType>(args), std::forward<CommType>(CommInfo));
    return;
  }

  U split1 = (args.localDimension>>args.split); split1 = util::get_next_power2(split1); U split2 = args.localDimension-split1;
  U save1 = args.localDimension; U save2 = args.globalDimension; U save3=args.AendX; U save4=args.AendY; U save5=args.TIendX; U save6=args.TIendY;
  args.localDimension=split1; args.globalDimension=(args.globalDimension>>1); args.AendX=args.AstartX+split1; args.AendY=args.AstartY+split1; args.TIendX=args.TIstartX+split1; args.TIendY=args.TIstartY+split1;
  factor(A, TI, std::forward<ArgType>(args), std::forward<CommType>(CommInfo));
  args.localDimension=save1; args.globalDimension=save2; args.AendX=save3; args.AendY=save4; args.TIendX=save5; args.TIendY=save6;

  serialize<rect,typename SP::structure>::invoke(TI, IP::invoke(args.policy_table,std::make_pair(split1,split1)), args.TIstartX, args.TIstartX+split1, args.TIstartY, args.TIstartY+split1);
  util::transpose(IP::invoke(args.policy_table,std::make_pair(split1,split1)), std::forward<CommType>(CommInfo));
  blas::ArgPack_trmm<T> trmmArgs(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper, blas::Transpose::AblasTrans, blas::Diag::AblasNonUnit, 1.);

  // 2nd case: Extra optimization for the case when we only perform TRSM at the top level.
  serialize<rect,rect>::invoke(A, IP::invoke(args.rect_table,std::make_pair(split2,split1)), args.AstartX+split1, args.AendX, args.AstartY, args.AstartY+split1);
  matmult::summa::invoke(IP::invoke(args.policy_table,std::make_pair(split1,split1)), IP::invoke(args.rect_table,std::make_pair(split2,split1)), std::forward<CommType>(CommInfo), trmmArgs);
  serialize<rect,rect>::invoke(A, IP::invoke(args.rect_table,std::make_pair(split2,split1)), args.AstartX+split1, args.AendX, args.AstartY, args.AstartY+split1, true);

  blas::ArgPack_syrk<T> syrkArgs(blas::Order::AblasColumnMajor, blas::UpLo::AblasUpper, blas::Transpose::AblasTrans, -1., 1.);
  serialize<rect,rect>::invoke(A, IP::invoke(args.rect_table,std::make_pair(split2,split1)), args.AstartX+split1, args.AendX, args.AstartY, args.AstartY+split1);
  serialize<rect,typename SP::structure>::invoke(A, IP::invoke(args.policy_table,std::make_pair(split2,split2)), args.AstartX+split1, args.AendX, args.AstartY+split1, args.AendY);
  matmult::summa::invoke(IP::invoke(args.rect_table,std::make_pair(split2,split1)), IP::invoke(args.policy_table,std::make_pair(split2,split2)), std::forward<CommType>(CommInfo), syrkArgs);
  serialize<rect,typename SP::structure>::invoke(A, IP::invoke(args.policy_table,std::make_pair(split2,split2)), args.AstartX+split1, args.AendX, args.AstartY+split1, args.AendY, true);

  save1 = args.localDimension; save2 = args.globalDimension; save3=args.AstartX; save4=args.AstartY; save5=args.TIstartX; save6=args.TIstartY;
  args.localDimension=split2; args.globalDimension=split2*CommInfo.d; args.AstartX=args.AstartX+split1; args.AstartY=args.AstartY+split1; args.TIstartX=args.TIstartX+split1; args.TIstartY=args.TIstartY+split1;
  factor(A, TI, std::forward<ArgType>(args), std::forward<CommType>(CommInfo));
  args.localDimension=save1; args.globalDimension=save2; args.AstartX=save3; args.AstartY=save4; args.TIstartX=save5; args.TIstartY=save6;

  // Next step : temp <- R_{12}*TI_{22}
  if (!(!args.complete_inv && (args.globalDimension==args.trueGlobalDimension))){
    serialize<rect,rect>::invoke(A, IP::invoke(args.rect_table,std::make_pair(split2,split1)), args.AstartX+split1, args.AendX, args.AstartY, args.AstartY+split1);
    serialize<rect,typename SP::structure>::invoke(TI, IP::invoke(args.policy_table,std::make_pair(split1,split1)), args.TIstartX, args.TIstartX+split1, args.TIstartY, args.TIstartY+split1);
    blas::ArgPack_trmm<T> invPackage1(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper, blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    matmult::summa::invoke(IP::invoke(args.policy_table,std::make_pair(split1,split1)), IP::invoke(args.rect_table,std::make_pair(split2,split1)), std::forward<CommType>(CommInfo), invPackage1);
    // Next step: finish the Triangular inverse calculation
    invPackage1.alpha = -1.; invPackage1.side = blas::Side::AblasRight;
    serialize<rect,typename SP::structure>::invoke(TI, IP::invoke(args.policy_table,std::make_pair(split2,split2)), args.TIstartX+split1, args.TIendX, args.TIstartY+split1, args.TIendY);
    matmult::summa::invoke(IP::invoke(args.policy_table,std::make_pair(split2,split2)), IP::invoke(args.rect_table,std::make_pair(split2,split1)), std::forward<CommType>(CommInfo), invPackage1);
    serialize<rect,rect>::invoke(TI, IP::invoke(args.rect_table,std::make_pair(split2,split1)), args.TIstartX+split1, args.TIendX, args.TIstartY, args.TIstartY+split1, true);
  }
  IP::flush(args.rect_table[std::make_pair(split2,split1)]); IP::flush(args.policy_table_diaginv[std::make_pair(split1,split1)]); IP::flush(args.policy_table[std::make_pair(split1,split1)]);
  IP::flush(args.policy_table[std::make_pair(split2,split2)]);
}


template<class SerializePolicy, class IntermediatesPolicy, class OverlapPolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>::base_case(MatrixType& A, MatrixType& TI, ArgType&& args, CommType&& CommInfo){

  using SP = SerializePolicy; using IP = IntermediatesPolicy; using OP = OverlapPolicy;
  using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType; using Offload = typename MatrixType::OffloadType;

  // No matter what path we are on, if we get into the base case, we will do regular Cholesky + Triangular inverse
  // First: AllGather matrix A so that every processor has the same replicated diagonal square partition of matrix A of dimension bcDimension
  //          Note that processors only want to communicate with those on their same 2D slice, since the matrices are replicated on every slice
  //          Note that before the AllGather, we need to serialize the matrix A into the small square matrix
  // Second: Data will be received in a blocked order due to AllGather semantics, which is not what we want. We need to get back to cyclic again
  //           This is an ugly process, as it was in the last code.
  // Third: Once data is in cyclic format, we call call sequential Cholesky Factorization and Triangular Inverse.
  // Fourth: Save the data that each processor owns according to the cyclic rule.

  if (args.localDimension==0) return;
  int rankSlice; MPI_Comm_rank(CommInfo.slice, &rankSlice);
  auto index_pair = std::make_pair(args.AendX-args.AstartX,args.AendY-args.AstartY);
  serialize<rect,typename SP::structure>::invoke(A, IP::invoke(args.base_case_table,index_pair), args.AstartX, args.AendX, args.AstartY, args.AendY);
  SP::invoke(IP::invoke(args.base_case_table,index_pair),args.base_case_blocked_table[index_pair],IP::invoke(args.base_case_cyclic_table,index_pair),std::forward<CommType>(CommInfo));

  if ((args.AendY == args.trueLocalDimension) && (args.trueLocalDimension*CommInfo.d-args.trueGlobalDimension != 0)){
    U checkDim = args.localDimension*CommInfo.d;
    U finalDim = (checkDim - (args.trueLocalDimension*CommInfo.d - args.trueGlobalDimension));
    std::vector<T> deepBaseCase(finalDim*finalDim,0);
    for (U i=0; i<finalDim; i++){
      for (U j=0; j<finalDim; j++){
        deepBaseCase[i*finalDim+j] = IP::invoke(args.base_case_cyclic_table,index_pair).data()[i*checkDim+j];
      }
    }
    lapack::ArgPack_potrf potrfArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper);
    lapack::ArgPack_trtri trtriArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper, lapack::Diag::AlapackNonUnit);
    lapack::engine::_potrf(&deepBaseCase[0],finalDim,finalDim,potrfArgs);
    std::vector<T> deepBaseCaseInv = deepBaseCase;              // true copy because we have to, unless we want to iterate (see below) two different times
    lapack::engine::_trtri(&deepBaseCaseInv[0],finalDim,finalDim,trtriArgs);
    std::vector<T> deepBaseCaseFill(checkDim*checkDim,0);
    std::vector<T> deepBaseCaseInvFill(checkDim*checkDim,0);
    for (U i=0; i<finalDim; i++){
      for (U j=0; j<finalDim; j++){
        deepBaseCaseFill[i*checkDim+j] = deepBaseCase[i*finalDim+j];
        deepBaseCaseInvFill[i*checkDim+j] = deepBaseCaseInv[i*finalDim+j];
      }
    }
    util::cyclic_to_local(&deepBaseCaseFill[0], &deepBaseCaseInvFill[0], args.localDimension, args.globalDimension, args.bcDimension, CommInfo.d, rankSlice);
    matrix<T,U,rect,Offload> tempMat(&deepBaseCaseFill[0], args.localDimension, args.localDimension, CommInfo.d, CommInfo.d);
    matrix<T,U,rect,Offload> tempMatInv(&deepBaseCaseInvFill[0], args.localDimension, args.localDimension, CommInfo.d, CommInfo.d);
    serialize<rect,rect>::invoke(A, tempMat, args.AstartY, args.AendY, args.AstartY, args.AendY, true);
    serialize<rect,rect>::invoke(TI, tempMatInv, args.TIstartX, args.TIendX, args.TIstartY, args.TIendY, true);
  }
  else{
    U fTranDim1 = args.localDimension*CommInfo.d;
    // Until then, assume a double datatype and simply use LAPACKE_dpotrf. Worry about adding more capabilities later.
    lapack::ArgPack_potrf potrfArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper);
    lapack::ArgPack_trtri trtriArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper, lapack::Diag::AlapackNonUnit);
    lapack::engine::_potrf(IP::invoke(args.base_case_cyclic_table,index_pair).data(),fTranDim1,fTranDim1,potrfArgs);
    std::memcpy(IP::invoke(args.base_case_cyclic_table,index_pair).scratch(),IP::invoke(args.base_case_cyclic_table,index_pair).data(),sizeof(T)*IP::invoke(args.base_case_cyclic_table,index_pair).num_elems());
    lapack::engine::_trtri(IP::invoke(args.base_case_cyclic_table,index_pair).scratch(),fTranDim1,fTranDim1,trtriArgs);
    util::cyclic_to_local(IP::invoke(args.base_case_cyclic_table,index_pair).data(),IP::invoke(args.base_case_cyclic_table,index_pair).scratch(), args.localDimension, args.globalDimension, args.bcDimension, CommInfo.d,rankSlice);
    serialize<rect,rect>::invoke(A, IP::invoke(args.base_case_cyclic_table,index_pair), args.AstartY, args.AendY, args.AstartY, args.AendY, true);
    IP::invoke(args.base_case_cyclic_table,index_pair).swap();	// puts the inverse buffer into the `data` member before final serialization
    serialize<rect,rect>::invoke(TI, IP::invoke(args.base_case_cyclic_table,index_pair), args.TIstartX, args.TIendX, args.TIstartY, args.TIendY, true);
    IP::invoke(args.base_case_cyclic_table,index_pair).swap();	// puts the inverse buffer into the `data` member before final serialization
  }
  IP::flush(args.base_case_table[index_pair]); IP::flush(args.base_case_cyclic_table[index_pair]);
  return;
}
}
