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
  auto bcMult = args.bc_mult_dim; if (bcMult<0){ bcMult *= (-1); for (int i=0;i<bcMult; i++) bcDimLocal/=2;}
  else{for (int i=0;i<bcMult; i++) bcDimLocal*=2;}
  bcDimLocal  = std::max(minDimLocal,bcDimLocal); bcDimLocal  = std::min(localDimension,bcDimLocal); U bcDimension = CommInfo.d*bcDimLocal;

  simulate(std::forward<ArgType>(args), localDimension, localDimension, bcDimension, globalDimension, globalDimension,
           0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, args.complete_inv, std::forward<CommType>(CommInfo));
  factor(A, TI, std::forward<ArgType>(args), localDimension, localDimension, bcDimension, globalDimension, globalDimension,
         0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, args.complete_inv, std::forward<CommType>(CommInfo));
}

template<class SerializePolicy, class IntermediatesPolicy, class OverlapPolicy>
template<typename T, typename U, typename ArgType, typename CommType>
std::pair<T*,T*> cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>::invoke(T* A, T* TI, U localDim, U globalDim, ArgType&& args, CommType&& CommInfo){
  //TODO: Test with non-power-of-2 global matrix dimensions
  matrix<T,U,rect,cyclic> mA(A,localDim,localDim,globalDim,globalDim,CommInfo.c,CommInfo.c);	// re-used in SquareTable
  matrix<T,U,rect,cyclic> mTI(R,localDim,localDim,globalDim,globalDim,CommInfo.c,CommInfo.c);
  invoke(mA,mTI,std::forward<ArgType>(args),std::forward<CommType>(CommInfo));
  return std::make_pair(mA.get_data(),mTI.get_data());
}

template<class SerializePolicy, class IntermediatesPolicy, class OverlapPolicy>
template<typename ArgType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>::simulate(ArgType&& args, int64_t localDimension, int64_t trueLocalDimension, int64_t bcDimension, int64_t globalDimension, int64_t trueGlobalDimension,
                                                                          int64_t AstartX, int64_t AendX, int64_t AstartY, int64_t AendY, int64_t RIstartX, int64_t RIendX, int64_t RIstartY, int64_t RIendY, bool complete_inv, CommType&& CommInfo){

  using SP = SerializePolicy; using IP = IntermediatesPolicy; using OP = OverlapPolicy;
  using U = int64_t;
  if ((localDimension*CommInfo.d) <= bcDimension){
    simulate_basecase(std::forward<ArgType>(args), localDimension, trueLocalDimension, bcDimension, globalDimension, trueGlobalDimension,
                      AstartX, AendX, AstartY, AendY, RIstartX, RIendX, RIstartY, RIendY, std::forward<CommType>(CommInfo));
    return;
  }

  U cut1 = (localDimension>>1); cut1 = util::get_next_power2(cut1); U cut2 = localDimension-cut1;
  simulate(std::forward<ArgType>(args), cut1, trueLocalDimension, bcDimension, (globalDimension>>1), trueGlobalDimension,
           AstartX, AstartX+cut1, AstartY, AstartY+cut1, RIstartX, RIstartX+cut1, RIstartY, RIstartY+cut1, complete_inv, std::forward<CommType>(CommInfo));

  IP::init(args.policy_table,std::make_pair(cut1,cut1),nullptr,cut1,cut1,CommInfo.d,CommInfo.d);
  IP::init(args.square_table1,std::make_pair(cut2,cut1),nullptr,cut2,cut1,CommInfo.d,CommInfo.d);
  IP::init(args.square_table2,std::make_pair(cut2,cut2),nullptr,cut2,cut2,CommInfo.d,CommInfo.d);	// this might be a problem. It was coupled before
  simulate(std::forward<ArgType>(args), cut2, trueLocalDimension, bcDimension, cut2*CommInfo.d, trueGlobalDimension, AstartX+cut1, AendX, AstartY+cut1, AendY,
           RIstartX+cut1, RIendX, RIstartY+cut1, RIendY, complete_inv, std::forward<CommType>(CommInfo));

  if (!(!complete_inv && (globalDimension==trueGlobalDimension))){
    IP::init(args.policy_table,std::make_pair(cut1,cut1),nullptr,cut1,cut1,CommInfo.d,CommInfo.d);
    IP::init(args.policy_table,std::make_pair(cut2,cut2),nullptr,cut2,cut2,CommInfo.d,CommInfo.d);
  }
}

template<class SerializePolicy, class IntermediatesPolicy, class OverlapPolicy>
template<typename ArgType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>::simulate_basecase(ArgType&& args, int64_t localDimension, int64_t trueLocalDimension, int64_t bcDimension, int64_t globalDimension, int64_t trueGlobalDimension,
                                                                                   int64_t AstartX, int64_t AendX, int64_t AstartY, int64_t AendY, int64_t matIstartX, int64_t matIendX, int64_t matIstartY, int64_t matIendY, CommType&& CommInfo){

  using SP = SerializePolicy; using IP = IntermediatesPolicy; using OP = OverlapPolicy; using U = int64_t;
  if (localDimension==0) return;
  IP::init(args.base_case_table, std::make_pair(AendX-AstartX,AendY-AstartY), nullptr,AendX-AstartX,AendY-AstartY,CommInfo.d,CommInfo.d);
  U num_elems = args.base_case_table[std::make_pair(AendX-AstartX,AendY-AstartY)].num_elems()*CommInfo.d*CommInfo.d;
  IP::init(args.base_case_blocked_table,std::make_pair(AendX-AstartX,AendY-AstartY), num_elems);
  IP::init(args.base_case_cyclic_table, std::make_pair(AendX-AstartX,AendY-AstartY),nullptr,bcDimension,bcDimension,CommInfo.d,CommInfo.d);
}

template<class SerializePolicy, class IntermediatesPolicy, class OverlapPolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>::factor(MatrixType& A, MatrixType& RI, ArgType&& args, typename MatrixType::DimensionType localDimension, typename MatrixType::DimensionType trueLocalDimension,
                                                                        typename MatrixType::DimensionType bcDimension, typename MatrixType::DimensionType globalDimension, typename MatrixType::DimensionType trueGlobalDimension,
                                                                        typename MatrixType::DimensionType AstartX, typename MatrixType::DimensionType AendX, typename MatrixType::DimensionType AstartY,
                                                                        typename MatrixType::DimensionType AendY, typename MatrixType::DimensionType RIstartX, typename MatrixType::DimensionType RIendX,
                                                                        typename MatrixType::DimensionType RIstartY, typename MatrixType::DimensionType RIendY, bool complete_inv, CommType&& CommInfo){

  using SP = SerializePolicy; using IP = IntermediatesPolicy; using OP = OverlapPolicy;
  using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType; using Offload = typename MatrixType::OffloadType;
  if ((localDimension*CommInfo.d) <= bcDimension){
    base_case(A, RI, std::forward<ArgType>(args), localDimension, trueLocalDimension, bcDimension, globalDimension, trueGlobalDimension,
              AstartX, AendX, AstartY, AendY, RIstartX, RIendX, RIstartY, RIendY, std::forward<CommType>(CommInfo));
    return;
  }

  U cut1 = (localDimension>>1); cut1 = util::get_next_power2(cut1); U cut2 = localDimension-cut1;
  factor(A, RI, std::forward<ArgType>(args), cut1, trueLocalDimension, bcDimension, (globalDimension>>1), trueGlobalDimension, AstartX, AstartX+cut1, AstartY,
         AstartY+cut1, RIstartX, RIstartX+cut1, RIstartY, RIstartY+cut1, complete_inv, std::forward<CommType>(CommInfo));

  serialize<rect,typename SP::structure>::invoke(RI, IP::invoke(args.policy_table,std::make_pair(cut1,cut1)), RIstartX, RIstartX+cut1, RIstartY, RIstartY+cut1);
  util::transpose(IP::invoke(args.policy_table,std::make_pair(cut1,cut1)), std::forward<CommType>(CommInfo));
  blas::ArgPack_trmm<T> trmmArgs(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper, blas::Transpose::AblasTrans, blas::Diag::AblasNonUnit, 1.);

  // 2nd case: Extra optimization for the case when we only perform TRSM at the top level.
  serialize<rect,rect>::invoke(A, IP::invoke(args.square_table1,std::make_pair(cut2,cut1)), AstartX+cut1, AendX, AstartY, AstartY+cut1);
  matmult::summa::invoke(IP::invoke(args.policy_table,std::make_pair(cut1,cut1)), IP::invoke(args.square_table1,std::make_pair(cut2,cut1)), std::forward<CommType>(CommInfo), trmmArgs);
  serialize<rect,rect>::invoke(A, IP::invoke(args.square_table1,std::make_pair(cut2,cut1)), AstartX+cut1, AendX, AstartY, AstartY+cut1, true);

  blas::ArgPack_syrk<T> syrkArgs(blas::Order::AblasColumnMajor, blas::UpLo::AblasUpper, blas::Transpose::AblasTrans, -1., 1.);
  serialize<rect,rect>::invoke(A, IP::invoke(args.square_table1,std::make_pair(cut2,cut1)), AstartX+cut1, AendX, AstartY, AstartY+cut1);
  serialize<rect,rect>::invoke(A, IP::invoke(args.square_table2,std::make_pair(cut2,cut2)), AstartX+cut1, AendX, AstartY+cut1, AendY);
  matmult::summa::invoke(IP::invoke(args.square_table1,std::make_pair(cut2,cut1)), IP::invoke(args.square_table2,std::make_pair(cut2,cut2)), std::forward<CommType>(CommInfo), syrkArgs);
  serialize<rect,rect>::invoke(A, IP::invoke(args.square_table2,std::make_pair(cut2,cut2)), AstartX+cut1, AendX, AstartY+cut1, AendY, true);

  factor(A, RI, std::forward<ArgType>(args), cut2, trueLocalDimension, bcDimension, cut2*CommInfo.d, trueGlobalDimension, AstartX+cut1, AendX, AstartY+cut1, AendY,
         RIstartX+cut1, RIendX, RIstartY+cut1, RIendY, complete_inv, std::forward<CommType>(CommInfo));

  // Next step : temp <- R_{12}*RI_{22}
  if (!(!complete_inv && (globalDimension==trueGlobalDimension))){
    serialize<rect,rect>::invoke(A, IP::invoke(args.square_table1,std::make_pair(cut2,cut1)), AstartX+cut1, AendX, AstartY, AstartY+cut1);
    serialize<rect,typename SP::structure>::invoke(RI, IP::invoke(args.policy_table,std::make_pair(cut1,cut1)), RIstartX, RIstartX+cut1, RIstartY, RIstartY+cut1);
    blas::ArgPack_trmm<T> invPackage1(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper, blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    matmult::summa::invoke(IP::invoke(args.policy_table,std::make_pair(cut1,cut1)), IP::invoke(args.square_table1,std::make_pair(cut2,cut1)), std::forward<CommType>(CommInfo), invPackage1);
    // Next step: finish the Triangular inverse calculation
    invPackage1.alpha = -1.; invPackage1.side = blas::Side::AblasRight;
    serialize<rect,typename SP::structure>::invoke(RI, IP::invoke(args.policy_table,std::make_pair(cut2,cut2)), RIstartX+cut1, RIendX, RIstartY+cut1, RIendY);
    matmult::summa::invoke(IP::invoke(args.policy_table,std::make_pair(cut2,cut2)), IP::invoke(args.square_table1,std::make_pair(cut2,cut1)), std::forward<CommType>(CommInfo), invPackage1);
    serialize<rect,rect>::invoke(RI, IP::invoke(args.square_table1,std::make_pair(cut2,cut1)), RIstartX+cut1, RIendX, RIstartY, RIstartY+cut1, true);
  }
  IP::flush(args.square_table1[std::make_pair(cut2,cut1)]); IP::flush(args.policy_table_diaginv[std::make_pair(cut1,cut1)]); IP::flush(args.policy_table[std::make_pair(cut1,cut1)]);
  IP::flush(args.square_table2[std::make_pair(cut2,cut2)]); IP::flush(args.policy_table[std::make_pair(cut2,cut2)]);
}


template<class SerializePolicy, class IntermediatesPolicy, class OverlapPolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>::base_case(MatrixType& A, MatrixType& I, ArgType&& args, typename MatrixType::DimensionType localDimension,
                                                                           typename MatrixType::DimensionType trueLocalDimension, typename MatrixType::DimensionType bcDimension,
                                                                           typename MatrixType::DimensionType globalDimension, typename MatrixType::DimensionType trueGlobalDimension,
                                                                           typename MatrixType::DimensionType AstartX, typename MatrixType::DimensionType AendX, typename MatrixType::DimensionType AstartY,
                                                                           typename MatrixType::DimensionType AendY, typename MatrixType::DimensionType matIstartX, typename MatrixType::DimensionType matIendX,
                                                                           typename MatrixType::DimensionType matIstartY, typename MatrixType::DimensionType matIendY, CommType&& CommInfo){

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

  if (localDimension==0) return;
  int rankSlice; MPI_Comm_rank(CommInfo.slice, &rankSlice);
  auto index_pair = std::make_pair(AendX-AstartX,AendY-AstartY);
  serialize<rect,typename SP::structure>::invoke(A, IP::invoke(args.base_case_table,index_pair), AstartX, AendX, AstartY, AendY);
  SP::invoke(IP::invoke(args.base_case_table,index_pair),args.base_case_blocked_table[index_pair],IP::invoke(args.base_case_cyclic_table,index_pair),std::forward<CommType>(CommInfo));

  if ((AendY == trueLocalDimension) && (trueLocalDimension*CommInfo.d - trueGlobalDimension != 0)){
    U checkDim = localDimension*CommInfo.d;
    U finalDim = (checkDim - (trueLocalDimension*CommInfo.d - trueGlobalDimension));
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
    util::cyclic_to_local(&deepBaseCaseFill[0], &deepBaseCaseInvFill[0], localDimension, globalDimension, bcDimension, CommInfo.d, rankSlice);
    matrix<T,U,rect,Offload> tempMat(&deepBaseCaseFill[0], localDimension, localDimension, CommInfo.d, CommInfo.d);
    matrix<T,U,rect,Offload> tempMatInv(&deepBaseCaseInvFill[0], localDimension, localDimension, CommInfo.d, CommInfo.d);
    serialize<rect,rect>::invoke(A, tempMat, AstartY, AendY, AstartY, AendY, true);
    serialize<rect,rect>::invoke(I, tempMatInv, matIstartX, matIendX, matIstartY, matIendY, true);
  }
  else{
    auto index_pair = std::make_pair(AendX-AstartX,AendY-AstartY); U fTranDim1 = localDimension*CommInfo.d;
    // Until then, assume a double datatype and simply use LAPACKE_dpotrf. Worry about adding more capabilities later.
    lapack::ArgPack_potrf potrfArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper);
    lapack::ArgPack_trtri trtriArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper, lapack::Diag::AlapackNonUnit);
    lapack::engine::_potrf(IP::invoke(args.base_case_cyclic_table,index_pair).data(),fTranDim1,fTranDim1,potrfArgs);
    std::memcpy(IP::invoke(args.base_case_cyclic_table,index_pair).scratch(),IP::invoke(args.base_case_cyclic_table,index_pair).data(),sizeof(T)*IP::invoke(args.base_case_cyclic_table,index_pair).num_elems());
    lapack::engine::_trtri(IP::invoke(args.base_case_cyclic_table,index_pair).scratch(),fTranDim1,fTranDim1,trtriArgs);
    util::cyclic_to_local(IP::invoke(args.base_case_cyclic_table,index_pair).data(),IP::invoke(args.base_case_cyclic_table,index_pair).scratch(), localDimension, globalDimension, bcDimension, CommInfo.d,rankSlice);
    serialize<rect,rect>::invoke(A, IP::invoke(args.base_case_cyclic_table,index_pair), AstartY, AendY, AstartY, AendY, true);
    IP::invoke(args.base_case_cyclic_table,index_pair).swap();	// puts the inverse buffer into the `data` member before final serialization
    serialize<rect,rect>::invoke(I, IP::invoke(args.base_case_cyclic_table,index_pair), matIstartX, matIendX, matIstartY, matIendY, true);
    IP::invoke(args.base_case_cyclic_table,index_pair).swap();	// puts the inverse buffer into the `data` member before final serialization
  }
  IP::flush(args.base_case_table[index_pair]); IP::flush(args.base_case_cyclic_table[index_pair]);
  return;
}
}
