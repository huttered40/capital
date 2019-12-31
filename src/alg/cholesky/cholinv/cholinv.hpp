/* Author: Edward Hutter */

namespace cholesky{
template<class SerializePolicy, class IntermediatesPolicy, class OverlapPolicy>
template<typename MatrixAType, typename MatrixTIType, typename ArgType, typename CommType>
std::pair<bool,std::vector<typename MatrixAType::DimensionType>>
cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>::invoke(MatrixAType& A, MatrixTIType& TI, ArgType&& args, CommType&& CommInfo){

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using Structure = typename MatrixAType::StructureType;
  using Distribution = typename MatrixAType::DistributionType;
  using Offload = typename MatrixAType::OffloadType;
  assert(args.dir == 'U');	// Removed support for 'L'. Necessary future support for this case can be handled via a final transpose.
  U localDimension = A.num_rows_local();
  U globalDimension = A.num_rows_global();
  U minDimLocal = 1;
  U bcDimLocal = util::get_next_power2(localDimension/(CommInfo.c*CommInfo.d));
  auto bcMult = args.bc_mult_dim;
  if (bcMult<0){ bcMult *= (-1); for (int i=0;i<bcMult; i++) bcDimLocal/=2;}
  else{for (int i=0;i<bcMult; i++) bcDimLocal*=2;}
  bcDimLocal  = std::max(minDimLocal,bcDimLocal);	// min prevents recursing into a 0x0 local matrix
  U bcDimension = CommInfo.d*bcDimLocal;

  U save = globalDimension;
  for (size_t i=0; i<args.inv_cut_off_dim; i++){
    save >>= 1;
  }
  save = std::max(localDimension*2,save);
  std::pair<bool,std::vector<int64_t>> baseCaseDimList;

  // Pre-allocate recursive matrix<> instances for intermediate (non-base-case) matrix multiplications
  std::map<std::pair<U,U>,matrix<T,U,typename SerializePolicy::structure,Distribution,Offload>> policy_table;
  std::map<std::pair<U,U>,matrix<T,U,typename SerializePolicy::structure,Distribution,Offload>> policy_table_diaginv;
  std::map<std::pair<U,U>,matrix<T,U,Structure,Distribution,Offload>> square_table1;// assume Structure == rect or square
  std::map<std::pair<U,U>,matrix<T,U,Structure,Distribution,Offload>> square_table2;// assume Structure == rect or square
  std::map<std::pair<U,U>,matrix<T,U,typename SerializePolicy::structure,Distribution,Offload>> base_case_table;	// assume Structure == rect or square
  std::map<std::pair<U,U>,std::vector<T>> base_case_blocked_table;	// assume Structure == rect or square
  std::map<std::pair<U,U>,matrix<T,U,Structure,Distribution,Offload>> base_case_cyclic_table;	// assume Structure == rect or square

  baseCaseDimList.first = (save >= globalDimension ? true : false);
  simulate(policy_table, policy_table_diaginv, square_table1, square_table2, base_case_table, base_case_blocked_table, base_case_cyclic_table, localDimension, localDimension, bcDimension, globalDimension, globalDimension,
           0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, std::forward<CommType>(CommInfo), baseCaseDimList.first, baseCaseDimList.second, args.inv_cut_off_dim);

  baseCaseDimList.first = (save >= globalDimension ? true : false); baseCaseDimList.second.clear();
  factor(A, TI, policy_table, policy_table_diaginv, square_table1, square_table2, base_case_table, base_case_blocked_table, base_case_cyclic_table, localDimension, localDimension, bcDimension, globalDimension, globalDimension,
         0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, std::forward<CommType>(CommInfo), baseCaseDimList.first, baseCaseDimList.second, args.inv_cut_off_dim);
  return baseCaseDimList;
}

//TODO: Notice how this routine does not pass back a list of integers like the other invoke method. Should this be supported?
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
template<typename PolicyTableType, typename SquareTableType, typename BaseCaseTableType, typename BaseCaseBlockedTableType, typename BaseCaseCyclicTableType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>::simulate(
                           PolicyTableType& policy_table, PolicyTableType& policy_table_diaginvert, SquareTableType& square_table1, SquareTableType& square_table2,
                           BaseCaseTableType& base_case_table, BaseCaseBlockedTableType& base_case_blocked_table, BaseCaseCyclicTableType& base_case_cyclic_table,
                           int64_t localDimension, int64_t trueLocalDimension, int64_t bcDimension, int64_t globalDimension, int64_t trueGlobalDimension,
                           int64_t AstartX, int64_t AendX, int64_t AstartY, int64_t AendY, int64_t RIstartX, int64_t RIendX, int64_t RIstartY, int64_t RIendY,
                           CommType&& CommInfo, bool& isInversePath, std::vector<int64_t>& baseCaseDimList, int64_t inverseCutoffGlobalDimension){

  if (globalDimension <= bcDimension){
    simulate_basecase(base_case_table, base_case_blocked_table, base_case_cyclic_table, localDimension, trueLocalDimension, bcDimension, globalDimension, trueGlobalDimension,
                      AstartX, AendX, AstartY, AendY, RIstartX, RIendX, RIstartY, RIendY, std::forward<CommType>(CommInfo), isInversePath, baseCaseDimList, inverseCutoffGlobalDimension, 'U');
    return;
  }

  using U = int64_t;

  U localShift = (localDimension>>1);
  localShift = util::get_next_power2(localShift);
  U globalShift = (globalDimension>>1);
  U reverseDimLocal = localDimension-localShift;
  U reverseDimGlobal = reverseDimLocal*CommInfo.d;
  bool saveSwitch = isInversePath;
  size_t saveIndexPrev = baseCaseDimList.size();

  update_inverse_path_simulate(inverseCutoffGlobalDimension, globalDimension, isInversePath, localDimension);
  simulate(policy_table, policy_table_diaginvert, square_table1, square_table2, base_case_table, base_case_blocked_table, base_case_cyclic_table, localShift, trueLocalDimension, bcDimension, globalShift, trueGlobalDimension,
    AstartX, AstartX+localShift, AstartY, AstartY+localShift, RIstartX, RIstartX+localShift, RIstartY, RIstartY+localShift, std::forward<CommType>(CommInfo), isInversePath, baseCaseDimList, inverseCutoffGlobalDimension);
  size_t saveIndexAfter = baseCaseDimList.size();

  if (policy_table.find(std::make_pair(localShift,localShift)) == policy_table.end()){
    policy_table.emplace(std::piecewise_construct,std::forward_as_tuple(std::make_pair(localShift,localShift)),std::forward_as_tuple(nullptr,localShift,localShift,CommInfo.d,CommInfo.d));
  }
  if (!((isInversePath) || (globalDimension == inverseCutoffGlobalDimension*2))){
    // create a new subvector
    U len = saveIndexAfter - saveIndexPrev;
    std::vector<U> subBaseCaseDimList(len);
    for (U i=saveIndexPrev; i<saveIndexAfter; i++){
      subBaseCaseDimList[i-saveIndexPrev] = baseCaseDimList[i];
    }
    if (policy_table_diaginvert.find(std::make_pair(localShift,localShift)) == policy_table_diaginvert.end()){
      policy_table_diaginvert.emplace(std::piecewise_construct,std::forward_as_tuple(std::make_pair(localShift,localShift)),std::forward_as_tuple(nullptr,localShift,localShift,CommInfo.d,CommInfo.d));
    } //TODO: Order of first two integers below might be switched
    simulate_solve(square_table1, square_table2, base_case_cyclic_table, std::forward<CommType>(CommInfo), localShift, reverseDimLocal, localShift, subBaseCaseDimList);
  }
  
  if (square_table1.find(std::make_pair(reverseDimLocal,localShift)) == square_table1.end()){
    square_table1.emplace(std::piecewise_construct,std::forward_as_tuple(std::make_pair(reverseDimLocal,localShift)),std::forward_as_tuple(nullptr,reverseDimLocal,localShift,CommInfo.d,CommInfo.d));
    square_table2.emplace(std::piecewise_construct,std::forward_as_tuple(std::make_pair(reverseDimLocal,localShift)),std::forward_as_tuple(nullptr,reverseDimLocal,reverseDimLocal,CommInfo.d,CommInfo.d));
  }
  simulate(policy_table, policy_table_diaginvert, square_table1, square_table2, base_case_table, base_case_blocked_table, base_case_cyclic_table,
           reverseDimLocal, trueLocalDimension, bcDimension, reverseDimGlobal, trueGlobalDimension,
           AstartX+localShift, AendX, AstartY+localShift, AendY, RIstartX+localShift, RIendX, RIstartY+localShift, RIendY, std::forward<CommType>(CommInfo), isInversePath, baseCaseDimList, inverseCutoffGlobalDimension);

  if (isInversePath){
    if (policy_table.find(std::make_pair(localShift,localShift)) == policy_table.end()){
      policy_table.emplace(std::piecewise_construct,std::forward_as_tuple(std::make_pair(localShift,localShift)),std::forward_as_tuple(nullptr,localShift,localShift,CommInfo.d,CommInfo.d));
    }
    if (policy_table.find(std::make_pair(reverseDimLocal,reverseDimLocal)) == policy_table.end()){
      policy_table.emplace(std::piecewise_construct,std::forward_as_tuple(std::make_pair(reverseDimLocal,reverseDimLocal)),std::forward_as_tuple(nullptr,reverseDimLocal,reverseDimLocal,CommInfo.d,CommInfo.d));
    }
  }
  isInversePath = saveSwitch;
}

template<class SerializePolicy, class IntermediatesPolicy, class OverlapPolicy>
template<typename BaseCaseTableType, typename BaseCaseBlockedTableType, typename BaseCaseCyclicTableType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>::simulate_basecase(
                     BaseCaseTableType& base_case_table, BaseCaseBlockedTableType& base_case_blocked_table, BaseCaseCyclicTableType& base_case_cyclic_table,
                     int64_t localDimension, int64_t trueLocalDimension, int64_t bcDimension, int64_t globalDimension, int64_t trueGlobalDimension,
                     int64_t AstartX, int64_t AendX, int64_t AstartY, int64_t AendY, int64_t matIstartX, int64_t matIendX, int64_t matIstartY, int64_t matIendY,
                     CommType&& CommInfo, bool& isInversePath, std::vector<int64_t>& baseCaseDimList, int64_t inverseCutoffGlobalDimension, char dir){

  using U = int64_t;
  assert(localDimension>0);
  if (!isInversePath){
    baseCaseDimList.push_back(localDimension);
  }
  if (base_case_table.find(std::make_pair(AendX-AstartX,AendY-AstartY)) == base_case_table.end()){
    base_case_table.emplace(std::piecewise_construct,std::forward_as_tuple(std::make_pair(AendX-AstartX,AendY-AstartY)),std::forward_as_tuple(nullptr,AendX-AstartX,AendY-AstartY,CommInfo.d,CommInfo.d));
    // create the blocked and cyclic vectors as well
    U aggregSize = base_case_table[std::make_pair(AendX-AstartX,AendY-AstartY)].num_elems()*CommInfo.d*CommInfo.d;
    base_case_blocked_table.emplace(std::piecewise_construct,std::forward_as_tuple(std::make_pair(AendX-AstartX,AendY-AstartY)),std::forward_as_tuple(aggregSize));
    //TODO: I'm skeptical that `bcDimension` is correct for non-power-of-2 global dimensions
    base_case_cyclic_table.emplace(std::piecewise_construct,std::forward_as_tuple(std::make_pair(AendX-AstartX,AendY-AstartY)),std::forward_as_tuple(nullptr,bcDimension,bcDimension,CommInfo.d,CommInfo.d));
  }
  return;
}

// For solving LA=B for A. But note that B is being modified in place and will turn into A
template<class SerializePolicy, class IntermediatesPolicy, class OverlapPolicy>
template<typename SquareTableType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>::simulate_solve(SquareTableType& square_table1, SquareTableType& square_table2, SquareTableType& square_table3, CommType&& CommInfo, int64_t num_cols_A,
                                                                     int64_t num_rows_A, int64_t num_cols_L, std::vector<int64_t>& baseCaseDimList){

  using U = int64_t;

  U AendX = num_cols_A; U AendY = num_rows_A; U matLendX = num_cols_L;
  U offset1 = 0; U offset2 = baseCaseDimList[0]; U offset3 = 0;
  // Note that the beginning cases might not be correct. They are not currently used for anything though.
  U arg1 = offset1; U arg2 = matLendX; U arg3 = offset3; U arg4 = offset1;

  for (U i=0; i<baseCaseDimList.size(); i++){
    // Only update once first panel is solved
    if (i>0){
      // As i increases, the size of these updates gets smaller.
      // Special handling. This might only work since the triangular matrix is square, which should be ok
      // Note that the beginning cases might not be correct. They are not currently used for anything though.
      arg1 = offset1; arg2 = matLendX; arg3 = offset3; arg4 = offset1;
      if (square_table1.find(std::make_pair(arg2-arg1,arg4-arg3)) == square_table1.end()){
        square_table1.emplace(std::piecewise_construct,std::forward_as_tuple(std::make_pair(arg2-arg1,arg4-arg3)),std::forward_as_tuple(nullptr,arg2-arg1,arg4-arg3,CommInfo.d,CommInfo.d));
      }
      if (square_table2.find(std::make_pair(AendX,offset1-offset3)) == square_table2.end()){
        square_table2.emplace(std::piecewise_construct,std::forward_as_tuple(std::make_pair(AendX,offset1-offset3)),std::forward_as_tuple(nullptr,AendX,offset1-offset3,CommInfo.d,CommInfo.d));
      }
      if (square_table3.find(std::make_pair(AendX,AendY-offset1)) == square_table3.end()){
        square_table3.emplace(std::piecewise_construct,std::forward_as_tuple(std::make_pair(AendX,AendY-offset1)),std::forward_as_tuple(nullptr,AendX,AendY-offset1,CommInfo.d,CommInfo.d));
      }
    }

    if (square_table1.find(std::make_pair(offset2-offset1,offset2-offset1)) == square_table1.end()){
      square_table1.emplace(std::piecewise_construct,std::forward_as_tuple(std::make_pair(offset2-offset1,offset2-offset1)),std::forward_as_tuple(nullptr,offset2-offset1,offset2-offset1,CommInfo.d,CommInfo.d));
    }
    if (square_table2.find(std::make_pair(AendX,offset2-offset1)) == square_table2.end()){
      square_table2.emplace(std::piecewise_construct,std::forward_as_tuple(std::make_pair(AendX,offset2-offset1)),std::forward_as_tuple(nullptr,AendX,offset2-offset1,CommInfo.d,CommInfo.d));
    }
    if ((i+1) < baseCaseDimList.size()){
      // Update the offsets
      offset3 = offset1;
      offset1 = offset2;
      offset2 += baseCaseDimList[i+1];
    }
  }
}

template<class SerializePolicy, class IntermediatesPolicy, class OverlapPolicy>
template<typename MatrixAType, typename MatrixRIType, typename PolicyTableType, typename SquareTableType, typename BaseCaseTableType, typename BaseCaseBlockedTableType, typename BaseCaseCyclicTableType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>::factor(
                           MatrixAType& A, MatrixRIType& RI, PolicyTableType& policy_table, PolicyTableType& policy_table_diaginvert,
                           SquareTableType& square_table1, SquareTableType& square_table2, BaseCaseTableType& base_case_table, BaseCaseBlockedTableType& base_case_blocked_table, BaseCaseCyclicTableType& base_case_cyclic_table,
                           typename MatrixAType::DimensionType localDimension, typename MatrixAType::DimensionType trueLocalDimension,
                           typename MatrixAType::DimensionType bcDimension, typename MatrixAType::DimensionType globalDimension, typename MatrixAType::DimensionType trueGlobalDimension,
                           typename MatrixAType::DimensionType AstartX, typename MatrixAType::DimensionType AendX, typename MatrixAType::DimensionType AstartY,
                           typename MatrixAType::DimensionType AendY, typename MatrixAType::DimensionType RIstartX, typename MatrixAType::DimensionType RIendX,
                           typename MatrixAType::DimensionType RIstartY, typename MatrixAType::DimensionType RIendY,
                           CommType&& CommInfo, bool& isInversePath, std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                           typename MatrixAType::DimensionType inverseCutoffGlobalDimension){

  if (globalDimension <= bcDimension){
    base_case(A, RI, base_case_table, base_case_blocked_table, base_case_cyclic_table, localDimension, trueLocalDimension, bcDimension, globalDimension, trueGlobalDimension,
             AstartX, AendX, AstartY, AendY, RIstartX, RIendX, RIstartY, RIendY, std::forward<CommType>(CommInfo), isInversePath, baseCaseDimList, inverseCutoffGlobalDimension, 'U');
    return;
  }

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using Distribution = typename MatrixAType::DistributionType;
  using Offload = typename MatrixAType::OffloadType;

  U localShift = (localDimension>>1);
  // move localShift up to the next power of 2
  localShift = util::get_next_power2(localShift);
  U globalShift = (globalDimension>>1);
  U reverseDimLocal = localDimension-localShift;
  U reverseDimGlobal = reverseDimLocal*CommInfo.d;
  bool saveSwitch = isInversePath;
  size_t saveIndexPrev = baseCaseDimList.size();

  update_inverse_path(inverseCutoffGlobalDimension, globalDimension, isInversePath, baseCaseDimList, localDimension);
  factor(A, RI, policy_table, policy_table_diaginvert, square_table1, square_table2, base_case_table, base_case_blocked_table, base_case_cyclic_table, localShift, trueLocalDimension, bcDimension, globalShift, trueGlobalDimension,
         AstartX, AstartX+localShift, AstartY, AstartY+localShift, RIstartX, RIstartX+localShift, RIstartY, RIstartY+localShift,
         std::forward<CommType>(CommInfo), isInversePath, baseCaseDimList, inverseCutoffGlobalDimension);

  size_t saveIndexAfter = baseCaseDimList.size();

  serialize<square,typename SerializePolicy::structure>::invoke(RI, policy_table[std::make_pair(localShift,localShift)], RIstartX, RIstartX+localShift, RIstartY, RIstartY+localShift);
  util::transpose(policy_table[std::make_pair(localShift,localShift)], std::forward<CommType>(CommInfo));
  blas::ArgPack_trmm<T> trmmArgs(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper, blas::Transpose::AblasTrans, blas::Diag::AblasNonUnit, 1.);

  // 2nd case: Extra optimization for the case when we only perform TRSM at the top level.
  if ((isInversePath) || (globalDimension == inverseCutoffGlobalDimension*2)){
    serialize<square,square>::invoke(A, square_table1[std::make_pair(reverseDimLocal,localShift)], AstartX+localShift, AendX, AstartY, AstartY+localShift);
    matmult::summa::invoke(policy_table[std::make_pair(localShift,localShift)], square_table1[std::make_pair(reverseDimLocal,localShift)], std::forward<CommType>(CommInfo), trmmArgs);
    serialize<square,square>::invoke(A, square_table1[std::make_pair(reverseDimLocal,localShift)], AstartX+localShift, AendX, AstartY, AstartY+localShift, true);
  }
  else{
    blas::ArgPack_gemm<T> trsmArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasTrans, blas::Transpose::AblasNoTrans, 1., 0.);
    // create a new subvector
    U len = saveIndexAfter - saveIndexPrev; std::vector<U> subBaseCaseDimList(len);
    for (U i=saveIndexPrev; i<saveIndexAfter; i++){
      subBaseCaseDimList[i-saveIndexPrev] = baseCaseDimList[i];
    }
    // make extra copy to avoid corrupting A
    // Future optimization: Copy a part of A into Acopy, to avoid excessing copying
    // Note: some of these globalShifts are wrong, but I don't know any easy way to fix them. Everything might still work though.
    serialize<square,square>::invoke(A, square_table1[std::make_pair(reverseDimLocal,localShift)], AstartX+localShift, AendX, AstartY, AstartY+localShift);
    // Also need to serialize top-left quadrant of L so that its size matches packedMatrix
    serialize<square,typename SerializePolicy::structure>::invoke(A, policy_table_diaginvert[std::make_pair(localShift,localShift)], AstartX, AstartX+localShift, AstartY, AstartY+localShift);
    // Swap, same as we did with inverse
    util::transpose(policy_table_diaginvert[std::make_pair(localShift,localShift)], std::forward<CommType>(CommInfo));
    blas::ArgPack_trmm<T> trmmPackage(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper,
      blas::Transpose::AblasTrans, blas::Diag::AblasNonUnit, 1.);
    solve(policy_table_diaginvert[std::make_pair(localShift,localShift)], policy_table[std::make_pair(localShift,localShift)], square_table1[std::make_pair(reverseDimLocal,localShift)],
          square_table1, square_table2, base_case_cyclic_table, std::forward<CommType>(CommInfo), subBaseCaseDimList, trsmArgs);
    // Inject back into R, TODO: Im skeptical that square_table1 is not correct, or was overwritten in solve
    serialize<square,square>::invoke(A, square_table1[std::make_pair(reverseDimLocal,localShift)], AstartX+localShift, AendX, AstartY, AstartY+localShift, true);
  }

  blas::ArgPack_syrk<T> syrkArgs(blas::Order::AblasColumnMajor, blas::UpLo::AblasUpper, blas::Transpose::AblasTrans, -1., 1.);
  serialize<square,square>::invoke(A, square_table1[std::make_pair(reverseDimLocal,localShift)], AstartX+localShift, AendX, AstartY, AstartY+localShift);
  serialize<square,square>::invoke(A, square_table2[std::make_pair(reverseDimLocal,localShift)], AstartX+localShift, AendX, AstartY+localShift, AendY);
  matmult::summa::invoke(square_table1[std::make_pair(reverseDimLocal,localShift)], square_table2[std::make_pair(reverseDimLocal,localShift)], std::forward<CommType>(CommInfo), syrkArgs);
  serialize<square,square>::invoke(A, square_table2[std::make_pair(reverseDimLocal,localShift)], AstartX+localShift, AendX, AstartY+localShift, AendY, true);

  factor(A, RI, policy_table, policy_table_diaginvert, square_table1, square_table2, base_case_table, base_case_blocked_table, base_case_cyclic_table, reverseDimLocal, trueLocalDimension, bcDimension, reverseDimGlobal, trueGlobalDimension,
         AstartX+localShift, AendX, AstartY+localShift, AendY, RIstartX+localShift, RIendX, RIstartY+localShift, RIendY,
         std::forward<CommType>(CommInfo), isInversePath, baseCaseDimList, inverseCutoffGlobalDimension);

  // Next step : temp <- R_{12}*RI_{22}
  if (isInversePath){
    serialize<square,square>::invoke(A, square_table1[std::make_pair(reverseDimLocal,localShift)], AstartX+localShift, AendX, AstartY, AstartY+localShift);
    serialize<square,typename SerializePolicy::structure>::invoke(RI, policy_table[std::make_pair(reverseDimLocal,reverseDimLocal)], RIstartX+localShift, RIendX, RIstartY+localShift, RIendY);
    blas::ArgPack_trmm<T> invPackage1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper, blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    matmult::summa::invoke(policy_table[std::make_pair(reverseDimLocal,reverseDimLocal)], square_table1[std::make_pair(reverseDimLocal,localShift)], std::forward<CommType>(CommInfo), invPackage1);
    // Next step: finish the Triangular inverse calculation
    invPackage1.alpha = -1.;
    invPackage1.side = blas::Side::AblasLeft;
    serialize<square,typename SerializePolicy::structure>::invoke(RI, policy_table[std::make_pair(localShift,localShift)], RIstartX, RIstartX+localShift, RIstartY, RIstartY+localShift);
    matmult::summa::invoke(policy_table[std::make_pair(localShift,localShift)], square_table1[std::make_pair(reverseDimLocal,localShift)], std::forward<CommType>(CommInfo), invPackage1);
    serialize<square,square>::invoke(RI, square_table1[std::make_pair(reverseDimLocal,localShift)], RIstartX+localShift, RIendX, RIstartY, RIstartY+localShift, true);
  }
  isInversePath = saveSwitch;
}


template<class SerializePolicy, class IntermediatesPolicy, class OverlapPolicy>
template<typename MatrixAType, typename MatrixIType, typename BaseCaseTableType, typename BaseCaseBlockedTableType, typename BaseCaseCyclicTableType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>::base_case(
                     MatrixAType& A, MatrixIType& I, BaseCaseTableType& base_case_table, BaseCaseBlockedTableType& base_case_blocked_table, BaseCaseCyclicTableType& base_case_cyclic_table,
                     typename MatrixAType::DimensionType localDimension, typename MatrixAType::DimensionType trueLocalDimension,
                     typename MatrixAType::DimensionType bcDimension, typename MatrixAType::DimensionType globalDimension, typename MatrixAType::DimensionType trueGlobalDimension,
                     typename MatrixAType::DimensionType AstartX, typename MatrixAType::DimensionType AendX, typename MatrixAType::DimensionType AstartY,
                     typename MatrixAType::DimensionType AendY, typename MatrixAType::DimensionType matIstartX, typename MatrixAType::DimensionType matIendX,
                     typename MatrixAType::DimensionType matIstartY, typename MatrixAType::DimensionType matIendY,
                     CommType&& CommInfo, bool& isInversePath, std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                     typename MatrixAType::DimensionType inverseCutoffGlobalDimension, char dir){

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using Distribution = typename MatrixAType::DistributionType;
  using Offload = typename MatrixAType::OffloadType;

  if (!isInversePath){
    baseCaseDimList.push_back(localDimension);
  }
  assert(localDimension>0);

  // No matter what path we are on, if we get into the base case, we will do regular Cholesky + Triangular inverse
  // First: AllGather matrix A so that every processor has the same replicated diagonal square partition of matrix A of dimension bcDimension
  //          Note that processors only want to communicate with those on their same 2D slice, since the matrices are replicated on every slice
  //          Note that before the AllGather, we need to serialize the matrix A into the small square matrix
  // Second: Data will be received in a blocked order due to AllGather semantics, which is not what we want. We need to get back to cyclic again
  //           This is an ugly process, as it was in the last code.
  // Third: Once data is in cyclic format, we call call sequential Cholesky Factorization and Triangular Inverse.
  // Fourth: Save the data that each processor owns according to the cyclic rule.

  int rankSlice;
  MPI_Comm_rank(CommInfo.slice, &rankSlice);

  auto index_pair = std::make_pair(AendX-AstartX,AendY-AstartY);
  serialize<square,typename SerializePolicy::structure>::invoke(A, base_case_table[index_pair], AstartX, AendX, AstartY, AendY);
  SerializePolicy::invoke(base_case_table[index_pair],base_case_blocked_table[index_pair],base_case_cyclic_table[index_pair].data(),std::forward<CommType>(CommInfo));

  if ((AendY == trueLocalDimension) && (trueLocalDimension*CommInfo.d - trueGlobalDimension != 0)){
    U checkDim = localDimension*CommInfo.d;
    U finalDim = (checkDim - (trueLocalDimension*CommInfo.d - trueGlobalDimension));
    std::vector<T> deepBaseCase(finalDim*finalDim,0);
    for (U i=0; i<finalDim; i++){
      for (U j=0; j<finalDim; j++){
        deepBaseCase[i*finalDim+j] = base_case_cyclic_table[index_pair].data()[i*checkDim+j];
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
    cyclic_to_local(&deepBaseCaseFill[0], &deepBaseCaseInvFill[0], localDimension, globalDimension, bcDimension, CommInfo.d, rankSlice, dir);
    matrix<T,U,square,Distribution,Offload> tempMat(&deepBaseCaseFill[0], localDimension, localDimension, CommInfo.d, CommInfo.d);
    matrix<T,U,square,Distribution,Offload> tempMatInv(&deepBaseCaseInvFill[0], localDimension, localDimension, CommInfo.d, CommInfo.d);
    serialize<square,square>::invoke(A, tempMat, AstartY, AendY, AstartY, AendY, true);
    serialize<square,square>::invoke(I, tempMatInv, matIstartX, matIendX, matIstartY, matIendY, true);
  }
  else{
    auto index_pair = std::make_pair(AendX-AstartX,AendY-AstartY);
    U fTranDim1 = localDimension*CommInfo.d;
    // Until then, assume a double datatype and simply use LAPACKE_dpotrf. Worry about adding more capabilities later.
    lapack::ArgPack_potrf potrfArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper);
    lapack::ArgPack_trtri trtriArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper, lapack::Diag::AlapackNonUnit);
    lapack::engine::_potrf(base_case_cyclic_table[index_pair].data(),fTranDim1,fTranDim1,potrfArgs);
    std::memcpy(base_case_cyclic_table[index_pair].scratch(),base_case_cyclic_table[index_pair].data(),sizeof(T)*base_case_cyclic_table[index_pair].num_elems());
    lapack::engine::_trtri(base_case_cyclic_table[index_pair].scratch(),fTranDim1,fTranDim1,trtriArgs);
    cyclic_to_local(base_case_cyclic_table[index_pair].data(),base_case_cyclic_table[index_pair].scratch(), localDimension, globalDimension, bcDimension, CommInfo.d,rankSlice,dir);
    serialize<square,square>::invoke(A, base_case_cyclic_table[index_pair], AstartY, AendY, AstartY, AendY, true);
    base_case_cyclic_table[index_pair].swap();	// puts the inverse buffer into the `data` member before final serialization
    serialize<square,square>::invoke(I, base_case_cyclic_table[index_pair], matIstartX, matIendX, matIstartY, matIendY, true);
    base_case_cyclic_table[index_pair].swap();	// puts the inverse buffer into the `data` member before final serialization
  }
  return;
}


// This method can be called from Lower and Upper with one tweak, but note that currently, we iterate over the entire square,
//   when we are really only writing to a triangle. So there is a source of optimization here at least in terms of
//   number of flops, but in terms of memory accesses and cache lines, not sure. Note that with this optimization,
//   we may need to separate into two different functions
template<class SerializePolicy, class IntermediatesPolicy, class OverlapPolicy>
template<typename T, typename U>
void cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>::cyclic_to_local(
                 T* storeT, T* storeTI, U localDimension, U globalDimension, U bcDimension, int64_t sliceDim, int64_t rankSlice, char dir){

  U writeIndex = 0;
  U rowOffsetWithinBlock = rankSlice / sliceDim;
  U columnOffsetWithinBlock = rankSlice % sliceDim;
  U numCyclicBlocksPerRowCol = localDimension/*bcDimension/sliceDim*/;
  // modify bcDimension
  bcDimension = localDimension*sliceDim;
  // MACRO loop over all cyclic "blocks"
  for (U i=0; i<numCyclicBlocksPerRowCol; i++){
    // We know which row corresponds to our processor in each cyclic "block"
    // Inner loop over all cyclic "blocks" partitioning up the columns
    // Future improvement: only need to iterate over lower triangular.
    for (U j=0; j<numCyclicBlocksPerRowCol; j++){
      // We know which column corresponds to our processor in each cyclic "block"
      // Future improvement: get rid of the inner if statement and separate out this inner loop into 2 loops
      // Further improvement: use only triangular matrices and then invoke into a square later?
      U readIndexCol = i*sliceDim + columnOffsetWithinBlock;
      U readIndexRow = j*sliceDim + rowOffsetWithinBlock;
      if ((dir == 'U') && (readIndexCol >= readIndexRow)){
        storeT[writeIndex] = storeT[readIndexCol*bcDimension + readIndexRow];
        storeTI[writeIndex] = storeTI[readIndexCol*bcDimension + readIndexRow];
      }
      else{
        storeT[writeIndex] = 0.;
        storeTI[writeIndex] = 0.;
      }
      writeIndex++;
    }
  }
}

template<class SerializePolicy, class IntermediatesPolicy, class OverlapPolicy>
template<typename U>
void cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>::update_inverse_path(U inverseCutoffGlobalDimension, U globalDimension,
                                                                                     bool& isInversePath, std::vector<U>& baseCaseDimList, U localDimension){
  if (inverseCutoffGlobalDimension >= globalDimension){
    if (isInversePath == false){
      baseCaseDimList.push_back(localDimension);
    }
    isInversePath = true;
  }
}

template<class SerializePolicy, class IntermediatesPolicy, class OverlapPolicy>
template<typename U>
void cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>::update_inverse_path_simulate(U inverseCutoffGlobalDimension, U globalDimension,
                                                                                     bool& isInversePath, U localDimension){
  if (inverseCutoffGlobalDimension >= globalDimension){
    if (isInversePath == false){
    }
    isInversePath = true;
  }
}


// For solving LA=B for A. But note that B is being modified in place and will turn into A
template<class SerializePolicy, class IntermediatesPolicy, class OverlapPolicy>
template<typename MatrixLType, typename MatrixLIType, typename MatrixAType, typename SquareTableType, typename CommType>
void cholinv<SerializePolicy,IntermediatesPolicy,OverlapPolicy>::solve(
                                  MatrixLType& L, MatrixLIType& LI, MatrixAType& A,
                                  SquareTableType& square_table1, SquareTableType& square_table2, SquareTableType& square_table3, CommType&& CommInfo,
                                  std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                                  blas::ArgPack_gemm<typename MatrixAType::ScalarType>& gemmPackage){

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using Distribution = typename MatrixAType::DistributionType;
  using Offload = typename MatrixAType::OffloadType;

  blas::ArgPack_trmm<T> trmmPackage(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper,
                                    blas::Transpose::AblasTrans, blas::Diag::AblasNonUnit, 1.);
  // Lets operate on individual columns at a time
  // Potential optimization 1): Don't use MM3D if the columns are too skinny in relation to the block size!
     // Or this could just be taken care of when we tune block sizes?
  // Potential optimization 2) Lots of serializing going on with each MM3D, this needs to be reduced.

  U AendX = A.num_columns_local();
  U AendY = A.num_rows_local();
  U matLendX = L.num_columns_local();

  U offset1 = 0; U offset2 = baseCaseDimList[0]; U offset3 = 0;
  // Note that the beginning cases might not be correct. They are not currently used for anything though.
  U arg1 = offset1; U arg2 = matLendX; U arg3 = offset3; U arg4 = offset1;

  for (U i=0; i<baseCaseDimList.size(); i++){

    // Update the current column by accumulating the updates via MM
    gemmPackage.alpha = -1;
    gemmPackage.beta = 1.;

    // Only update once first panel is solved
    if (i>0){
      // As i increases, the size of these updates gets smaller.
      // Special handling. This might only work since the triangular matrix is square, which should be ok

      arg1 = offset1; arg2 = matLendX; arg3 = offset3; arg4 = offset1;
      serialize<typename SerializePolicy::structure,square>::invoke(L, square_table1[std::make_pair(arg2-arg1,arg4-arg3)], arg1, arg2, arg3, arg4);
      serialize<square,square>::invoke(A, square_table2[std::make_pair(AendX,offset1-offset3)], 0, AendX, offset3, offset1);
      serialize<square,square>::invoke(A, square_table3[std::make_pair(AendX,AendY-offset1)], 0, AendX, offset1, AendY);
      matmult::summa::invoke(square_table1[std::make_pair(arg2-arg1,arg4-arg3)], square_table2[std::make_pair(AendX,offset1-offset3)], square_table3[std::make_pair(AendX,AendY-offset1)],
                             std::forward<CommType>(CommInfo), gemmPackage);
      serialize<square,square>::invoke(A, square_table3[std::make_pair(AendX,AendY-offset1)], 0, AendX, offset1, AendY, true);
    }

    // Solve via MM
    serialize<typename SerializePolicy::structure,square>::invoke(LI, square_table1[std::make_pair(offset2-offset1,offset2-offset1)], offset1, offset2, offset1, offset2);
    serialize<square,square>::invoke(A, square_table2[std::make_pair(AendX,offset2-offset1)], 0, AendX, offset1, offset2);
    matmult::summa::invoke(square_table1[std::make_pair(offset2-offset1,offset2-offset1)], square_table2[std::make_pair(AendX,offset2-offset1)], std::forward<CommType>(CommInfo), trmmPackage);
    serialize<square,square>::invoke(A, square_table2[std::make_pair(AendX,offset2-offset1)], 0, AendX, offset1, offset2);
    if ((i+1) < baseCaseDimList.size()){
      // Update the offsets
      offset3 = offset1;
      offset1 = offset2;
      offset2 += baseCaseDimList[i+1];
    }
  }
}

}
