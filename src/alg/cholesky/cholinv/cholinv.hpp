/* Author: Edward Hutter */

namespace cholesky{
template<class TrailingMatrixUpdateLocalCompPolicy, class OverlapGatherPolicy>
template<typename MatrixAType, typename MatrixTIType, typename ArgType, typename CommType>
std::pair<bool,std::vector<typename MatrixAType::DimensionType>>
cholinv<TrailingMatrixUpdateLocalCompPolicy,OverlapGatherPolicy>::invoke(MatrixAType& A, MatrixTIType& TI, ArgType&& args, CommType&& CommInfo){

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using Structure = typename MatrixAType::StructureType;
  using Distribution = typename MatrixAType::DistributionType;
  using Offload = typename MatrixAType::OffloadType;
  assert(args.dir == 'U');	// Removed support for 'L'. Necessary future support for this case can be handled via a final transpose.
  int sliceDim1 = CommInfo.c*CommInfo.d;
  int sliceDim2 = CommInfo.d*CommInfo.d;
  U localDimension = A.num_rows_local();
  U globalDimension = A.num_rows_global();
  U minDimLocal = 1;
  U bcDimLocal  = std::max(minDimLocal,localDimension/(CommInfo.c*CommInfo.d));	// min prevents recursing into a 0x0 local matrix
  U bcDimension = CommInfo.d*bcDimLocal;

  U save = globalDimension;
  for (size_t i=0; i<args.inv_cut_off_dim; i++){
    save >>= 1;
  }
  save = std::max(localDimension*2,save);
  std::pair<bool,std::vector<U>> baseCaseDimList;

  baseCaseDimList.first = (save >= globalDimension ? true : false);
  if (1){//(CommInfo.num_chunks == 0) || (CommInfo.num_chunks > localDimension/sliceDim1)){
    matrix<T,U,uppertri,Distribution,Offload> matrix_base_case(bcDimension, bcDimension, CommInfo.d, CommInfo.d);
    U aggregSize = matrix_base_case.num_elems()*sliceDim2;
    std::vector<T> blocked_data(aggregSize);
    U aggregNumRows = bcDimension;
    U aggregNumColumns = bcDimension;
    U cyclicSize = aggregNumRows*aggregNumColumns;
    std::vector<T> cyclic_data(cyclicSize);
    factor(A, TI, matrix_base_case, blocked_data, cyclic_data, localDimension, localDimension, bcDimension, globalDimension, globalDimension,
      0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, std::forward<CommType>(CommInfo),
      baseCaseDimList.first, baseCaseDimList.second, args.inv_cut_off_dim);
  }
  else{
    matrix<T,U,square,Distribution,Offload> matrix_base_case(bcDimension, bcDimension, CommInfo.d, CommInfo.d);
    U aggregSize = matrix_base_case.num_elems()*sliceDim2;
    std::vector<T> blocked_data(aggregSize);
    U aggregNumRows = bcDimension;
    U aggregNumColumns = bcDimension;
    U cyclicSize = aggregNumRows*aggregNumColumns;
    std::vector<T> cyclic_data(cyclicSize);
    factor(A, TI, matrix_base_case, blocked_data, cyclic_data, localDimension, localDimension, bcDimension, globalDimension, globalDimension,
      0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, std::forward<CommType>(CommInfo),
      baseCaseDimList.first, baseCaseDimList.second, args.inv_cut_off_dim);
  }
  return baseCaseDimList;
}

//TODO: Notice how this routine does not pass back a list of integers like the other invoke method. Should this be supported?
template<class TrailingMatrixUpdateLocalCompPolicy, class OverlapGatherPolicy>
template<typename T, typename U, typename ArgType, typename CommType>
std::pair<T*,T*> cholinv<TrailingMatrixUpdateLocalCompPolicy,OverlapGatherPolicy>::invoke(T* A, T* TI, U localDim, U globalDim, ArgType&& args, CommType&& CommInfo){
  //TODO: Test with non-power-of-2 global matrix dimensions
  matrix<T,U,rect,cyclic> mA(A,localDim,localDim,globalDim,globalDim,CommInfo.c,CommInfo.c);
  matrix<T,U,rect,cyclic> mTI(R,localDim,localDim,globalDim,globalDim,CommInfo.c,CommInfo.c);
  invoke(mA,mTI,std::forward<ArgType>(args),std::forward<CommType>(CommInfo));
  return std::make_pair(mA.get_data(),mTI.get_data());
}

template<class TrailingMatrixUpdateLocalCompPolicy, class OverlapGatherPolicy>
template<typename MatrixAType, typename MatrixRIType, typename BaseCaseMatrixType, typename CommType>
void cholinv<TrailingMatrixUpdateLocalCompPolicy,OverlapGatherPolicy>::factor(MatrixAType& A, MatrixRIType& RI, BaseCaseMatrixType& matrix_base_case,
                           std::vector<typename MatrixAType::ScalarType>& blocked_data, std::vector<typename MatrixAType::ScalarType>& cyclic_data,
                           typename MatrixAType::DimensionType localDimension, typename MatrixAType::DimensionType trueLocalDimension,
                           typename MatrixAType::DimensionType bcDimension, typename MatrixAType::DimensionType globalDimension, typename MatrixAType::DimensionType trueGlobalDimension,
                           typename MatrixAType::DimensionType AstartX, typename MatrixAType::DimensionType AendX, typename MatrixAType::DimensionType AstartY,
                           typename MatrixAType::DimensionType AendY, typename MatrixAType::DimensionType RIstartX, typename MatrixAType::DimensionType RIendX,
                           typename MatrixAType::DimensionType RIstartY, typename MatrixAType::DimensionType RIendY,
                           CommType&& CommInfo, bool& isInversePath, std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                           typename MatrixAType::DimensionType inverseCutoffGlobalDimension){

  if (globalDimension <= bcDimension){
    basecase(A, RI, matrix_base_case, blocked_data, cyclic_data, localDimension, trueLocalDimension, bcDimension, globalDimension, trueGlobalDimension,
      AstartX, AendX, AstartY, AendY, RIstartX, RIendX, RIstartY, RIendY,
      std::forward<CommType>(CommInfo), isInversePath, baseCaseDimList, inverseCutoffGlobalDimension, 'U');
    return;
  }

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using Distribution = typename MatrixAType::DistributionType;
  using Offload = typename MatrixAType::OffloadType;

  int rank;
  // use MPI_COMM_WORLD for this p2p communication for transpose, but could use a smaller communicator
  MPI_Comm_rank(CommInfo.world, &rank);
  // globalDimension will always be a power of 2, but localDimension won't
  U localShift = (localDimension>>1);
  // move localShift up to the next power of 2
  localShift = util::get_next_power2(localShift);
  U globalShift = (globalDimension>>1);
  bool saveSwitch = isInversePath;
  size_t saveIndexPrev = baseCaseDimList.size();

  update_inverse_path(inverseCutoffGlobalDimension, globalDimension, isInversePath, baseCaseDimList, localDimension);
  factor(A, RI, matrix_base_case, blocked_data, cyclic_data, localShift, trueLocalDimension, bcDimension, globalShift, trueGlobalDimension,
    AstartX, AstartX+localShift, AstartY, AstartY+localShift,
    RIstartX, RIstartX+localShift, RIstartY, RIstartY+localShift,
    std::forward<CommType>(CommInfo), isInversePath, baseCaseDimList, inverseCutoffGlobalDimension);

  size_t saveIndexAfter = baseCaseDimList.size();

  // Regardless of whether or not we need to communicate for the transpose, we still need to serialize into a square buffer
  matrix<T,U,uppertri,Distribution,Offload> packedMatrix(nullptr,localShift,localShift,CommInfo.d,CommInfo.d);
  // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
  serialize<square,uppertri>::invoke(RI, packedMatrix, RIstartX, RIstartX+localShift, RIstartY, RIstartY+localShift);
  util::transpose(packedMatrix, std::forward<CommType>(CommInfo));
  blas::ArgPack_trmm<T> trmmArgs(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper,
    blas::Transpose::AblasTrans, blas::Diag::AblasNonUnit, 1.);

  // 2nd case: Extra optimization for the case when we only perform TRSM at the top level.
  if ((isInversePath) || (globalDimension == inverseCutoffGlobalDimension*2)){
    matmult::summa::invoke(packedMatrix, A, 0, localShift, 0, localShift, AstartX+localShift, AendX, AstartY, AstartY+localShift,
      std::forward<CommType>(CommInfo), trmmArgs, false, true);
  }
  else{
    assert(0);
    blas::ArgPack_gemm<T> trsmArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasTrans, blas::Transpose::AblasNoTrans, 1., 0.);

    // create a new subvector
    U len = saveIndexAfter - saveIndexPrev;
    std::vector<U> subBaseCaseDimList(len);
    for (U i=saveIndexPrev; i<saveIndexAfter; i++){
      subBaseCaseDimList[i-saveIndexPrev] = baseCaseDimList[i];
    }
    // make extra copy to avoid corrupting A
    // Future optimization: Copy a part of A into Acopy, to avoid excessing copying
    // Note: some of these globalShifts are wrong, but I don't know any easy way to fix them. Everything might still work though.
    matrix<T,U,square,Distribution,Offload> Rcopy(nullptr, AendX-(AstartX+localShift), localShift, CommInfo.d, CommInfo.d);
    serialize<square,square>::invoke(A, Rcopy, AstartX+localShift, AendX, AstartY, AstartY+localShift);
    // Also need to serialize top-left quadrant of L so that its size matches packedMatrix
    matrix<T,U,uppertri,Distribution,Offload> packedMatrixR(nullptr, localShift, localShift, CommInfo.d, CommInfo.d);
    // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
    serialize<square,uppertri>::invoke(A, packedMatrixR, AstartX, AstartX+localShift, AstartY, AstartY+localShift);
    // Swap, same as we did with inverse
    util::transpose(packedMatrixR, std::forward<CommType>(CommInfo));

    blas::ArgPack_trmm<T> trmmPackage(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper,
      blas::Transpose::AblasTrans, blas::Diag::AblasNonUnit, 1.);
    solve_lower_right(packedMatrixR, packedMatrix, Rcopy, std::forward<CommType>(CommInfo), subBaseCaseDimList, trsmArgs);

    // Inject back into R
    serialize<square,square>::invoke(A, Rcopy, AstartX+localShift, AendX, AstartY, AstartY+localShift, true);
  }

  U reverseDimLocal = localDimension-localShift;
  U reverseDimGlobal = reverseDimLocal*CommInfo.d;

  blas::ArgPack_syrk<T> syrkArgs(blas::Order::AblasColumnMajor, blas::UpLo::AblasUpper, blas::Transpose::AblasTrans, -1., 1.);
  matmult::summa::invoke(A, A, AstartX+localShift, AendX, AstartY, AstartY+localShift,
    AstartX+localShift, AendX, AstartY+localShift, AendY, std::forward<CommType>(CommInfo), syrkArgs, true, true);

  factor(A, RI, matrix_base_case, blocked_data, cyclic_data, reverseDimLocal, trueLocalDimension, bcDimension, reverseDimGlobal/*globalShift*/, trueGlobalDimension,
    AstartX+localShift, AendX, AstartY+localShift, AendY,
    RIstartX+localShift, RIendX, RIstartY+localShift, RIendY,
    std::forward<CommType>(CommInfo), isInversePath, baseCaseDimList, inverseCutoffGlobalDimension);

  // Next step : temp <- R_{12}*RI_{22}
  // We can re-use holdRsyrk as our temporary output Matrix.

  if (isInversePath){
    // Tradeoff: By encapsulating the transpose/serialization of L21 in the syrk summa routine above, I can't reuse that buffer and must re-serialize L21
    matrix<T,U,square,Distribution,Offload> tempInverse(nullptr, reverseDimLocal, localShift, CommInfo.d, CommInfo.d);
    serialize<square,square>::invoke(A, tempInverse, AstartX+localShift, AendX, AstartY, AstartY+localShift);

    blas::ArgPack_trmm<T> invPackage1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper,
      blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    matmult::summa::invoke(RI, tempInverse, RIstartX+localShift, RIendX, RIstartY+localShift, RIendY, 0, reverseDimLocal, 0, localShift,
      std::forward<CommType>(CommInfo), invPackage1, true, false);

    // Next step: finish the Triangular inverse calculation
    invPackage1.alpha = -1.;
    invPackage1.side = blas::Side::AblasLeft;
    matmult::summa::invoke(RI, tempInverse, RIstartX, RIstartX+localShift, RIstartY, RIstartY+localShift, 0, reverseDimLocal, 0, localShift,
      std::forward<CommType>(CommInfo), invPackage1, true, false);
    serialize<square,square>::invoke(RI, tempInverse, RIstartX+localShift, RIendX, RIstartY, RIstartY+localShift, true);
  }
  isInversePath = saveSwitch;
}


template<class TrailingMatrixUpdateLocalCompPolicy, class OverlapGatherPolicy>
template<typename MatrixAType, typename MatrixIType, typename BaseCaseMatrixType, typename CommType>
void cholinv<TrailingMatrixUpdateLocalCompPolicy,OverlapGatherPolicy>::basecase(MatrixAType& A, MatrixIType& I, BaseCaseMatrixType& matrix_base_case,
                     std::vector<typename MatrixAType::ScalarType>& blocked_data, std::vector<typename MatrixAType::ScalarType>& cyclic_data,
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
    // Only save if we never got onto the inverse path
    baseCaseDimList.push_back(localDimension);
  }
  if (localDimension == 0) return;

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

  // Should be fast pass-by-value via move semantics
  aggregate(A, matrix_base_case, blocked_data, cyclic_data, localDimension, globalDimension, globalDimension/*bcDimension*/,
    (dir == 'L' ? AstartX : AstartY), (dir == 'L' ? AendX : AendY),
    (dir == 'L' ? AstartX : AstartY), (dir == 'L' ? AendX : AendY), std::forward<CommType>(CommInfo), dir);

  U fTranDim1 = localDimension*CommInfo.d;
  std::vector<T>& storeMat = cyclic_data;
  // Until then, assume a double datatype and simply use LAPACKE_dpotrf. Worry about adding more capabilities later.
  lapack::ArgPack_potrf potrfArgs(lapack::Order::AlapackColumnMajor, (dir == 'L' ? lapack::UpLo::AlapackLower : lapack::UpLo::AlapackUpper));
  lapack::ArgPack_trtri trtriArgs(lapack::Order::AlapackColumnMajor, (dir == 'L' ? lapack::UpLo::AlapackLower : lapack::UpLo::AlapackUpper), lapack::Diag::AlapackNonUnit);
  lapack::engine::_potrf(&storeMat[0],fTranDim1,fTranDim1,potrfArgs);
  std::vector<T> storeMatInv = storeMat;		// true copy because we have to, unless we want to iterate (see below) two different times
  lapack::engine::_trtri(&storeMatInv[0],fTranDim1,fTranDim1,trtriArgs);

  // Only truly a "square-to-square" serialization because we store L as a square (no packed storage yet!)

  // Now, before we can serialize into L and LI, we need to save the values that this processor owns according to the cyclic rule.
  // Only then can we serialize.

  // Iterate and pick out. I would like not to have to create any more memory and I would only like to iterate once, not twice, for storeL and storeLI
  //   Use the "overwrite" trick that I have used in CASI code, as well as other places

  // I am going to use a sneaky trick: I will take the vectorData from storeL and storeLI by reference, overwrite its values,
  //   and then "move" them cheaply into new Matrix structures before I call invoke on them individually.

  cyclicToLocalTransformation(storeMat, storeMatInv, localDimension, globalDimension, globalDimension/*bcDimension*/, CommInfo.d,rankSlice,dir);

  matrix<T,U,square,Distribution,Offload> tempMat(&storeMat[0],localDimension,localDimension,globalDimension,globalDimension,CommInfo.d,CommInfo.d);
  matrix<T,U,square,Distribution,Offload> tempMatInv(&storeMatInv[0],localDimension,localDimension,globalDimension,globalDimension,CommInfo.d,CommInfo.d);

  // invoke into the existing Matrix data structures owned by the user
  serialize<square,square>::invoke(A, tempMat, (dir == 'L' ? AstartX : AstartY), (dir == 'L' ? AendX : AendY),
    (dir == 'L' ? AstartX : AstartY), (dir == 'L' ? AendX : AendY), true);
  serialize<square,square>::invoke(I, tempMatInv, matIstartX, matIendX, matIstartY, matIendY, true);
  return;
}


template<class TrailingMatrixUpdateLocalCompPolicy, class OverlapGatherPolicy>
template<typename MatrixType, typename BaseCaseMatrixType, typename CommType>
void cholinv<TrailingMatrixUpdateLocalCompPolicy,OverlapGatherPolicy>::aggregate(MatrixType& A, BaseCaseMatrixType& matrix_base_case, std::vector<typename MatrixType::ScalarType>& blocked_data,
                   std::vector<typename MatrixType::ScalarType>& cyclic_data, typename MatrixType::DimensionType localDimension, typename MatrixType::DimensionType globalDimension,
                               typename MatrixType::DimensionType bcDimension, typename MatrixType::DimensionType AstartX, typename MatrixType::DimensionType AendX,
                               typename MatrixType::DimensionType AstartY, typename MatrixType::DimensionType AendY, CommType&& CommInfo, char dir){

  //assert(matrix_base_case.num_columns_local() == localDimension);
  using T = typename MatrixType::ScalarType;
  using U = typename MatrixType::DimensionType;
  using Distribution = typename MatrixType::DistributionType;
  using Offload = typename MatrixType::OffloadType;
  using BaseCaseStructure = typename BaseCaseMatrixType::StructureType;

  serialize<square,BaseCaseStructure>::invoke(A, matrix_base_case, AstartX, AendX, AstartY, AendY);
  policy::cholinv::OverlapGatherPolicyClass<OverlapGatherPolicy>::invoke(matrix_base_case,blocked_data,cyclic_data,std::forward<CommType>(CommInfo));
}


// This method can be called from Lower and Upper with one tweak, but note that currently, we iterate over the entire square,
//   when we are really only writing to a triangle. So there is a source of optimization here at least in terms of
//   number of flops, but in terms of memory accesses and cache lines, not sure. Note that with this optimization,
//   we may need to separate into two different functions
template<class TrailingMatrixUpdateLocalCompPolicy, class OverlapGatherPolicy>
template<typename T, typename U>
void cholinv<TrailingMatrixUpdateLocalCompPolicy,OverlapGatherPolicy>::cyclicToLocalTransformation(std::vector<T>& storeT, std::vector<T>& storeTI, U localDimension,
                                                                                             U globalDimension, U bcDimension, int64_t sliceDim, int64_t rankSlice, char dir){

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
      if (((dir == 'L') && (readIndexCol <= readIndexRow)) ||  ((dir == 'U') && (readIndexCol >= readIndexRow))){
        storeT[writeIndex] = storeT[readIndexCol*bcDimension + readIndexRow];
        storeTI[writeIndex] = storeTI[readIndexCol*bcDimension + readIndexRow];
      }
      else{
//        break;
        storeT[writeIndex] = 0.;
        storeTI[writeIndex] = 0.;
      }
      writeIndex++;
    }
  }
}

template<class TrailingMatrixUpdateLocalCompPolicy, class OverlapGatherPolicy>
template<typename U>
void cholinv<TrailingMatrixUpdateLocalCompPolicy,OverlapGatherPolicy>::update_inverse_path(U inverseCutoffGlobalDimension, U globalDimension,
                                                                                     bool& isInversePath, std::vector<U>& baseCaseDimList, U localDimension){
  if (inverseCutoffGlobalDimension >= globalDimension){
    if (isInversePath == false){
      baseCaseDimList.push_back(localDimension);
    }
    isInversePath = true;
  }
}


// For solving LA=B for A. But note that B is being modified in place and will turn into A
template<class TrailingMatrixUpdateLocalCompPolicy, class OverlapGatherPolicy>
template<typename MatrixLType, typename MatrixLIType, typename MatrixAType, typename CommType>
void cholinv<TrailingMatrixUpdateLocalCompPolicy,OverlapGatherPolicy>::solve_lower_right(MatrixLType& L, MatrixLIType& LI, MatrixAType& A, CommType&& CommInfo,
                                  std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                                  blas::ArgPack_gemm<typename MatrixAType::ScalarType>& gemmPackage){

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using StructureTri = typename MatrixLType::StructureType;
  using Distribution = typename MatrixAType::DistributionType;
  using Offload = typename MatrixAType::OffloadType;

  blas::ArgPack_trmm<T> trmmPackage(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper,
      blas::Transpose::AblasTrans, blas::Diag::AblasNonUnit, 1.);
  // to catch debugging issues, assert that this has at least one size
  assert(baseCaseDimList.size());

  // Lets operate on individual columns at a time
  // Potential optimization 1): Don't use MM3D if the columns are too skinny in relation to the block size!
     // Or this could just be taken care of when we tune block sizes?
  // Potential optimization 2) Lots of serializing going on with each MM3D, this needs to be reduced.

  U AendX = A.num_columns_local();
  U AendY = A.num_rows_local();
  U matLendX = L.num_columns_local();

  U offset1 = 0; U offset2 = baseCaseDimList[0]; U offset3 = 0;
  // Note that the beginning cases might not be correct. They are not currently used for anything though.
  U arg1 = (gemmPackage.transposeA == blas::Transpose::AblasNoTrans ? offset3 : offset1);
  U arg2 = (gemmPackage.transposeA == blas::Transpose::AblasNoTrans ? offset1 : matLendX);
  U arg3 = (gemmPackage.transposeA == blas::Transpose::AblasNoTrans ? offset3 : offset3);
  U arg4 = (gemmPackage.transposeA == blas::Transpose::AblasNoTrans ? offset1 : offset1);
  matrix<T,U,rect,Distribution,Offload> Lpartition(nullptr, arg2-arg1, arg4-arg3, CommInfo.d, CommInfo.d);

  for (U i=0; i<baseCaseDimList.size(); i++){

    // Update the current column by accumulating the updates via MM
    gemmPackage.alpha = -1;
    gemmPackage.beta = 1.;

    // Only update once first panel is solved
    if (i>0){
      // As i increases, the size of these updates gets smaller.
      // Special handling. This might only work since the triangular matrix is square, which should be ok

      // Note that the beginning cases might not be correct. They are not currently used for anything though.
      arg1 = (gemmPackage.transposeA == blas::Transpose::AblasNoTrans ? offset3 : offset1);
      arg2 = (gemmPackage.transposeA == blas::Transpose::AblasNoTrans ? offset1 : matLendX);
      arg3 = (gemmPackage.transposeA == blas::Transpose::AblasNoTrans ? offset3 : offset3);
      arg4 = (gemmPackage.transposeA == blas::Transpose::AblasNoTrans ? offset1 : offset1);

      serialize<StructureTri,rect>::invoke(L, Lpartition, arg1, arg2, arg3, arg4);
//      matmult::summa::invoke(Lpartition, A, A, 0, arg2-arg1, 0, arg4-arg3, 0, AendX, offset3, offset1,
//        0, AendX, offset1, AendY, std::forward<CommType>(CommInfo), gemmPackage, false, true, true);
    }

    // Solve via MM
//    matmult::summa::invoke(LI, A, offset1, offset2, offset1, offset2, 0, AendX, offset1, offset2, std::forward<CommType>(CommInfo), trmmPackage, true, true);

    if ((i+1) < baseCaseDimList.size()){
      // Update the offsets
      offset3 = offset1;
      offset1 = offset2;
      offset2 += baseCaseDimList[i+1];
    }
  }
}

}
