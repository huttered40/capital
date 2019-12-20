/* Author: Edward Hutter */

namespace cholesky{
template<typename MatrixAType, typename MatrixTIType, typename ArgType, typename CommType>
std::pair<bool,std::vector<typename MatrixAType::DimensionType>>
cholinv::invoke(MatrixAType& matrixA, MatrixTIType& matrixTI, ArgType&& args, CommType&& CommInfo){

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using Distribution = typename MatrixAType::DistributionType;
  using Offload = typename MatrixAType::OffloadType;
  int sliceDim1 = CommInfo.c*CommInfo.d;
  int sliceDim2 = CommInfo.d*CommInfo.d;
  U localDimension = matrixA.getNumRowsLocal();
  U globalDimension = matrixA.getNumRowsGlobal();
  // the division below may have a remainder, but I think integer division will be ok, as long as we change the base case condition to be <= and not just ==
  U bcDimension = globalDimension/(CommInfo.c*CommInfo.d);

  U save = args.inv_cut_off_dim;
  args.inv_cut_off_dim = globalDimension;
  for (size_t i=0; i<save; i++){
    args.inv_cut_off_dim >>= 1;
  }
  args.inv_cut_off_dim = std::max(localDimension*2,args.inv_cut_off_dim);
  std::pair<bool,std::vector<U>> baseCaseDimList;

  if (args.dir == 'L'){
    baseCaseDimList.first = (args.inv_cut_off_dim >= globalDimension ? true : false);
    if ((CommInfo.num_chunks == 0) || (CommInfo.num_chunks > localDimension/sliceDim1)){
      matrix<T,U,lowertri,Distribution,Offload> matrix_base_case(std::vector<T>(), localDimension/sliceDim1, localDimension/sliceDim1, globalDimension/sliceDim1, globalDimension/sliceDim1);
      U aggregSize = matrix_base_case.getNumElems()*sliceDim2;
      std::vector<T> blocked_data(aggregSize);
      U aggregNumRows = localDimension/CommInfo.d;
      U aggregNumColumns = localDimension/CommInfo.d;
      U cyclicSize = aggregNumRows*aggregNumColumns;
      std::vector<T> cyclic_data(cyclicSize);
      factor_lower(matrixA, matrixTI, matrix_base_case, blocked_data, cyclic_data, localDimension, localDimension, bcDimension, globalDimension, globalDimension,
        0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, std::forward<CommType>(CommInfo),
        baseCaseDimList.first, baseCaseDimList.second, args.inv_cut_off_dim);
    }
    else{
      matrix<T,U,square,Distribution,Offload> matrix_base_case(std::vector<T>(), localDimension/sliceDim1, localDimension/sliceDim1, globalDimension/sliceDim1, globalDimension/sliceDim1);
      U aggregSize = matrix_base_case.getNumElems()*sliceDim2;
      std::vector<T> blocked_data(aggregSize);
      U aggregNumRows = localDimension/CommInfo.d;
      U aggregNumColumns = localDimension/CommInfo.d;
      U cyclicSize = aggregNumRows*aggregNumColumns;
      std::vector<T> cyclic_data(cyclicSize);
      factor_lower(matrixA, matrixTI, matrix_base_case, blocked_data, cyclic_data, localDimension, localDimension, bcDimension, globalDimension, globalDimension,
        0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, std::forward<CommType>(CommInfo),
        baseCaseDimList.first, baseCaseDimList.second, args.inv_cut_off_dim);
    }
  }
  else if (args.dir == 'U'){
    baseCaseDimList.first = (args.inv_cut_off_dim >= globalDimension ? true : false);
    if ((CommInfo.num_chunks == 0) || (CommInfo.num_chunks > localDimension/sliceDim1)){
      matrix<T,U,uppertri,Distribution,Offload> matrix_base_case(std::vector<T>(), localDimension/sliceDim1, localDimension/sliceDim1, globalDimension/sliceDim1, globalDimension/sliceDim1);
      U aggregSize = matrix_base_case.getNumElems()*sliceDim2;
      std::vector<T> blocked_data(aggregSize);
      U aggregNumRows = localDimension/CommInfo.d;
      U aggregNumColumns = localDimension/CommInfo.d;
      U cyclicSize = aggregNumRows*aggregNumColumns;
      std::vector<T> cyclic_data(cyclicSize);
      factor_upper(matrixA, matrixTI, matrix_base_case, blocked_data, cyclic_data, localDimension, localDimension, bcDimension, globalDimension, globalDimension,
        0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, std::forward<CommType>(CommInfo),
        baseCaseDimList.first, baseCaseDimList.second, args.inv_cut_off_dim);
    }
    else{
      matrix<T,U,square,Distribution,Offload> matrix_base_case(std::vector<T>(), localDimension/sliceDim1, localDimension/sliceDim1, globalDimension/sliceDim1, globalDimension/sliceDim1);
      U aggregSize = matrix_base_case.getNumElems()*sliceDim2;
      std::vector<T> blocked_data(aggregSize);
      U aggregNumRows = localDimension/CommInfo.d;
      U aggregNumColumns = localDimension/CommInfo.d;
      U cyclicSize = aggregNumRows*aggregNumColumns;
      std::vector<T> cyclic_data(cyclicSize);
      factor_upper(matrixA, matrixTI, matrix_base_case, blocked_data, cyclic_data, localDimension, localDimension, bcDimension, globalDimension, globalDimension,
        0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, std::forward<CommType>(CommInfo),
        baseCaseDimList.first, baseCaseDimList.second, args.inv_cut_off_dim);
    }
  }
  return baseCaseDimList;
}

template<typename MatrixAType, typename MatrixLIType, typename BaseCaseMatrixType, typename CommType>
void cholinv::factor_lower(MatrixAType& matrixA, MatrixLIType& matrixLI, BaseCaseMatrixType& matrix_base_case,
                           std::vector<typename MatrixAType::ScalarType>& blocked_data, std::vector<typename MatrixAType::ScalarType>& cyclic_data,
                           typename MatrixAType::DimensionType localDimension, typename MatrixAType::DimensionType trueLocalDimension,
                           typename MatrixAType::DimensionType bcDimension, typename MatrixAType::DimensionType globalDimension, typename MatrixAType::DimensionType trueGlobalDimension,
                           typename MatrixAType::DimensionType matAstartX, typename MatrixAType::DimensionType matAendX, typename MatrixAType::DimensionType matAstartY,
                           typename MatrixAType::DimensionType matAendY, typename MatrixAType::DimensionType matLIstartX, typename MatrixAType::DimensionType matLIendX,
                           typename MatrixAType::DimensionType matLIstartY, typename MatrixAType::DimensionType matLIendY,
                           CommType&& CommInfo, bool& isInversePath, std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                           typename MatrixAType::DimensionType inverseCutoffGlobalDimension){

  if (globalDimension <= bcDimension){
    baseCase(matrixA, matrixLI, matrix_base_case, blocked_data, cyclic_data, localDimension, trueLocalDimension, bcDimension, globalDimension, trueGlobalDimension,
      matAstartX, matAendX, matAstartY, matAendY, matLIstartX, matLIendX, matLIstartY, matLIendY,
      std::forward<CommType>(CommInfo), isInversePath, baseCaseDimList, inverseCutoffGlobalDimension, 'L');
    return;
  }

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using Distribution = typename MatrixAType::DistributionType;
  using Offload = typename MatrixAType::OffloadType;

  int rank;
  MPI_Comm_rank(CommInfo.world, &rank);
  // globalDimension will always be a power of 2, but localDimension won't
  U localShift = (localDimension>>1);
  // move localShift up to the next power of 2, only useful if matrix dimensions are not powers of 2
  localShift = util::getNextPowerOf2(localShift);
  // Note: I think this globalShift calculation is wrong, but it hasn't been an issue because its not really used for anything.
  U globalShift = (globalDimension>>1);
  bool saveSwitch = isInversePath;
  size_t saveIndexPrev = baseCaseDimList.size();

  updateInversePath(inverseCutoffGlobalDimension, globalDimension, isInversePath, baseCaseDimList, localDimension);
  factor_lower(matrixA, matrixLI, matrix_base_case, blocked_data, cyclic_data, localShift, trueLocalDimension, bcDimension, globalShift, trueGlobalDimension,
    matAstartX, matAstartX+localShift, matAstartY, matAstartY+localShift,
    matLIstartX, matLIstartX+localShift, matLIstartY, matLIstartY+localShift,
    std::forward<CommType>(CommInfo), isInversePath, baseCaseDimList, inverseCutoffGlobalDimension);

  size_t saveIndexAfter = baseCaseDimList.size();

  // Regardless of whether or not we need to communicate for the transpose, we still need to serialize into a buffer
  matrix<T,U,lowertri,Distribution,Offload> packedMatrix(std::vector<T>(), localShift, localShift, globalShift, globalShift);
  // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
  serialize<square,lowertri>::invoke(matrixLI, packedMatrix, matLIstartX, matLIstartX+localShift, matLIstartY, matLIstartY+localShift);
  util::transposeSwap(packedMatrix, std::forward<CommType>(CommInfo));

  blas::ArgPack_trmm<T> trmmArgs(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasLower,
    blas::Transpose::AblasTrans, blas::Diag::AblasNonUnit, 1.);

  // 2nd case: Extra optimization for the case when we only perform TRSM at the top level.
  if ((isInversePath) || (globalDimension == inverseCutoffGlobalDimension*2)){
    //std::cout << "tell me localDim and localshIFT - " << localDimension << " " << localShift << std::endl;
    matmult::summa::invoke(packedMatrix, matrixA, 0, localShift, 0, localShift, matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY, std::forward<CommType>(CommInfo), trmmArgs, false, true);
  }
  else{
    // Note: keep this a gemm package, because we still need to use gemm in TRSM3D in the update, which is just rectangular and non-triangular matrices.
    blas::ArgPack_gemm<T> trsmArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasNoTrans, blas::Transpose::AblasTrans, 1., 0.);

    // create a new subvector
    U len = saveIndexAfter - saveIndexPrev;
    std::vector<U> subBaseCaseDimList(len);
    for (U i=saveIndexPrev; i<saveIndexAfter; i++){
      subBaseCaseDimList[i-saveIndexPrev] = baseCaseDimList[i];
    }

    // TODO: Note: some of those steps are unnecessary if we are doing TRSM3D to one level deep.
    // make extra copy to avoid corrupting matrixA
    // Note: some of these globalShifts are wrong, but I don't know any easy way to fix them. Everything might still work though.
    matrix<T,U,square,Distribution,Offload> matrixLcopy(std::vector<T>(), localShift, matAendY-(matAstartY+localShift), globalShift, globalShift);
    serialize<square,square>::invoke(matrixA, matrixLcopy, matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY);
    // Also need to serialize top-left quadrant of matrixL so that its size matches packedMatrix
    matrix<T,U,lowertri,Distribution,Offload> packedMatrixL(std::vector<T>(), localShift, localShift, globalShift, globalShift);
    // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
    serialize<square,lowertri>::invoke(matrixA, packedMatrixL, matAstartX, matAstartX+localShift, matAstartY, matAstartY+localShift);
    // Swap, same as we did with inverse
    util::transposeSwap(packedMatrixL, std::forward<CommType>(CommInfo));

    blas::ArgPack_trmm<T> trmmPackage(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasLower,
      blas::Transpose::AblasTrans, blas::Diag::AblasNonUnit, 1.);
    iSolveUpperLeft(matrixLcopy, packedMatrixL, packedMatrix, std::forward<CommType>(CommInfo), subBaseCaseDimList, trsmArgs);

    // inject matrixLcopy back into matrixA.
    // Future optimization: avoid copying matrixL here, and utilize leading dimension and the column vectors.
    serialize<square,square>::invoke(matrixA, matrixLcopy, matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY, true);
//      if (rank == 0) matrixLcopy.print();
  }

  // Now we need to perform L_{21}L_{21}^T via syrk
  //   Actually, I am havin trouble with SYRK, lets try gemm instead
  // As of January 2017, still having trouble with SYRK.

  // Later optimization: avoid this recalculation at each recursive level, since it will always be the same.
  U reverseDimLocal = localDimension-localShift;
  U reverseDimGlobal = reverseDimLocal*CommInfo.d;

  blas::ArgPack_syrk<T> syrkArgs(blas::Order::AblasColumnMajor, blas::UpLo::AblasLower, blas::Transpose::AblasNoTrans, -1., 1.);
  matmult::summa::invoke(matrixA, matrixA, matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY,
    matAstartX+localShift, matAendX, matAstartY+localShift, matAendY, std::forward<CommType>(CommInfo), syrkArgs, true, true);

  factor_lower(matrixA, matrixLI, matrix_base_case, blocked_data, cyclic_data, reverseDimLocal, trueLocalDimension, bcDimension, reverseDimGlobal/*globalShift*/, trueGlobalDimension,
    matAstartX+localShift, matAendX, matAstartY+localShift, matAendY,
    matLIstartX+localShift, matLIendX, matLIstartY+localShift, matLIendY,
    std::forward<CommType>(CommInfo), isInversePath, baseCaseDimList, inverseCutoffGlobalDimension);

  if (isInversePath){
    // Next step : temp <- L_{21}*LI_{11}
    // Tradeoff: By encapsulating the transpose/serialization of L21 in the syrk summa routine above, I can't reuse that buffer and must re-serialize L21
    matrix<T,U,square,Distribution,Offload> tempInverse(std::vector<T>(), localShift, reverseDimLocal, globalShift, reverseDimGlobal);
    // NOTE: WE BROKE SQUARE SEMANTICS WITH THIS. CHANGE LATER!
    serialize<square,square>::invoke(matrixA, tempInverse, matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY);

    blas::ArgPack_trmm<T> invPackage1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasLower,
      blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    matmult::summa::invoke(matrixLI, tempInverse, matLIstartX, matLIstartX+localShift, matLIstartY,
        matLIstartY+localShift, 0, localShift, 0, reverseDimLocal, std::forward<CommType>(CommInfo), invPackage1, true, false);

    // Next step: finish the Triangular inverse calculation
    invPackage1.alpha = -1.;
    invPackage1.side = blas::Side::AblasLeft;
    matmult::summa::invoke(matrixLI, tempInverse, matLIstartX+localShift, matLIendX, matLIstartY+localShift, matLIendY, 0, localShift, 0, reverseDimLocal,
                             std::forward<CommType>(CommInfo), invPackage1, true, false);
    // One final serialize of tempInverse into matrixLI
    serialize<square,square>::invoke(matrixLI, tempInverse, matLIstartX, matLIstartX+localShift, matLIstartY+localShift, matLIendY, true);
  }
  isInversePath = saveSwitch;
}


template<typename MatrixAType, typename MatrixRIType, typename BaseCaseMatrixType, typename CommType>
void cholinv::factor_upper(MatrixAType& matrixA, MatrixRIType& matrixRI, BaseCaseMatrixType& matrix_base_case,
                           std::vector<typename MatrixAType::ScalarType>& blocked_data, std::vector<typename MatrixAType::ScalarType>& cyclic_data,
                           typename MatrixAType::DimensionType localDimension, typename MatrixAType::DimensionType trueLocalDimension,
                           typename MatrixAType::DimensionType bcDimension, typename MatrixAType::DimensionType globalDimension, typename MatrixAType::DimensionType trueGlobalDimension,
                           typename MatrixAType::DimensionType matAstartX, typename MatrixAType::DimensionType matAendX, typename MatrixAType::DimensionType matAstartY,
                           typename MatrixAType::DimensionType matAendY, typename MatrixAType::DimensionType matRIstartX, typename MatrixAType::DimensionType matRIendX,
                           typename MatrixAType::DimensionType matRIstartY, typename MatrixAType::DimensionType matRIendY,
                           CommType&& CommInfo, bool& isInversePath, std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                           typename MatrixAType::DimensionType inverseCutoffGlobalDimension){

  if (globalDimension <= bcDimension){
    baseCase(matrixA, matrixRI, matrix_base_case, blocked_data, cyclic_data, localDimension, trueLocalDimension, bcDimension, globalDimension, trueGlobalDimension,
      matAstartX, matAendX, matAstartY, matAendY, matRIstartX, matRIendX, matRIstartY, matRIendY,
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
  localShift = util::getNextPowerOf2(localShift);
  U globalShift = (globalDimension>>1);
  bool saveSwitch = isInversePath;
  size_t saveIndexPrev = baseCaseDimList.size();

  updateInversePath(inverseCutoffGlobalDimension, globalDimension, isInversePath, baseCaseDimList, localDimension);
  factor_upper(matrixA, matrixRI, matrix_base_case, blocked_data, cyclic_data, localShift, trueLocalDimension, bcDimension, globalShift, trueGlobalDimension,
    matAstartX, matAstartX+localShift, matAstartY, matAstartY+localShift,
    matRIstartX, matRIstartX+localShift, matRIstartY, matRIstartY+localShift,
    std::forward<CommType>(CommInfo), isInversePath, baseCaseDimList, inverseCutoffGlobalDimension);

  size_t saveIndexAfter = baseCaseDimList.size();

  // Regardless of whether or not we need to communicate for the transpose, we still need to serialize into a square buffer
  matrix<T,U,uppertri,Distribution,Offload> packedMatrix(std::vector<T>(), localShift, localShift, globalShift, globalShift);
  // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
  serialize<square,uppertri>::invoke(matrixRI, packedMatrix, matRIstartX, matRIstartX+localShift, matRIstartY, matRIstartY+localShift);
  util::transposeSwap(packedMatrix, std::forward<CommType>(CommInfo));
  blas::ArgPack_trmm<T> trmmArgs(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper,
    blas::Transpose::AblasTrans, blas::Diag::AblasNonUnit, 1.);

  // 2nd case: Extra optimization for the case when we only perform TRSM at the top level.
  if ((isInversePath) || (globalDimension == inverseCutoffGlobalDimension*2)){
    matmult::summa::invoke(packedMatrix, matrixA, 0, localShift, 0, localShift, matAstartX+localShift, matAendX, matAstartY, matAstartY+localShift,
      std::forward<CommType>(CommInfo), trmmArgs, false, true);
  }
  else{
    blas::ArgPack_gemm<T> trsmArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasTrans, blas::Transpose::AblasNoTrans, 1., 0.);

    // create a new subvector
    U len = saveIndexAfter - saveIndexPrev;
    std::vector<U> subBaseCaseDimList(len);
    for (U i=saveIndexPrev; i<saveIndexAfter; i++){
      subBaseCaseDimList[i-saveIndexPrev] = baseCaseDimList[i];
    }
    // make extra copy to avoid corrupting matrixA
    // Future optimization: Copy a part of A into matrixAcopy, to avoid excessing copying
    // Note: some of these globalShifts are wrong, but I don't know any easy way to fix them. Everything might still work though.
    matrix<T,U,square,Distribution,Offload> matrixRcopy(std::vector<T>(), matAendX-(matAstartX+localShift), localShift, globalShift, globalShift);
    serialize<square,square>::invoke(matrixA, matrixRcopy, matAstartX+localShift, matAendX, matAstartY, matAstartY+localShift);
    // Also need to serialize top-left quadrant of matrixL so that its size matches packedMatrix
    matrix<T,U,uppertri,Distribution,Offload> packedMatrixR(std::vector<T>(), localShift, localShift, globalShift, globalShift);
    // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
    serialize<square,uppertri>::invoke(matrixA, packedMatrixR, matAstartX, matAstartX+localShift, matAstartY, matAstartY+localShift);
    // Swap, same as we did with inverse
    util::transposeSwap(packedMatrixR, std::forward<CommType>(CommInfo));

    blas::ArgPack_trmm<T> trmmPackage(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper,
      blas::Transpose::AblasTrans, blas::Diag::AblasNonUnit, 1.);
    iSolveLowerRight(packedMatrixR, packedMatrix, matrixRcopy, std::forward<CommType>(CommInfo), subBaseCaseDimList, trsmArgs);

    // Inject back into matrixR
    serialize<square,square>::invoke(matrixA, matrixRcopy, matAstartX+localShift, matAendX, matAstartY, matAstartY+localShift, true);
  }

  U reverseDimLocal = localDimension-localShift;
  U reverseDimGlobal = reverseDimLocal*CommInfo.d;

  blas::ArgPack_syrk<T> syrkArgs(blas::Order::AblasColumnMajor, blas::UpLo::AblasUpper, blas::Transpose::AblasTrans, -1., 1.);
  matmult::summa::invoke(matrixA, matrixA, matAstartX+localShift, matAendX, matAstartY, matAstartY+localShift,
    matAstartX+localShift, matAendX, matAstartY+localShift, matAendY, std::forward<CommType>(CommInfo), syrkArgs, true, true);

  factor_upper(matrixA, matrixRI, matrix_base_case, blocked_data, cyclic_data, reverseDimLocal, trueLocalDimension, bcDimension, reverseDimGlobal/*globalShift*/, trueGlobalDimension,
    matAstartX+localShift, matAendX, matAstartY+localShift, matAendY,
    matRIstartX+localShift, matRIendX, matRIstartY+localShift, matRIendY,
    std::forward<CommType>(CommInfo), isInversePath, baseCaseDimList, inverseCutoffGlobalDimension);

  // Next step : temp <- R_{12}*RI_{22}
  // We can re-use holdRsyrk as our temporary output matrix.

  if (isInversePath){
    // Tradeoff: By encapsulating the transpose/serialization of L21 in the syrk summa routine above, I can't reuse that buffer and must re-serialize L21
    matrix<T,U,square,Distribution,Offload> tempInverse(std::vector<T>(), reverseDimLocal, localShift, reverseDimGlobal, globalShift);
    // NOTE: WE BROKE SQUARE SEMANTICS WITH THIS. CHANGE LATER!
    serialize<square,square>::invoke(matrixA, tempInverse, matAstartX+localShift, matAendX, matAstartY, matAstartY+localShift);

    blas::ArgPack_trmm<T> invPackage1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper,
      blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    matmult::summa::invoke(matrixRI, tempInverse, matRIstartX+localShift, matRIendX, matRIstartY+localShift, matRIendY, 0, reverseDimLocal, 0, localShift,
      std::forward<CommType>(CommInfo), invPackage1, true, false);

    // Next step: finish the Triangular inverse calculation
    invPackage1.alpha = -1.;
    invPackage1.side = blas::Side::AblasLeft;
    matmult::summa::invoke(matrixRI, tempInverse, matRIstartX, matRIstartX+localShift, matRIstartY, matRIstartY+localShift, 0, reverseDimLocal, 0, localShift,
      std::forward<CommType>(CommInfo), invPackage1, true, false);
    serialize<square,square>::invoke(matrixRI, tempInverse, matRIstartX+localShift, matRIendX, matRIstartY, matRIstartY+localShift, true);
  }
  isInversePath = saveSwitch;
}


template<typename MatrixAType, typename MatrixIType, typename BaseCaseMatrixType, typename CommType>
void cholinv::baseCase(MatrixAType& matrixA, MatrixIType& matrixI, BaseCaseMatrixType& matrix_base_case,
                     std::vector<typename MatrixAType::ScalarType>& blocked_data, std::vector<typename MatrixAType::ScalarType>& cyclic_data,
                     typename MatrixAType::DimensionType localDimension, typename MatrixAType::DimensionType trueLocalDimension,
                     typename MatrixAType::DimensionType bcDimension, typename MatrixAType::DimensionType globalDimension, typename MatrixAType::DimensionType trueGlobalDimension,
                     typename MatrixAType::DimensionType matAstartX, typename MatrixAType::DimensionType matAendX, typename MatrixAType::DimensionType matAstartY,
                     typename MatrixAType::DimensionType matAendY, typename MatrixAType::DimensionType matIstartX, typename MatrixAType::DimensionType matIendX,
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
  aggregate(matrixA, matrix_base_case, blocked_data, cyclic_data, localDimension, globalDimension, globalDimension/*bcDimension*/,
    (dir == 'L' ? matAstartX : matAstartY), (dir == 'L' ? matAendX : matAendY),
    (dir == 'L' ? matAstartX : matAstartY), (dir == 'L' ? matAendX : matAendY), std::forward<CommType>(CommInfo), dir);

  // TODO: Note: with my new optimizations, this case might never pass, because A is serialized into. Watch out!
  if (((dir == 'L') && (matAendX == trueLocalDimension)) || ((dir == 'U') && (matAendY == trueLocalDimension))){
    //U finalDim = trueLocalDimension*CommInfo.d - trueGlobalDimension;
    U checkDim = localDimension*CommInfo.d;
    U finalDim = (checkDim - (trueLocalDimension*CommInfo.d - trueGlobalDimension));

    std::vector<T> deepBaseCase(finalDim*finalDim,0);
    // manual serialize
    for (U i=0; i<finalDim; i++){
      for (U j=0; j<finalDim; j++){
        deepBaseCase[i*finalDim+j] = cyclic_data[i*checkDim+j];
      }
    }

    lapack::ArgPack_potrf potrfArgs(lapack::Order::AlapackColumnMajor, (dir == 'L' ? lapack::UpLo::AlapackLower : lapack::UpLo::AlapackUpper));
    lapack::ArgPack_trtri trtriArgs(lapack::Order::AlapackColumnMajor, (dir == 'L' ? lapack::UpLo::AlapackLower : lapack::UpLo::AlapackUpper), lapack::Diag::AlapackNonUnit);
    lapack::engine::_potrf(&deepBaseCase[0],finalDim,finalDim,potrfArgs);
    std::vector<T> deepBaseCaseInv = deepBaseCase;              // true copy because we have to, unless we want to iterate (see below) two different times
    lapack::engine::_trtri(&deepBaseCaseInv[0],finalDim,finalDim,trtriArgs);

    // Only truly a "square-to-square" serialization because we store matrixL as a square (no packed storage yet!)

    // Now, before we can serialize into matrixL and matrixLI, we need to save the values that this processor owns according to the cyclic rule.
    // Only then can we serialize.

    // Iterate and pick out. I would like not to have to create any more memory and I would only like to iterate once, not twice, for storeL and storeLI
    //   Use the "overwrite" trick that I have used in CASI code, as well as other places

    // I am going to use a sneaky trick: I will take the vectorData from storeL and storeLI by reference, overwrite its values,
    //   and then "move" them cheaply into new Matrix structures before I call invoke on them individually.

    // re-serialize with zeros
    std::vector<T> deepBaseCaseFill(checkDim*checkDim,0);
    std::vector<T> deepBaseCaseInvFill(checkDim*checkDim,0);
    // manual serialize
    for (U i=0; i<finalDim; i++){
      for (U j=0; j<finalDim; j++){
        deepBaseCaseFill[i*checkDim+j] = deepBaseCase[i*finalDim+j];
        deepBaseCaseInvFill[i*checkDim+j] = deepBaseCaseInv[i*finalDim+j];
      }
    }

    cyclicToLocalTransformation(deepBaseCaseFill, deepBaseCaseInvFill, localDimension, globalDimension, globalDimension/*bcDimension*/, CommInfo.d, rankSlice, dir);
    // "Inject" the first part of these vectors into Matrices (Square Structure is the only option for now)
    //   This is a bit sneaky, since the vector we "move" into the Matrix has a larger size than the Matrix knows, but with the right member
    //    variables, this should be ok.

    matrix<T,U,square,Distribution,Offload> tempMat(std::move(deepBaseCaseFill), localDimension, localDimension, globalDimension, globalDimension, true);
    matrix<T,U,square,Distribution,Offload> tempMatInv(std::move(deepBaseCaseInvFill), localDimension, localDimension, globalDimension, globalDimension, true);

    // invoke into the existing Matrix data structures owned by the user
//      if (tempRank == 0) { std::cout << "check these 4 numbers - " << matLstartX << "," << matLendX << "," << matLstartY << "," << matLendY << std::endl;}
    serialize<square,square>::invoke(matrixA, tempMat, (dir == 'L' ? matAstartX : matAstartY), (dir == 'L' ? matAendX : matAendY),
      (dir == 'L' ? matAstartX : matAstartY), (dir == 'L' ? matAendX : matAendY), true);
    serialize<square,square>::invoke(matrixI, tempMatInv, matIstartX, matIendX, matIstartY, matIendY, true);
  }
  else{
    U fTranDim1 = localDimension*CommInfo.d;
    std::vector<T> storeMat = cyclic_data;	// TODO: Expensive copy?
    // Until then, assume a double datatype and simply use LAPACKE_dpotrf. Worry about adding more capabilities later.
    lapack::ArgPack_potrf potrfArgs(lapack::Order::AlapackColumnMajor, (dir == 'L' ? lapack::UpLo::AlapackLower : lapack::UpLo::AlapackUpper));
    lapack::ArgPack_trtri trtriArgs(lapack::Order::AlapackColumnMajor, (dir == 'L' ? lapack::UpLo::AlapackLower : lapack::UpLo::AlapackUpper), lapack::Diag::AlapackNonUnit);
    lapack::engine::_potrf(&storeMat[0],fTranDim1,fTranDim1,potrfArgs);
    std::vector<T> storeMatInv = storeMat;		// true copy because we have to, unless we want to iterate (see below) two different times
    lapack::engine::_trtri(&storeMatInv[0],fTranDim1,fTranDim1,trtriArgs);

    // Only truly a "square-to-square" serialization because we store matrixL as a square (no packed storage yet!)

    // Now, before we can serialize into matrixL and matrixLI, we need to save the values that this processor owns according to the cyclic rule.
    // Only then can we serialize.

    // Iterate and pick out. I would like not to have to create any more memory and I would only like to iterate once, not twice, for storeL and storeLI
    //   Use the "overwrite" trick that I have used in CASI code, as well as other places

    // I am going to use a sneaky trick: I will take the vectorData from storeL and storeLI by reference, overwrite its values,
    //   and then "move" them cheaply into new Matrix structures before I call invoke on them individually.

    cyclicToLocalTransformation(storeMat, storeMatInv, localDimension, globalDimension, globalDimension/*bcDimension*/, CommInfo.d, rankSlice, dir);

    // "Inject" the first part of these vectors into Matrices (Square Structure is the only option for now)
    //   This is a bit sneaky, since the vector we "move" into the Matrix has a larger size than the Matrix knows, but with the right member
    //    variables, this should be ok.

    matrix<T,U,square,Distribution,Offload> tempMat(std::move(storeMat), localDimension, localDimension, globalDimension, globalDimension, true);
    matrix<T,U,square,Distribution,Offload> tempMatInv(std::move(storeMatInv), localDimension, localDimension, globalDimension, globalDimension, true);

    // invoke into the existing Matrix data structures owned by the user
    serialize<square,square>::invoke(matrixA, tempMat, (dir == 'L' ? matAstartX : matAstartY), (dir == 'L' ? matAendX : matAendY),
      (dir == 'L' ? matAstartX : matAstartY), (dir == 'L' ? matAendX : matAendY), true);
    serialize<square,square>::invoke(matrixI, tempMatInv, matIstartX, matIendX, matIstartY, matIendY, true);
  }
  return;
}


template<typename MatrixType, typename BaseCaseMatrixType, typename CommType>
void
cholinv::aggregate(MatrixType& matA, BaseCaseMatrixType& matrix_base_case, std::vector<typename MatrixType::ScalarType>& blocked_data,
                   std::vector<typename MatrixType::ScalarType>& cyclic_data, typename MatrixType::DimensionType localDimension, typename MatrixType::DimensionType globalDimension,
                               typename MatrixType::DimensionType bcDimension, typename MatrixType::DimensionType matAstartX, typename MatrixType::DimensionType matAendX,
                               typename MatrixType::DimensionType matAstartY, typename MatrixType::DimensionType matAendY, CommType&& CommInfo, char dir){

  assert(matrix_base_case.getNumColumnsLocal() == localDimension);
  using T = typename MatrixType::ScalarType;
  using U = typename MatrixType::DimensionType;
  using Distribution = typename MatrixType::DistributionType;
  using Offload = typename MatrixType::OffloadType;
  using BaseCaseStructure = typename BaseCaseMatrixType::StructureType;

  serialize<square,BaseCaseStructure>::invoke(matA, matrix_base_case, matAstartX, matAendX, matAstartY, matAendY);
  if (CommInfo.num_chunks == 0 || (CommInfo.num_chunks > localDimension)){
    MPI_Allgather(matrix_base_case.getRawData(), matrix_base_case.getNumElems(), mpi_type<T>::type, &blocked_data[0], matrix_base_case.getNumElems(), mpi_type<T>::type, CommInfo.slice);
    util::block_to_cyclic(blocked_data, cyclic_data, localDimension, localDimension, CommInfo.d, dir);
  }
  else{
    // initiate distribution of allgather into chunks of local columns, multiples of localDimension
    std::vector<MPI_Request> req(CommInfo.num_chunks);
    std::vector<MPI_Status> stat(CommInfo.num_chunks);
    U offset = localDimension*(localDimension%CommInfo.num_chunks);
    U progress=0;
    for (size_t idx=0; idx < CommInfo.num_chunks; idx++){
      MPI_Iallgather(matrix_base_case.getRawData()+progress, idx==(CommInfo.num_chunks-1) ? localDimension*(localDimension/CommInfo.num_chunks+offset) : localDimension*(localDimension/CommInfo.num_chunks),
                     mpi_type<T>::type, &blocked_data[progress], idx==(CommInfo.num_chunks-1) ? localDimension*(localDimension/CommInfo.num_chunks+offset) : localDimension*(localDimension/CommInfo.num_chunks),
                     mpi_type<T>::type, CommInfo.slice, &req[idx]);
      progress += localDimension * (localDimension/CommInfo.num_chunks);
    }
    // initiate distribution along columns and complete distribution across rows
    progress=0;
    for (size_t idx=0; idx < CommInfo.num_chunks; idx++){
      MPI_Wait(&req[idx],&stat[idx]);
      util::block_to_cyclic(&blocked_data[progress], &cyclic_data[progress], localDimension,
                            idx==(CommInfo.num_chunks-1) ? (localDimension+offset)/CommInfo.num_chunks : localDimension/CommInfo.num_chunks, CommInfo.d);
      progress += (localDimension * (localDimension/CommInfo.num_chunks))*CommInfo.d*CommInfo.d;
    }
  }
}


// This method can be called from Lower and Upper with one tweak, but note that currently, we iterate over the entire square,
//   when we are really only writing to a triangle. So there is a source of optimization here at least in terms of
//   number of flops, but in terms of memory accesses and cache lines, not sure. Note that with this optimization,
//   we may need to separate into two different functions
template<typename T, typename U>
void cholinv::cyclicToLocalTransformation(std::vector<T>& storeT, std::vector<T>& storeTI, U localDimension, U globalDimension, U bcDimension, int64_t sliceDim, int64_t rankSlice, char dir){

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
//        if (storeT[writeIndex] != 0) std::cout << "val - " << storeT[writeIndex] << std::endl;
        storeT[writeIndex] = 0.;
        storeTI[writeIndex] = 0.;
      }
      writeIndex++;
    }
  }
}

template<typename U>
void cholinv::updateInversePath(U inverseCutoffGlobalDimension, U globalDimension, bool& isInversePath, std::vector<U>& baseCaseDimList, U localDimension){
  if (inverseCutoffGlobalDimension >= globalDimension){
    if (isInversePath == false){
      baseCaseDimList.push_back(localDimension);
    }
    isInversePath = true;
  }
}

template<typename MatrixAType, typename MatrixUType, typename MatrixUIType, typename CommType>
void cholinv::iSolveUpperLeft(MatrixAType& matrixA, MatrixUType& matrixU, MatrixUIType& matrixUI, CommType&& CommInfo,
                                 std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                                 blas::ArgPack_gemm<typename MatrixAType::ScalarType>& gemmPackage){

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using StructureTri = typename MatrixUType::StructureType;
  using Distribution = typename MatrixAType::DistributionType;
  using Offload = typename MatrixAType::OffloadType;

  blas::ArgPack_trmm<T> trmmPackage(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasLower,
      blas::Transpose::AblasTrans, blas::Diag::AblasNonUnit, 1.);
  // to catch debugging issues, assert that this has at least one size
  assert(baseCaseDimList.size());

  // Lets operate on individual columns at a time
  // Potential optimization 1): Don't use MM3D if the columns are too skinny in relation to the block size!
     // Or this could just be taken care of when we tune block sizes?
  // Potential optimization 2) Lots of serializing going on with each MM3D, this needs to be reduced.

  // Communicate matrixA and matrixU and matrixUI immediately.
    // These 3 matrices should never need to be communicated again.
  // matrixB however will need to be AllReduced at each iteration so that final results can be summed and updated before next iteration

  U matAendX = matrixA.getNumColumnsLocal();
  U matAendY = matrixA.getNumRowsLocal();
  U matUendX = matrixU.getNumColumnsLocal();

  U offset1 = 0;
  U offset2 = (baseCaseDimList.size() < 1 ? matAendX : baseCaseDimList[0]);
  U offset3 = 0;
  for (U i=0; i<baseCaseDimList.size()/*numBlockColumns*/; i++){
    // Update the current column by accumulating the updates via MM
    gemmPackage.alpha = -1;
    gemmPackage.beta = 1.;

    // Only update once first panel is solved
    if (i>0){
      // As i increases, the size of these updates gets smaller.
      // Special handling. This might only work since the triangular matrix is square, which should be ok
      U arg1 = (gemmPackage.transposeB == blas::Transpose::AblasNoTrans ? offset1 : offset3);
      U arg2 = (gemmPackage.transposeB == blas::Transpose::AblasNoTrans ? matUendX : offset1);
      U arg3 = (gemmPackage.transposeB == blas::Transpose::AblasNoTrans ? offset3 : offset1);
      U arg4 = (gemmPackage.transposeB == blas::Transpose::AblasNoTrans ? offset1 : matUendX);

      matrix<T,U,rect,Distribution,Offload> matrixUpartition(std::vector<T>(), arg2-arg1, arg4-arg3, (arg2-arg1)*CommInfo.d, (arg4-arg3)*CommInfo.d);
      serialize<StructureTri,rect>::invoke(matrixU, matrixUpartition, arg1, arg2, arg3, arg4);
      matmult::summa::invoke(matrixA.getRawData()+(offset3*matAendY), matrixUpartition, matrixA.getRawData()+(offset1*matAendY),
        offset1-offset3, matAendY, arg2-arg1, arg4-arg3, matAendX-offset1, matAendY, std::forward<CommType>(CommInfo), gemmPackage);
    }

    // Solve via TRMM
    U save1 = offset2-offset1;
    // New optimization: prevent this copy if we are doing TRSM only at the top level
    // Note: this change might be rendered useless now that I modified CFR3D.hpp with a similar optimization for that top level of TRSM
    if (baseCaseDimList.size() <= 1){
      matmult::summa::invoke(matrixUI, matrixA.getRawData()+(offset1*matAendY), save1, save1, save1, matAendY, std::forward<CommType>(CommInfo), trmmPackage);
    }
    else{
      matrix<T,U,StructureTri,Distribution,Offload> matrixUIpartition(std::vector<T>(), save1, save1, save1*CommInfo.d, save1*CommInfo.d);
      serialize<StructureTri,StructureTri>::invoke(matrixUI, matrixUIpartition, offset1, offset2, offset1, offset2);
      matmult::summa::invoke(matrixUIpartition, matrixA.getRawData()+(offset1*matAendY), save1, save1, save1, matAendY, std::forward<CommType>(CommInfo), trmmPackage);
    }
    if ((i+1) < baseCaseDimList.size()){
      // Update the offsets
      offset3 = offset1;
      offset1 = offset2;
      offset2 += baseCaseDimList[i+1];
    }
  }
}

// For solving LA=B for A. But note that B is being modified in place and will turn into A
template<typename MatrixLType, typename MatrixLIType, typename MatrixAType, typename CommType>
void cholinv::iSolveLowerRight(MatrixLType& matrixL, MatrixLIType& matrixLI, MatrixAType& matrixA, CommType&& CommInfo,
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

  U matAendX = matrixA.getNumColumnsLocal();
  U matAendY = matrixA.getNumRowsLocal();
  U matLendX = matrixL.getNumColumnsLocal();

  U offset1 = 0;
  U offset2 = (baseCaseDimList.size() < 1 ? matAendX : baseCaseDimList[0]);
  U offset3 = 0;
  for (U i=0; i<baseCaseDimList.size()/*numBlockColumns*/; i++){

    // Update the current column by accumulating the updates via MM
    gemmPackage.alpha = -1;
    gemmPackage.beta = 1.;

    // Only update once first panel is solved
    if (i>0){
      // As i increases, the size of these updates gets smaller.
      // Special handling. This might only work since the triangular matrix is square, which should be ok

      // Note that the beginning cases might not be correct. They are not currently used for anything though.
      U arg1 = (gemmPackage.transposeA == blas::Transpose::AblasNoTrans ? offset3 : offset1);
      U arg2 = (gemmPackage.transposeA == blas::Transpose::AblasNoTrans ? offset1 : matLendX);
      U arg3 = (gemmPackage.transposeA == blas::Transpose::AblasNoTrans ? offset3 : offset3);
      U arg4 = (gemmPackage.transposeA == blas::Transpose::AblasNoTrans ? offset1 : offset1);

      matrix<T,U,rect,Distribution,Offload> matrixLpartition(std::vector<T>(), arg2-arg1, arg4-arg3, (arg2-arg1)*CommInfo.d, (arg4-arg3)*CommInfo.d);
      serialize<StructureTri,rect>::invoke(matrixL, matrixLpartition, arg1, arg2, arg3, arg4);
      U zero = 0;
      matmult::summa::invoke(matrixLpartition, matrixA, matrixA, zero, arg2-arg1, zero, arg4-arg3, zero, matAendX, offset3, offset1,
        zero, matAendX, offset1, matAendY, std::forward<CommType>(CommInfo), gemmPackage, false, true, true);
    }

    // Solve via MM
    matmult::summa::invoke(matrixLI, matrixA, offset1, offset2, offset1, offset2, 0, matAendX, offset1, offset2, std::forward<CommType>(CommInfo), trmmPackage, true, true);

    if ((i+1) < baseCaseDimList.size()){
      // Update the offsets
      offset3 = offset1;
      offset1 = offset2;
      offset2 += baseCaseDimList[i+1];
    }
  }
}

}
