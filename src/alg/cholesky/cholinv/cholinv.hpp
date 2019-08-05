/* Author: Edward Hutter */

namespace cholesky{
template<typename MatrixAType, typename MatrixTIType, typename CommType>
std::pair<bool,std::vector<typename MatrixAType::DimensionType>>
cholinv::invoke(MatrixAType& matrixA, MatrixTIType& matrixTI, CommType&& CommInfo, typename MatrixAType::DimensionType inverseCutOffGlobalDimension,
                typename MatrixAType::DimensionType blockSizeMultiplier, typename MatrixAType::DimensionType panelDimensionMultiplier, char dir){
  TAU_FSTART(cholinv::invoke);

  using U = typename MatrixAType::DimensionType;
  U localDimension = matrixA.getNumRowsLocal();
  U globalDimension = matrixA.getNumRowsGlobal();
  // the division below may have a remainder, but I think integer division will be ok, as long as we change the base case condition to be <= and not just ==
  U bcDimension = globalDimension/helper;

  for (size_t i=0; i<blockSizeMultiplier; i++){
    bcDimension *= 2;
  }

  U save = inverseCutOffGlobalDimension;
  inverseCutOffGlobalDimension = globalDimension;
  for (size_t i=0; i<save; i++){
    inverseCutOffGlobalDimension >>= 1;
  }
  inverseCutOffGlobalDimension = std::max(localDimension*2,inverseCutOffGlobalDimension);

  save = panelDimensionMultiplier;
  panelDimensionMultiplier = bcDimension;
  for (size_t i=0; i<save; i++){
    panelDimensionMultiplier <<= 1;
  }
  panelDimensionMultiplier = std::min(localDimension, panelDimensionMultiplier);
  std::pair<bool,std::vector<U>> baseCaseDimList;

  if (dir == 'L'){
    baseCaseDimList.first = (inverseCutOffGlobalDimension >= globalDimension ? true : false);
//    if (isInversePath) { baseCaseDimList.push_back(localDimension); }
    rFactorLower(matrixA, matrixTI, localDimension, localDimension, bcDimension, globalDimension, globalDimension,
      0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, std::forward<CommType>(CommInfo),
      baseCaseDimList.first, baseCaseDimList.second, inverseCutOffGlobalDimension, panelDimensionMultiplier);
  }
  else if (dir == 'U'){
    baseCaseDimList.first = (inverseCutOffGlobalDimension >= globalDimension ? true : false);
//    if (isInversePath) { baseCaseDimList.push_back(localDimension); }
    rFactorUpper(matrixA, matrixTI, localDimension, localDimension, bcDimension, globalDimension, globalDimension,
      0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, std::forward<CommType>(CommInfo),
      baseCaseDimList.first, baseCaseDimList.second, inverseCutOffGlobalDimension, panelDimensionMultiplier);
  }
  TAU_FSTOP(cholinv::invoke);
  return baseCaseDimList;
}

template<typename MatrixAType, typename MatrixLIType, typename CommType>
void cholinv::rFactorLower(MatrixAType& matrixA, MatrixLIType& matrixLI, typename MatrixAType::DimensionType localDimension, typename MatrixAType::DimensionType trueLocalDimension,
                         typename MatrixAType::DimensionType bcDimension, typename MatrixAType::DimensionType globalDimension, typename MatrixAType::DimensionType trueGlobalDimension,
                         typename MatrixAType::DimensionType matAstartX, typename MatrixAType::DimensionType matAendX, typename MatrixAType::DimensionType matAstartY,
                         typename MatrixAType::DimensionType matAendY, typename MatrixAType::DimensionType matLIstartX, typename MatrixAType::DimensionType matLIendX,
                         typename MatrixAType::DimensionType matLIstartY, typename MatrixAType::DimensionType matLIendY,
                         CommType&& CommInfo, bool& isInversePath, std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                         typename MatrixAType::DimensionType inverseCutoffGlobalDimension, typename MatrixAType::DimensionType panelDimension){
  TAU_FSTART(cholinv::rFactorLower);

  if (globalDimension <= bcDimension){
    baseCase(matrixA, matrixLI, localDimension, trueLocalDimension, bcDimension, globalDimension, trueGlobalDimension,
      matAstartX, matAendX, matAstartY, matAendY, matLIstartX, matLIendX, matLIstartY, matLIendY,
      std::forward<CommType>(CommInfo), isInversePath, baseCaseDimList, inverseCutoffGlobalDimension, panelDimension, 'L');
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
  rFactorLower(matrixA, matrixLI, localShift, trueLocalDimension, bcDimension, globalShift, trueGlobalDimension,
    matAstartX, matAstartX+localShift, matAstartY, matAstartY+localShift,
    matLIstartX, matLIstartX+localShift, matLIstartY, matLIstartY+localShift,
    std::forward<CommType>(CommInfo), isInversePath, baseCaseDimList, inverseCutoffGlobalDimension, panelDimension);

  size_t saveIndexAfter = baseCaseDimList.size();

  // Regardless of whether or not we need to communicate for the transpose, we still need to serialize into a buffer
  matrix<T,U,lowertri,Distribution,Offload> packedMatrix(std::vector<T>(), localShift, localShift, globalShift, globalShift);
  // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
  serialize<square,lowertri>::invoke(matrixLI, packedMatrix, matLIstartX, matLIstartX+localShift, matLIstartY, matLIstartY+localShift);
  util::transposeSwap(packedMatrix, CommInfo.world);

  blasEngineArgumentPackage_trmm<T> trmmArgs(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasRight, blasEngineUpLo::AblasLower,
    blasEngineTranspose::AblasTrans, blasEngineDiag::AblasNonUnit, 1.);

  // 2nd case: Extra optimization for the case when we only perform TRSM at the top level.
  if ((isInversePath) || (globalDimension == inverseCutoffGlobalDimension*2)){
    //std::cout << "tell me localDim and localshIFT - " << localDimension << " " << localShift << std::endl;
    matmult::summa::invoke(packedMatrix, matrixA, 0, localShift, 0, localShift, matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY, std::forward<CommType>(CommInfo), trmmArgs, false, true);
  }
  else{
    // Note: keep this a gemm package, because we still need to use gemm in TRSM3D in the update, which is just rectangular and non-triangular matrices.
    blasEngineArgumentPackage_gemm<T> trsmArgs(blasEngineOrder::AblasColumnMajor, blasEngineTranspose::AblasNoTrans, blasEngineTranspose::AblasTrans, 1., 0.);

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
    util::transposeSwap(packedMatrixL, CommInfo.world);

    blasEngineArgumentPackage_trmm<T> trmmPackage(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasRight, blasEngineUpLo::AblasLower,
      blasEngineTranspose::AblasTrans, blasEngineDiag::AblasNonUnit, 1.);
    trsm::diaginvert::invoke(matrixLcopy, packedMatrixL, packedMatrix, std::forward<CommType>(CommInfo) 'U', 'L', subBaseCaseDimList, trsmArgs);

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

  blasEngineArgumentPackage_syrk<T> syrkArgs(blasEngineOrder::AblasColumnMajor, blasEngineUpLo::AblasLower, blasEngineTranspose::AblasNoTrans, -1., 1.);
  matmult::summa::invoke(matrixA, matrixA, matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY,
    matAstartX+localShift, matAendX, matAstartY+localShift, matAendY, std::forward<CommType>(CommInfo), syrkArgs, true, true);

  rFactorLower(matrixA, matrixLI, reverseDimLocal, trueLocalDimension, bcDimension, reverseDimGlobal/*globalShift*/, trueGlobalDimension,
    matAstartX+localShift, matAendX, matAstartY+localShift, matAendY,
    matLIstartX+localShift, matLIendX, matLIstartY+localShift, matLIendY,
    std::forward<CommType>(CommInfo), isInversePath, baseCaseDimList, inverseCutoffGlobalDimension, panelDimension);

  if (isInversePath){
    // Next step : temp <- L_{21}*LI_{11}
    // Tradeoff: By encapsulating the transpose/serialization of L21 in the syrk summa routine above, I can't reuse that buffer and must re-serialize L21
    matrix<T,U,square,Distribution,Offload> tempInverse(std::vector<T>(), localShift, reverseDimLocal, globalShift, reverseDimGlobal);
    // NOTE: WE BROKE SQUARE SEMANTICS WITH THIS. CHANGE LATER!
    serialize<square,square>::invoke(matrixA, tempInverse, matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY);

    blasEngineArgumentPackage_trmm<T> invPackage1(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasRight, blasEngineUpLo::AblasLower,
      blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
    matmult::summa::invoke(matrixLI, tempInverse, matLIstartX, matLIstartX+localShift, matLIstartY,
        matLIstartY+localShift, 0, localShift, 0, reverseDimLocal, std::forward<CommType>(CommInfo), invPackage1, true, false);

    // Next step: finish the Triangular inverse calculation
    invPackage1.alpha = -1.;
    invPackage1.side = blasEngineSide::AblasLeft;
    matmult::summa::invoke(matrixLI, tempInverse, matLIstartX+localShift, matLIendX, matLIstartY+localShift, matLIendY, 0, localShift, 0, reverseDimLocal,
                             std::forward<CommType>(CommInfo), invPackage1, true, false);
    // One final serialize of tempInverse into matrixLI
    serialize<square,square>::invoke(matrixLI, tempInverse, matLIstartX, matLIstartX+localShift, matLIstartY+localShift, matLIendY, true);
  }
  isInversePath = saveSwitch;
  TAU_FSTOP(cholinv::rFactorLower);
}


template<typename MatrixAType, typename MatrixRIType, typename CommType>
void cholinv::rFactorUpper(MatrixAType& matrixA, MatrixRIType& matrixRI, typename MatrixAType::DimensionType localDimension, typename MatrixAType::DimensionType trueLocalDimension,
                         typename MatrixAType::DimensionType bcDimension, typename MatrixAType::DimensionType globalDimension, typename MatrixAType::DimensionType trueGlobalDimension,
                         typename MatrixAType::DimensionType matAstartX, typename MatrixAType::DimensionType matAendX, typename MatrixAType::DimensionType matAstartY,
                         typename MatrixAType::DimensionType matAendY, typename MatrixAType::DimensionType matRIstartX, typename MatrixAType::DimensionType matRIendX,
                         typename MatrixAType::DimensionType matRIstartY, typename MatrixAType::DimensionType matRIendY,
                         CommType&& CommInfo, bool& isInversePath, std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                         typename MatrixAType::DimensionType inverseCutoffGlobalDimension, typename MatrixAType::DimensionType panelDimension){
  TAU_FSTART(cholinv::rFactorUpper);

  if (globalDimension <= bcDimension){
    baseCase(matrixA, matrixRI, localDimension, trueLocalDimension, bcDimension, globalDimension, trueGlobalDimension,
      matAstartX, matAendX, matAstartY, matAendY, matRIstartX, matRIendX, matRIstartY, matRIendY,
      std::forward<CommType>(CommInfo), isInversePath, baseCaseDimList, inverseCutoffGlobalDimension, panelDimension, 'U');
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
  rFactorUpper(matrixA, matrixRI, localShift, trueLocalDimension, bcDimension, globalShift, trueGlobalDimension,
    matAstartX, matAstartX+localShift, matAstartY, matAstartY+localShift,
    matRIstartX, matRIstartX+localShift, matRIstartY, matRIstartY+localShift,
    std::forward<CommType>(CommInfo), isInversePath, baseCaseDimList, inverseCutoffGlobalDimension, panelDimension);

  size_t saveIndexAfter = baseCaseDimList.size();

  // Regardless of whether or not we need to communicate for the transpose, we still need to serialize into a square buffer
  matrix<T,U,uppertri,Distribution,Offload> packedMatrix(std::vector<T>(), localShift, localShift, globalShift, globalShift);
  // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
  serialize<square,uppertri>::invoke(matrixRI, packedMatrix, matRIstartX, matRIstartX+localShift, matRIstartY, matRIstartY+localShift);
  util::transposeSwap(packedMatrix, CommInfo.world);
  blasEngineArgumentPackage_trmm<T> trmmArgs(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasLeft, blasEngineUpLo::AblasUpper,
    blasEngineTranspose::AblasTrans, blasEngineDiag::AblasNonUnit, 1.);

  // 2nd case: Extra optimization for the case when we only perform TRSM at the top level.
  if ((isInversePath) || (globalDimension == inverseCutoffGlobalDimension*2)){
    matmult::summa::invoke(packedMatrix, matrixA, 0, localShift, 0, localShift, matAstartX+localShift, matAendX, matAstartY, matAstartY+localShift,
      std::forward<CommType>(CommInfo), trmmArgs, false, true);
  }
  else{
    blasEngineArgumentPackage_gemm<T> trsmArgs(blasEngineOrder::AblasColumnMajor, blasEngineTranspose::AblasTrans, blasEngineTranspose::AblasNoTrans, 1., 0.);

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
    Matrix<T,U,uppertri,Distribution,Offload> packedMatrixR(std::vector<T>(), localShift, localShift, globalShift, globalShift);
    // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
    serialize<square,uppertri>::invoke(matrixA, packedMatrixR, matAstartX, matAstartX+localShift, matAstartY, matAstartY+localShift);
    // Swap, same as we did with inverse
    util::transposeSwap(packedMatrixR, CommInfo.world);

    blasEngineArgumentPackage_trmm<T> trmmPackage(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasLeft, blasEngineUpLo::AblasUpper,
      blasEngineTranspose::AblasTrans, blasEngineDiag::AblasNonUnit, 1.);
    trsm::diaginvert::invoke(packedMatrixR, packedMatrix, matrixRcopy, std::forward<CommType>(CommInfo), 'L', 'R', subBaseCaseDimList, trsmArgs);

    // Inject back into matrixR
    serialize<square,square>::invoke(matrixA, matrixRcopy, matAstartX+localShift, matAendX, matAstartY, matAstartY+localShift, true);
  }

  U reverseDimLocal = localDimension-localShift;
  U reverseDimGlobal = reverseDimLocal*CommInfo.d;

  blasEngineArgumentPackage_syrk<T> syrkArgs(blasEngineOrder::AblasColumnMajor, blasEngineUpLo::AblasUpper, blasEngineTranspose::AblasTrans, -1., 1.);
  matmult::summa::invoke(matrixA, matrixA, matAstartX+localShift, matAendX, matAstartY, matAstartY+localShift,
    matAstartX+localShift, matAendX, matAstartY+localShift, matAendY, std::forward<CommType>(CommInfo), syrkArgs, true, true);

  rFactorUpper(matrixA, matrixRI, reverseDimLocal, trueLocalDimension, bcDimension, reverseDimGlobal/*globalShift*/, trueGlobalDimension,
    matAstartX+localShift, matAendX, matAstartY+localShift, matAendY,
    matRIstartX+localShift, matRIendX, matRIstartY+localShift, matRIendY,
    std::forward<CommType>(CommInfo), isInversePath, baseCaseDimList, inverseCutoffGlobalDimension, panelDimension);

  // Next step : temp <- R_{12}*RI_{22}
  // We can re-use holdRsyrk as our temporary output matrix.

  if (isInversePath){
    // Tradeoff: By encapsulating the transpose/serialization of L21 in the syrk summa routine above, I can't reuse that buffer and must re-serialize L21
    matrix<T,U,square,Distribution,Offload> tempInverse(std::vector<T>(), reverseDimLocal, localShift, reverseDimGlobal, globalShift);
    // NOTE: WE BROKE SQUARE SEMANTICS WITH THIS. CHANGE LATER!
    serialize<square,square>::invoke(matrixA, tempInverse, matAstartX+localShift, matAendX, matAstartY, matAstartY+localShift);

    blasEngineArgumentPackage_trmm<T> invPackage1(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasRight, blasEngineUpLo::AblasUpper,
      blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
    matmult::summa::invoke(matrixRI, tempInverse, matRIstartX+localShift, matRIendX, matRIstartY+localShift, matRIendY, 0, reverseDimLocal, 0, localShift,
      std::forward<CommType>(CommInfo), invPackage1, true, false);

    // Next step: finish the Triangular inverse calculation
    invPackage1.alpha = -1.;
    invPackage1.side = blasEngineSide::AblasLeft;
    matmult::summa::invoke(matrixRI, tempInverse, matRIstartX, matRIstartX+localShift, matRIstartY, matRIstartY+localShift, 0, reverseDimLocal, 0, localShift,
      std::forward<CommType>(CommInfo), invPackage1, true, false);
    serialize<square,square>::invoke(matrixRI, tempInverse, matRIstartX+localShift, matRIendX, matRIstartY, matRIstartY+localShift, true);
  }
  isInversePath = saveSwitch;
  TAU_FSTOP(cholinv::rFactorUpper);
}


template<typename MatrixAType, typename MatrixIType, typename CommType>
void cholinv::baseCase(MatrixAType& matrixA, MatrixIType& matrixI, typename MatrixAType::DimensionType localDimension, typename MatrixAType::DimensionType trueLocalDimension,
                     typename MatrixAType::DimensionType bcDimension, typename MatrixAType::DimensionType globalDimension, typename MatrixAType::DimensionType trueGlobalDimension,
                     typename MatrixAType::DimensionType matAstartX, typename MatrixAType::DimensionType matAendX, typename MatrixAType::DimensionType matAstartY,
                     typename MatrixAType::DimensionType matAendY, typename MatrixAType::DimensionType matIstartX, typename MatrixAType::DimensionType matIendX,
                     typename MatrixAType::DimensionType matIstartY, typename MatrixAType::DimensionType matIendY,
                     CommType&& CommInfo, bool& isInversePath, std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                     typename MatrixAType::DimensionType inverseCutoffGlobalDimension, typename MatrixAType::DimensionType panelDimension, char dir){
  TAU_FSTART(cholinv::baseCase);

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

  MPI_Comm_rank(CommInfo.slice, &rankSlice);

  // Should be fast pass-by-value via move semantics
  std::vector<T> cyclicBaseCaseData = blockedToCyclicTransformation(
    matrixA, localDimension, globalDimension, globalDimension/*bcDimension*/, (dir == 'L' ? matAstartX : matAstartY), (dir == 'L' ? matAendX : matAendY),
    (dir == 'L' ? matAstartX : matAstartY), (dir == 'L' ? matAendX : matAendY), CommInfo.d, CommInfo.slice, dir);

  // TODO: Note: with my new optimizations, this case might never pass, because A is serialized into. Watch out!
  if (((dir == 'L') && (matAendX == trueLocalDimension)) || ((dir == 'U') && (matAendY == trueLocalDimension))){
    //U finalDim = trueLocalDimension*CommInfo.d - trueGlobalDimension;
    U checkDim = localDimension*CommInfo.d;
    U finalDim = (checkDim - (trueLocalDimension*CommInfo.d - trueGlobalDimension));

    std::vector<T> deepBaseCase(finalDim*finalDim,0);
    // manual serialize
    for (U i=0; i<finalDim; i++){
      for (U j=0; j<finalDim; j++){
        deepBaseCase[i*finalDim+j] = cyclicBaseCaseData[i*checkDim+j];
      }
    }

    lapackEngineArgumentPackage_potrf potrfArgs(lapackEngineOrder::AlapackColumnMajor, (dir == 'L' ? lapackEngineUpLo::AlapackLower : lapackEngineUpLo::AlapackUpper));
    lapackEngineArgumentPackage_trtri trtriArgs(lapackEngineOrder::AlapackColumnMajor, (dir == 'L' ? lapackEngineUpLo::AlapackLower : lapackEngineUpLo::AlapackUpper), lapackEngineDiag::AlapackNonUnit);
    lapackEngine::_potrf(&deepBaseCase[0],finalDim,finalDim,potrfArgs);
    std::vector<T> deepBaseCaseInv = deepBaseCase;              // true copy because we have to, unless we want to iterate (see below) two different times
    lapackEngine::_trtri(&deepBaseCaseInv[0],finalDim,finalDim,trtriArgs);

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
    size_t fTranDim1 = localDimension*CommInfo.d;
    std::vector<T>& storeMat = cyclicBaseCaseData;
    // Until then, assume a double datatype and simply use LAPACKE_dpotrf. Worry about adding more capabilities later.
    lapackEngineArgumentPackage_potrf potrfArgs(lapackEngineOrder::AlapackColumnMajor, (dir == 'L' ? lapackEngineUpLo::AlapackLower : lapackEngineUpLo::AlapackUpper));
    lapackEngineArgumentPackage_trtri trtriArgs(lapackEngineOrder::AlapackColumnMajor, (dir == 'L' ? lapackEngineUpLo::AlapackLower : lapackEngineUpLo::AlapackUpper), lapackEngineDiag::AlapackNonUnit);
    lapackEngine::_potrf(&storeMat[0],fTranDim1,fTranDim1,potrfArgs);
    std::vector<T> storeMatInv = storeMat;		// true copy because we have to, unless we want to iterate (see below) two different times
    lapackEngine::_trtri(&storeMatInv[0],fTranDim1,fTranDim1,trtriArgs);

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
    Matrix<T,U,square,Distribution,Offload> tempMatInv(std::move(storeMatInv), localDimension, localDimension, globalDimension, globalDimension, true);

    // invoke into the existing Matrix data structures owned by the user
    serialize<square,square>::invoke(matrixA, tempMat, (dir == 'L' ? matAstartX : matAstartY), (dir == 'L' ? matAendX : matAendY),
      (dir == 'L' ? matAstartX : matAstartY), (dir == 'L' ? matAendX : matAendY), true);
    serialize<square,square>::invoke(matrixI, tempMatInv, matIstartX, matIendX, matIstartY, matIendY, true);
  }
  TAU_FSTOP(cholinv::baseCase);
  return;
}


template<typename MatrixType>
std::vector<typename MatrixType::ScalarType>
cholinv::blockedToCyclicTransformation(MatrixType& matA, typename MatrixType::DimensionType localDimension, typename MatrixType::DimensionType globalDimension,
                                     typename MatrixType::DimensionType bcDimension, typename MatrixType::DimensionType matAstartX, typename MatrixType::DimensionType matAendX,
                                     typename MatrixType::DimensionType matAstartY, typename MatrixType::DimensionType matAendY, size_t sliceDim, MPI_Comm slice2Dcomm, char dir){
  TAU_FSTART(cholinv::blockedToCyclicTransformation);

  using T = typename MatrixType::ScalarType;
  using U = typename MatrixType::DimensionType;
  using Distribution = typename MatrixType::DistributionType;
  using Offload = typename MatrixType::OffloadType;

  if (dir == 'U'){
    matrix<T,U,uppertri,Distribution,Offload> baseCaseMatrixA(std::vector<T>(), localDimension, localDimension, globalDimension, globalDimension);
    serialize<square,uppertri>::invoke(matA, baseCaseMatrixA, matAstartX, matAendX, matAstartY, matAendY);
  //  U aggregDim = localDimension*sliceDim;
  //  std::vector<T> blockedBaseCaseData(aggregDim*aggregDim);
    U aggregSize = baseCaseMatrixA.getNumElems()*sliceDim*sliceDim;
    std::vector<T> blockedBaseCaseData(aggregSize);
    // Note: recv buffer will be larger tha send buffer * sliceDim**2! This should not crash, but we need this much memory anyway when calling DPOTRF and DTRTRI
    MPI_Allgather(baseCaseMatrixA.getRawData(), baseCaseMatrixA.getNumElems(), typename mpi_type<T>::type, &blockedBaseCaseData[0], baseCaseMatrixA.getNumElems(), typename mpi_type<T>::type, slice2Dcomm);

    TAU_FSTOP(cholinv::blockedToCyclicTransformation);
    return util::blockedToCyclicSpecial(blockedBaseCaseData, localDimension, localDimension, sliceDim, dir);
  }
  else{ // dir == 'L'
    matrix<T,U,lowertri,Distribution,Offload> baseCaseMatrixA(std::vector<T>(), localDimension, localDimension, globalDimension, globalDimension);
    serialize<square,lowertri>::invoke(matA, baseCaseMatrixA, matAstartX, matAendX, matAstartY, matAendY);
  //  U aggregDim = localDimension*sliceDim;
  //  std::vector<T> blockedBaseCaseData(aggregDim*aggregDim);
    U aggregSize = baseCaseMatrixA.getNumElems()*sliceDim*sliceDim;
    std::vector<T> blockedBaseCaseData(aggregSize);
    // Note: recv buffer will be larger tha send buffer * sliceDim**2! This should not crash, but we need this much memory anyway when calling DPOTRF and DTRTRI
    MPI_Allgather(baseCaseMatrixA.getRawData(), baseCaseMatrixA.getNumElems(), typename mpi_type<T>::type, &blockedBaseCaseData[0], baseCaseMatrixA.getNumElems(), typename mpi_type<T>::type, slice2Dcomm);

    TAU_FSTOP(cholinv::blockedToCyclicTransformation);
    return util::blockedToCyclicSpecial(blockedBaseCaseData, localDimension, localDimension, sliceDim, dir);
  }
}


// This method can be called from Lower and Upper with one tweak, but note that currently, we iterate over the entire square,
//   when we are really only writing to a triangle. So there is a source of optimization here at least in terms of
//   number of flops, but in terms of memory accesses and cache lines, not sure. Note that with this optimization,
//   we may need to separate into two different functions
template<typename T, typename U>
void cholinv::cyclicToLocalTransformation(std::vector<T>& storeT, std::vector<T>& storeTI, U localDimension, U globalDimension, U bcDimension, size_t sliceDim, size_t rankSlice, char dir){
  TAU_FSTART(cholinv::cyclicToLocalTransformation);

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
  TAU_FSTOP(cholinv::cyclicToLocalTransformation);
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
}
