/* Author: Edward Hutter */

namespace trsm{
/*
template<typename MatrixAType, typename MatrixTriType, typename CommType>
void diaginvert::iSolveLowerLeft(MatrixAType& matrixA, MatrixTriType& matrixL, MatrixTriType& matrixLI, CommType&& CommInfo,
                                 std::vector<typename MatrixTriType::DimensionType>& baseCaseDimList,
                                 blas::ArgPack_gemm<typename MatrixTriType::ScalarType>& gemmPackage){
}
*/

// For solving AU=B for A. But note that B is being modified in place and will turn into A
template<typename MatrixAType, typename MatrixTriType, typename CommType>
void diaginvert::iSolveUpperLeft(MatrixAType& matrixA, MatrixTriType& matrixU, MatrixTriType& matrixUI, CommType&& CommInfo,
                                 std::vector<typename MatrixTriType::DimensionType>& baseCaseDimList,
                                 blas::ArgPack_gemm<typename MatrixTriType::ScalarType>& gemmPackage){
  TAU_FSTART(diaginvert::iSolveUpperLeft);

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using StructureTri = typename MatrixTriType::StructureType;
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
  TAU_FSTOP(diaginvert::iSolveUpperLeft);
}


// For solving RA=A for A
template<typename MatrixTriType, typename MatrixAType, typename CommType>
void diaginvert::iSolveLowerRight(MatrixTriType& matrixL, MatrixTriType& matrixLI, MatrixAType& matrixA, CommType&& CommInfo,
                                  std::vector<typename MatrixTriType::DimensionType>& baseCaseDimList,
                                  blas::ArgPack_gemm<typename MatrixTriType::ScalarType>& gemmPackage){
  TAU_FSTART(diaginvert::iSolveLowerRight);

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using StructureTri = typename MatrixTriType::StructureType;
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
  TAU_FSTOP(diaginvert::iSolveLowerRight);
}

/*
template<typename MatrixTriType, typename MatrixAType, typename CommType>
void diaginvert::iSolveUpperRight(MatrixTriType& matrixU, MatrixTriType& matrixUI, MatrixAType& matrixA, CommType&& CommInfo,
                                  std::vector<typename MatrixTriType::DimensionType>& baseCaseDimList,
                                  blas::ArgPack_gemm<typename MatrixTriType::ScalarType>& gemmPackage){
}
*/

template<typename MatrixAType, typename MatrixBType, typename MatrixCType, typename CommType>
void diaginvert::invoke(MatrixAType& matrixA, MatrixBType& matrixB, MatrixCType& matrixC, CommType&& CommInfo,
                         char UpLo, char Dir, std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                         blas::ArgPack_gemm<typename MatrixAType::ScalarType>& gemmPackage){
  if ((UpLo=='L') && (Dir=='L')){
    //iSolveLowerLeft(matrixA,matrixB,matrixC,std::forward<CommType>(CommInfo),baseCaseDimList,gemmPackage);
  }
  else if ((UpLo=='U') && (Dir=='L')){
    iSolveUpperLeft(matrixA,matrixB,matrixC,std::forward<CommType>(CommInfo),baseCaseDimList,gemmPackage);
  }
  else if ((UpLo=='L') && (Dir=='R')){
    iSolveLowerRight(matrixA,matrixB,matrixC,std::forward<CommType>(CommInfo),baseCaseDimList,gemmPackage);
  }
  else if ((UpLo=='U') && (Dir=='R')){
    //iSolveUpperRight(matrixA,matrixB,matrixC,std::forward<CommType>(CommInfo),baseCaseDimList,gemmPackage);
  }
}
}
