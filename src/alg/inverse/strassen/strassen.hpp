/* Author: Edward Hutter */

namespace inverse{
template<typename MatrixType, typename CommType>
void strassen::invoke(MatrixType& matrix, CommType&& CommInfo, typename MatrixType::DimensionType NewtonDimension){
  TAU_FSTART(strassen::invoke);

  using U = typename MatrixType::DimensionType;
  U localDimension = matrix.getNumRowsLocal();
  U globalDimension = matrix.getNumRowsGlobal();
  // the division below may have a remainder, but I think integer division will be ok, as long as we change the base case condition to be <= and not just ==
  U bcDimension = globalDimension/(CommInfo.c*CommInfo.d);

  baseCaseDimList.first = (inverseCutOffGlobalDimension >= globalDimension ? true : false);
  invert(matrix, localDimension, localDimension, bcDimension, NewtonDimension, globalDimension, globalDimension,
    0, localDimension, 0, localDimension, std::forward<CommType>(CommInfo));
  TAU_FSTOP(strassen::invoke);
  return;
}

template<typename MatrixType, typename CommType>
void strassen::invert(MatrixType& matrix, typename MatrixType::DimensionType localDimension, typename MatrixType::DimensionType trueLocalDimension,
                         typename MatrixType::DimensionType bcDimension, typename MatrixType::DimensionType NewtonDimension, typename MatrixType::DimensionType globalDimension,
                         typename MatrixType::DimensionType trueGlobalDimension, typename MatrixType::DimensionType matAstartX,
                         typename MatrixType::DimensionType matAendX, typename MatrixType::DimensionType matAstartY,
                         typename MatrixType::DimensionType matAendY, CommType&& CommInfo){
  TAU_FSTART(strassen::invert);

  if (globalDimension <= NewtonDimension){
    newton::invoke(matrix,std::forward<CommType>(CommInfo));//TODO: Pass in a tolerance and max sweeps
  }
  if (globalDimension <= bcDimension){
    baseCase(matrix, localDimension, trueLocalDimension, bcDimension, globalDimension, trueGlobalDimension,
      matAstartX, matAendX, matAstartY, matAendY, std::forward<CommType>(CommInfo));
    return;
  }

  using T = typename MatrixType::ScalarType;
  using U = typename MatrixType::DimensionType;
  using Distribution = typename MatrixType::DistributionType;
  using Offload = typename MatrixType::OffloadType;

  int rank;
  MPI_Comm_rank(CommInfo.world, &rank);
  // globalDimension will always be a power of 2, but localDimension won't
  U localShift = (localDimension>>1);
  // move localShift up to the next power of 2, only useful if matrix dimensions are not powers of 2
  localShift = util::getNextPowerOf2(localShift);
  U globalShift = (globalDimension>>1);

  invert(matrix, localShift, trueLocalDimension, bcDimension, NewtonDimension, globalShift, trueGlobalDimension,
    matAstartX, matAstartX+localShift, matAstartY, matAstartY+localShift, std::forward<CommType>(CommInfo));

  // Regardless of whether or not we need to communicate for the transpose, we still need to serialize into a buffer
  matrix<T,U,lowertri,Distribution,Offload> packedMatrix(std::vector<T>(), localShift, localShift, globalShift, globalShift);
  // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
  serialize<square,lowertri>::invoke(matrixLI, packedMatrix, matLIstartX, matLIstartX+localShift, matLIstartY, matLIstartY+localShift);
  util::transposeSwap(packedMatrix, std::forward<CommType>(CommInfo));

  blas::ArgPack_trmm<T> trmmArgs(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasLower,
    blas::Transpose::AblasTrans, blas::Diag::AblasNonUnit, 1.);
  matmult::summa::invoke(packedMatrix, matrixA, 0, localShift, 0, localShift, matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY, std::forward<CommType>(CommInfo), trmmArgs, false, true);

  // Now we need to perform L_{21}L_{21}^T via syrk
  //   Actually, I am havin trouble with SYRK, lets try gemm instead
  // As of January 2017, still having trouble with SYRK.

  // Later optimization: avoid this recalculation at each recursive level, since it will always be the same.
  U reverseDimLocal = localDimension-localShift;
  U reverseDimGlobal = reverseDimLocal*CommInfo.d;

  blas::ArgPack_syrk<T> syrkArgs(blas::Order::AblasColumnMajor, blas::UpLo::AblasLower, blas::Transpose::AblasNoTrans, -1., 1.);
  matmult::summa::invoke(matrixA, matrixA, matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY,
    matAstartX+localShift, matAendX, matAstartY+localShift, matAendY, std::forward<CommType>(CommInfo), syrkArgs, true, true);

  invert(matrix, reverseDimLocal, trueLocalDimension, bcDimension, NewtonDimension, reverseDimGlobal/*globalShift*/, trueGlobalDimension,
    matAstartX+localShift, matAendX, matAstartY+localShift, matAendY, std::forward<CommType>(CommInfo));

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
  TAU_FSTOP(strassen::invert);
}


template<typename MatrixType, typename CommType>
void strassen::baseCase(MatrixType& matrix, typename MatrixType::DimensionType localDimension, typename MatrixType::DimensionType trueLocalDimension,
                     typename MatrixType::DimensionType bcDimension, typename MatrixType::DimensionType globalDimension, typename MatrixType::DimensionType trueGlobalDimension,
                     typename MatrixType::DimensionType matAstartX, typename MatrixType::DimensionType matAendX, typename MatrixType::DimensionType matAstartY,
                     typename MatrixType::DimensionType matAendY, CommType&& CommInfo){
  TAU_FSTART(strassen::baseCase);

  using T = typename MatrixType::ScalarType;
  using U = typename MatrixType::DimensionType;
  using Distribution = typename MatrixType::DistributionType;
  using Offload = typename MatrixType::OffloadType;

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
  std::vector<T> cyclicBaseCaseData = blockedToCyclicTransformation(
    matrix, localDimension, globalDimension, globalDimension/*bcDimension*/, (dir == 'L' ? matAstartX : matAstartY), (dir == 'L' ? matAendX : matAendY),
    (dir == 'L' ? matAstartX : matAstartY), (dir == 'L' ? matAendX : matAendY), CommInfo.d, CommInfo.slice);

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
    size_t fTranDim1 = localDimension*CommInfo.d;
    std::vector<T>& storeMat = cyclicBaseCaseData;
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
  TAU_FSTOP(strassen::baseCase);
  return;
}


template<typename MatrixType>
std::vector<typename MatrixType::ScalarType>
strassen::blockedToCyclicTransformation(MatrixType& matA, typename MatrixType::DimensionType localDimension, typename MatrixType::DimensionType globalDimension,
                                     typename MatrixType::DimensionType bcDimension, typename MatrixType::DimensionType matAstartX, typename MatrixType::DimensionType matAendX,
                                     typename MatrixType::DimensionType matAstartY, typename MatrixType::DimensionType matAendY, size_t sliceDim, MPI_Comm slice2Dcomm, char dir){
  TAU_FSTART(strassen::blockedToCyclicTransformation);

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
    MPI_Allgather(baseCaseMatrixA.getRawData(), baseCaseMatrixA.getNumElems(), mpi_type<T>::type, &blockedBaseCaseData[0], baseCaseMatrixA.getNumElems(), mpi_type<T>::type, slice2Dcomm);

    TAU_FSTOP(strassen::blockedToCyclicTransformation);
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
    MPI_Allgather(baseCaseMatrixA.getRawData(), baseCaseMatrixA.getNumElems(), mpi_type<T>::type, &blockedBaseCaseData[0], baseCaseMatrixA.getNumElems(), mpi_type<T>::type, slice2Dcomm);

    TAU_FSTOP(strassen::blockedToCyclicTransformation);
    return util::blockedToCyclicSpecial(blockedBaseCaseData, localDimension, localDimension, sliceDim, dir);
  }
}


// This method can be called from Lower and Upper with one tweak, but note that currently, we iterate over the entire square,
//   when we are really only writing to a triangle. So there is a source of optimization here at least in terms of
//   number of flops, but in terms of memory accesses and cache lines, not sure. Note that with this optimization,
//   we may need to separate into two different functions
template<typename T, typename U>
void strassen::cyclicToLocalTransformation(std::vector<T>& storeT, std::vector<T>& storeTI, U localDimension, U globalDimension, U bcDimension, size_t sliceDim, size_t rankSlice, char dir){
  TAU_FSTART(strassen::cyclicToLocalTransformation);

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
  TAU_FSTOP(strassen::cyclicToLocalTransformation);
}

}
