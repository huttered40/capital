/* Author: Edward Hutter */


template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
std::vector<U> CFR3D<T,U,blasEngine>::Factor(
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixTI,
  U inverseCutOffGlobalDimension,
  char dir,
  int tune,
  MPI_Comm commWorld,
  std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D,
  int MMid,
  int TSid
  )
{
  TAU_FSTART(CFR3D::Factor);
  // Need to split up the commWorld communicator into a 3D grid similar to Summa3D
  int pGridDimensionSize;
  MPI_Comm_size(std::get<0>(commInfo3D), &pGridDimensionSize);


  int helper = pGridDimensionSize;
  helper *= helper;
  int pGridCoordX = std::get<4>(commInfo3D);
  int pGridCoordY = std::get<5>(commInfo3D);
  int pGridCoordZ = std::get<6>(commInfo3D);
  int transposePartner = pGridCoordZ*helper + pGridCoordX*pGridDimensionSize + pGridCoordY;

  U localDimension = matrixA.getNumRowsLocal();
  U globalDimension = matrixA.getNumRowsGlobal();
  // the division below may have a remainder, but I think integer division will be ok, as long as we change the base case condition to be <= and not just ==
  U bcDimension = globalDimension/helper;		// Can be tuned later.

  // Basic tuner was added
  for (int i=0; i<tune; i++)
  {
    bcDimension *= 2;
  }
  bcDimension = std::min(bcDimension, globalDimension/pGridDimensionSize);

  int save = inverseCutOffGlobalDimension;
  inverseCutOffGlobalDimension = globalDimension;
  for (int i=0; i<save; i++)
  {
    inverseCutOffGlobalDimension >>= 1;
  }
  inverseCutOffGlobalDimension = std::max(bcDimension*2,inverseCutOffGlobalDimension);

  std::vector<U> baseCaseDimList;

  if (dir == 'L')
  {
    bool isInversePath = (inverseCutOffGlobalDimension >= globalDimension ? true : false);
    if (isInversePath) { baseCaseDimList.push_back(localDimension); }
    rFactorLower(
      matrixA, matrixTI, localDimension, localDimension, bcDimension, globalDimension, globalDimension,
      0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, transposePartner, MMid, TSid, commWorld, commInfo3D, isInversePath, baseCaseDimList, inverseCutOffGlobalDimension);
  }
  else if (dir == 'U')
  {
    bool isInversePath = (inverseCutOffGlobalDimension >= globalDimension ? true : false);
    if (isInversePath) { baseCaseDimList.push_back(localDimension); }
    rFactorUpper(
      matrixA, matrixTI, localDimension, localDimension, bcDimension, globalDimension, globalDimension,
      0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, transposePartner, MMid, TSid, commWorld, commInfo3D, isInversePath, baseCaseDimList, inverseCutOffGlobalDimension);
  }

  TAU_FSTOP(CFR3D::Factor);
  return baseCaseDimList;
}

template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CFR3D<T,U,blasEngine>::rFactorLower(
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixLI,
  U localDimension,
  U trueLocalDimension,
  U bcDimension,
  U globalDimension,
  U trueGlobalDimension,
  U matAstartX,
  U matAendX,
  U matAstartY,
  U matAendY,
  U matLIstartX,
  U matLIendX,
  U matLIstartY,
  U matLIendY,
  U transposePartner,
  int MM_id,
  int TS_id,
  MPI_Comm commWorld, 	// We want to pass in commWorld as MPI_COMM_WORLD because we want to pass that into 3D MM
  std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D,
  bool& isInversePath,
  std::vector<U>& baseCaseDimList,
  U inverseCutoffGlobalDimension)
{
  TAU_FSTART(CFR3D::rFactorLower);
  if (localDimension <= bcDimension)
  {
    if (!isInversePath)
    {
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

    int rankSlice,sizeSlice,pGridDimensionSize;
    MPI_Comm_size(std::get<0>(commInfo3D), &pGridDimensionSize);
    MPI_Comm_rank(std::get<2>(commInfo3D), &rankSlice);
    sizeSlice = pGridDimensionSize*pGridDimensionSize;

    // Should be fast pass-by-value via move semantics
    std::vector<T> cyclicBaseCaseData = blockedToCyclicTransformation(
      matrixA, localDimension, globalDimension, globalDimension/*bcDimension*/, matAstartX, matAendX,
      matAstartY, matAendY, pGridDimensionSize, std::get<2>(commInfo3D), 'L');

    // Now, I want to use something similar to a template class for libraries conforming to the standards of LAPACK, such as FLAME.
    //   I want to be able to mix and match.

    // TODO: Note: with my new optimizations, this case might never pass, because A is serialized into. Watch out!
    if ((matAendX == trueLocalDimension) && (matAendY == trueLocalDimension))
    {
      //U finalDim = trueLocalDimension*pGridDimensionSize - trueGlobalDimension;
      U checkDim = localDimension*pGridDimensionSize;
      U finalDim = (checkDim - (trueLocalDimension*pGridDimensionSize - trueGlobalDimension));
      std::vector<T> deepBaseCase(finalDim*finalDim,0);
      // manual serialize
      for (U i=0; i<finalDim; i++)
      {
        for (U j=0; j<finalDim; j++)
        {
          deepBaseCase[i*finalDim+j] = cyclicBaseCaseData[i*checkDim+j];
        }
      }
      // Until then, assume a double datatype and simply use LAPACKE_dpotrf. Worry about adding more capabilities later.
      LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', finalDim/*bcDimension*/, &deepBaseCase[0], finalDim/*bcDimension*/);
      // Now, we have L_{11} located inside the "square" vector cyclicBaseCaseData.
      //   We need to call the "move builder" constructor in order to "move" this "rawData" into its own matrix.
      //   Only then, can we call Serializer into the real matrixL. WRONG! We need to find the data we own according to the cyclic rule first!
      //    So it doesn't make any sense to "move" these into Matrices yet.
      // Finally, we need that data for calling the triangular inverse.

      // Next: sequential triangular inverse. Question: does DTRTRI require packed storage or square storage? I think square, so that it can use BLAS-3.
      std::vector<T> deepBaseCaseInv = deepBaseCase;		// true copy because we have to, unless we want to iterate (see below) two different times
      LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'L', 'N', finalDim/*bcDimension*/, &deepBaseCaseInv[0], finalDim/*bcDimension*/);
      // Only truly a "square-to-square" serialization because we store matrixL as a square (no packed storage yet!)

      // Now, before we can serialize into matrixL and matrixLI, we need to save the values that this processor owns according to the cyclic rule.
      // Only then can we serialize.

      // Iterate and pick out. I would like not to have to create any more memory and I would only like to iterate once, not twice, for storeL and storeLI
      //   Use the "overwrite" trick that I have used in CASI code, as well as other places

      // I am going to use a sneaky trick: I will take the vectorData from storeL and storeLI by reference, overwrite its values,
      //   and then "move" them cheaply into new Matrix structures before I call Serialize on them individually.

      // re-serialize with zeros
      std::vector<T> deepBaseCaseFill(checkDim*checkDim,0);
      std::vector<T> deepBaseCaseInvFill(checkDim*checkDim,0);
      // manual serialize
      for (U i=0; i<finalDim; i++)
      {
        for (U j=0; j<finalDim; j++)
        {
          deepBaseCaseFill[i*checkDim+j] = deepBaseCase[i*finalDim+j];
          deepBaseCaseInvFill[i*checkDim+j] = deepBaseCaseInv[i*finalDim+j];
        }
      }

      cyclicToLocalTransformation(
        deepBaseCaseFill, deepBaseCaseInvFill, localDimension, globalDimension, globalDimension/*bcDimension*/, pGridDimensionSize, rankSlice, 'L');
      // "Inject" the first part of these vectors into Matrices (Square Structure is the only option for now)
      //   This is a bit sneaky, since the vector we "move" into the Matrix has a larger size than the Matrix knows, but with the right member
      //    variables, this should be ok.

      Matrix<T,U,MatrixStructureSquare,Distribution> tempL(std::move(deepBaseCaseFill), localDimension, localDimension, globalDimension, globalDimension, true);
      Matrix<T,U,MatrixStructureSquare,Distribution> tempLI(std::move(deepBaseCaseInvFill), localDimension, localDimension, globalDimension, globalDimension, true);

      // Serialize into the existing Matrix data structures owned by the user
//      if (tempRank == 0) { std::cout << "check these 4 numbers - " << matLstartX << "," << matLendX << "," << matLstartY << "," << matLendY << std::endl;}
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixA, tempL, matAstartX, matAendX, matAstartY, matAendY, true);
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixLI, tempLI, matLIstartX, matLIendX, matLIstartY, matLIendY, true);
    }
    else
    {
      std::vector<T>& storeL = cyclicBaseCaseData;

      // Until then, assume a double datatype and simply use LAPACKE_dpotrf. Worry about adding more capabilities later.
      LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', localDimension*pGridDimensionSize/*bcDimension*/, &storeL[0], localDimension*pGridDimensionSize/*bcDimension*/);

      // Now, we have L_{11} located inside the "square" vector cyclicBaseCaseData.
      //   We need to call the "move builder" constructor in order to "move" this "rawData" into its own matrix.
      //   Only then, can we call Serializer into the real matrixL. WRONG! We need to find the data we own according to the cyclic rule first!
      //    So it doesn't make any sense to "move" these into Matrices yet.
      // Finally, we need that data for calling the triangular inverse.

      // Next: sequential triangular inverse. Question: does DTRTRI require packed storage or square storage? I think square, so that it can use BLAS-3.
      std::vector<T> storeLI = storeL;		// true copy because we have to, unless we want to iterate (see below) two different times
      LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'L', 'N', localDimension*pGridDimensionSize/*bcDimension*/, &storeLI[0], localDimension*pGridDimensionSize/*bcDimension*/);

      // Only truly a "square-to-square" serialization because we store matrixL as a square (no packed storage yet!)

      // Now, before we can serialize into matrixL and matrixLI, we need to save the values that this processor owns according to the cyclic rule.
      // Only then can we serialize.

      // Iterate and pick out. I would like not to have to create any more memory and I would only like to iterate once, not twice, for storeL and storeLI
      //   Use the "overwrite" trick that I have used in CASI code, as well as other places

      // I am going to use a sneaky trick: I will take the vectorData from storeL and storeLI by reference, overwrite its values,
      //   and then "move" them cheaply into new Matrix structures before I call Serialize on them individually.

      cyclicToLocalTransformation(
        storeL, storeLI, localDimension, globalDimension, globalDimension/*bcDimension*/, pGridDimensionSize, rankSlice, 'L');

      // "Inject" the first part of these vectors into Matrices (Square Structure is the only option for now)
      //   This is a bit sneaky, since the vector we "move" into the Matrix has a larger size than the Matrix knows, but with the right member
      //    variables, this should be ok.

      Matrix<T,U,MatrixStructureSquare,Distribution> tempL(std::move(storeL), localDimension, localDimension, globalDimension, globalDimension, true);
      Matrix<T,U,MatrixStructureSquare,Distribution> tempLI(std::move(storeLI), localDimension, localDimension, globalDimension, globalDimension, true);

      // Serialize into the existing Matrix data structures owned by the user
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixA, tempL, matAstartX, matAendX, matAstartY, matAendY, true);
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixLI, tempLI, matLIstartX, matLIendX, matLIstartY, matLIendY, true);
    }
    return;
  }

  int rank;
  MPI_Comm_rank(commWorld, &rank);
  // globalDimension will always be a power of 2, but localDimension won't
  U localShift = (localDimension>>1);
  // move localShift up to the next power of 2
  localShift = util<T,U>::getNextPowerOf2(localShift);
  U globalShift = (globalDimension>>1);
  bool saveSwitch = isInversePath;
  int saveIndexPrev = baseCaseDimList.size();
  if (inverseCutoffGlobalDimension >= globalDimension)
  {
    if (isInversePath == false)
    {
      baseCaseDimList.push_back(localShift);
    }
    isInversePath = true;
  }
  rFactorLower(
    matrixA, matrixLI, localShift, trueLocalDimension, bcDimension, globalShift, trueGlobalDimension,
    matAstartX, matAstartX+localShift, matAstartY, matAstartY+localShift,
    matLIstartX, matLIstartX+localShift, matLIstartY, matLIstartY+localShift, transposePartner, MM_id, TS_id,
    commWorld, commInfo3D, isInversePath, baseCaseDimList, inverseCutoffGlobalDimension);

  isInversePath = saveSwitch;
  int saveIndexAfter = baseCaseDimList.size();

  // Regardless of whether or not we need to communicate for the transpose, we still need to serialize into a buffer
  Matrix<T,U,MatrixStructureLowerTriangular,Distribution> packedMatrix(std::vector<T>(), localShift, localShift, globalShift, globalShift);
  // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
  Serializer<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular>::Serialize(matrixLI, packedMatrix,
    matLIstartX, matLIstartX+localShift, matLIstartY, matLIstartY+localShift);
  util<T,U>::transposeSwap(
    packedMatrix, rank, transposePartner, commWorld);

  blasEngineArgumentPackage_trmm<T> trmmArgs;
  trmmArgs.order = blasEngineOrder::AblasColumnMajor;
  trmmArgs.side = blasEngineSide::AblasRight;
  trmmArgs.uplo = blasEngineUpLo::AblasLower;
  trmmArgs.diag = blasEngineDiag::AblasNonUnit;
  trmmArgs.transposeA = blasEngineTranspose::AblasTrans;
  trmmArgs.alpha = 1.;

  if (isInversePath)
  {
    MM3D<T,U,blasEngine>::Multiply(
      packedMatrix, matrixA, 0, localShift, 0, localShift, matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY, commWorld, commInfo3D, trmmArgs, false, true, MM_id);
  }
  else
  {
    // try this for now
    if (inverseCutoffGlobalDimension*2 == trueGlobalDimension)
    {
      MM3D<T,U,blasEngine>::Multiply(
        packedMatrix, matrixA, 0, localShift, 0, localShift, matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY, commWorld, commInfo3D, trmmArgs, false, true, MM_id);
    }
    else
    {
      // Note: keep this a gemm package, because we still need to use gemm in TRSM3D in the update, which is just rectangular and non-triangular matrices.
      blasEngineArgumentPackage_gemm<T> trsmArgs;
      trsmArgs.order = blasEngineOrder::AblasColumnMajor;
      trsmArgs.transposeA = blasEngineTranspose::AblasNoTrans;
      trsmArgs.transposeB = blasEngineTranspose::AblasTrans;

      // create a new subvector
      U len = saveIndexAfter - saveIndexPrev;
      std::vector<U> subBaseCaseDimList(len);
      for (U i=saveIndexPrev; i<saveIndexAfter; i++)
      {
        subBaseCaseDimList[i-saveIndexPrev] = baseCaseDimList[i];
      }

      // TODO: Note: some of those steps are unnecessary if we are doing TRSM3D to one level deep.
      // make extra copy to avoid corrupting matrixA
      // Note: some of these globalShifts are wrong, but I don't know any easy way to fix them. Everything might still work though.
      Matrix<T,U,MatrixStructureSquare,Distribution> matrixLcopy(std::vector<T>(), localShift, matAendY-(matAstartY+localShift), globalShift, globalShift);
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixA, matrixLcopy,
        matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY);
      // Also need to serialize top-left quadrant of matrixL so that its size matches packedMatrix
      Matrix<T,U,MatrixStructureLowerTriangular,Distribution> packedMatrixL(std::vector<T>(), localShift, localShift, globalShift, globalShift);
      // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
      Serializer<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular>::Serialize(matrixA, packedMatrixL,
        matAstartX, matAstartX+localShift, matAstartY, matAstartY+localShift);
      // Swap, same as we did with inverse
      util<T,U>::transposeSwap(
        packedMatrixL, rank, transposePartner, commWorld);
      TRSM3D<T,U,blasEngine>::iSolveUpperLeft(
        matrixLcopy, packedMatrixL, packedMatrix,
        subBaseCaseDimList, trsmArgs, commWorld, commInfo3D, MM_id, TS_id);

      // inject matrixLcopy back into matrixA.
      // Future optimization: avoid copying matrixL here, and utilize leading dimension and the column vectors.
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixA, matrixLcopy,
        matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY, true);
    }
  }

  // Now we need to perform L_{21}L_{21}^T via syrk
  //   Actually, I am havin trouble with SYRK, lets try gemm instead
  // As of January 2017, still having trouble with SYRK.

  // Later optimization: avoid this recalculation at each recursive level, since it will always be the same.
  int pGridDimensionSize;
  MPI_Comm_size(std::get<0>(commInfo3D), &pGridDimensionSize);
  U reverseDimLocal = localDimension-localShift;
  U reverseDimGlobal = reverseDimLocal*pGridDimensionSize;

  // TODO: Might be able to re-use a buffer from above instead of creating squareL, packedMatrix, but not that packedMatrix was of size localShift, and we need reverseDimLocal, which is not always the same (if we have a bad data-to-grid fit)
  Matrix<T,U,MatrixStructureSquare,Distribution> squareL(std::vector<T>(), localShift, reverseDimLocal, globalShift, reverseDimGlobal);
  // NOTE: WE BROKE SQUARE SEMANTICS WITH THIS. CHANGE LATER!
  Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixA, squareL,
    matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY);
  Matrix<T,U,MatrixStructureSquare,Distribution> squareLSwap = squareL;

  util<T,U>::transposeSwap(
    squareLSwap, rank, transposePartner, commWorld);

  blasEngineArgumentPackage_gemm<T> blasArgs;
  blasArgs.order = blasEngineOrder::AblasColumnMajor;
  blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
  blasArgs.transposeB = blasEngineTranspose::AblasTrans;
  blasArgs.alpha = -1;
  blasArgs.beta = 1;
  MM3D<T,U,blasEngine>::Multiply(
    squareL, squareLSwap, matrixA, 0, localShift, 0, reverseDimLocal, 0, localShift, 0, reverseDimLocal,
    matAstartX+localShift, matAendX, matAstartY+localShift, matAendY, commWorld, commInfo3D, blasArgs, false, false, true, MM_id);

  // Only need to change the argument for matrixA
  saveSwitch = isInversePath;
  if (inverseCutoffGlobalDimension >= globalDimension)
  {
    if (isInversePath == false)
    {
      baseCaseDimList.push_back(localShift);
    }
    isInversePath = true;
  }

  rFactorLower(
    matrixA, matrixLI, reverseDimLocal, trueLocalDimension, bcDimension, reverseDimGlobal/*globalShift*/, trueGlobalDimension,
    matAstartX+localShift, matAendX, matAstartY+localShift, matAendY,
    matLIstartX+localShift, matLIendX, matLIstartY+localShift, matLIendY, transposePartner, MM_id, TS_id,
    commWorld, commInfo3D, isInversePath, baseCaseDimList, inverseCutoffGlobalDimension);

  isInversePath = saveSwitch;

  if (isInversePath)
  {

    // Next step : temp <- L_{21}*LI_{11}
    // We can re-use squareL as our temporary output matrix.

    Matrix<T,U,MatrixStructureSquare,Distribution>& tempInverse = squareL/*holdLsyrk*/;

    blasEngineArgumentPackage_trmm<T> invPackage1;
    invPackage1.order = blasEngineOrder::AblasColumnMajor;
    invPackage1.side = blasEngineSide::AblasRight;
    invPackage1.uplo = blasEngineUpLo::AblasLower;
    invPackage1.diag = blasEngineDiag::AblasNonUnit;
    invPackage1.transposeA = blasEngineTranspose::AblasNoTrans;
    invPackage1.alpha = 1.;
    MM3D<T,U,blasEngine>::Multiply(
      matrixLI, tempInverse, matLIstartX, matLIstartX+localShift, matLIstartY,
        matLIstartY+localShift, 0, localShift, 0, reverseDimLocal, commWorld, commInfo3D, invPackage1, true, false, MM_id);

    // Next step: finish the Triangular inverse calculation
    invPackage1.alpha = -1.;
    invPackage1.side = blasEngineSide::AblasLeft;
    MM3D<T,U,blasEngine>::Multiply(matrixLI, tempInverse, matLIstartX+localShift, matLIendX, matLIstartY+localShift, matLIendY, 0, localShift, 0, reverseDimLocal,
        commWorld, commInfo3D, invPackage1, true, false, MM_id);
    // One final serialize of tempInverse into matrixLI
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixLI, tempInverse,
      matLIstartX, matLIstartX+localShift, matLIstartY+localShift, matLIendY, true);
  }
  TAU_FSTOP(CFR3D::rFactorLower);
}


template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CFR3D<T,U,blasEngine>::rFactorUpper(
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixRI,
                       U localDimension,
                       U trueLocalDimension,
                       U bcDimension,
                       U globalDimension,
                       U trueGlobalDimension,
                       U matAstartX,
                       U matAendX,
                       U matAstartY,
                       U matAendY,
                       U matRIstartX,
                       U matRIendX,
                       U matRIstartY,
                       U matRIendY,
                       U transposePartner,
                       int MM_id,
                       int TS_id,
                       MPI_Comm commWorld,
                       std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D,
                       bool& isInversePath,
                       std::vector<U>& baseCaseDimList,
                       U inverseCutoffGlobalDimension
                     )
{
  TAU_FSTART(CFR3D::rFactorUpper);
  if (localDimension <= bcDimension)
  {
    if (!isInversePath)
    {
      // Only save if we never got onto the inverse path
      baseCaseDimList.push_back(localDimension);
    }


    if (localDimension == 0) return;
    // First: AllGather matrix A so that every processor has the same replicated diagonal square partition of matrix A of dimension bcDimension
    //          Note that processors only want to communicate with those on their same 2D slice, since the matrices are replicated on every slice
    //          Note that before the AllGather, we need to serialize the matrix A into the small square matrix
    // Second: Data will be received in a blocked order due to AllGather semantics, which is not what we want. We need to get back to cyclic again
    //           This is an ugly process, as it was in the last code.
    // Third: Once data is in cyclic format, we call call sequential Cholesky Factorization and Triangular Inverse.
    // Fourth: Save the data that each processor owns according to the cyclic rule.

    int rankSlice,sizeSlice,pGridDimensionSize;
    MPI_Comm_size(std::get<0>(commInfo3D), &pGridDimensionSize);
    MPI_Comm_rank(std::get<2>(commInfo3D), &rankSlice);
    sizeSlice = pGridDimensionSize*pGridDimensionSize;

    // Should be fast pass-by-value via move semantics
    std::vector<T> cyclicBaseCaseData = blockedToCyclicTransformation(
      matrixA, localDimension, globalDimension, globalDimension/*bcDimension*/, matAstartX, matAendX,
      matAstartY, matAendY, pGridDimensionSize, std::get<2>(commInfo3D), 'U');

    // Now, I want to use something similar to a template class for libraries conforming to the standards of LAPACK, such as FLAME.
    //   I want to be able to mix and match.

    if ((matAendX == trueLocalDimension) && (matAendY == trueLocalDimension))
    {
      //U finalDim = trueLocalDimension*pGridDimensionSize - trueGlobalDimension;
      U checkDim = localDimension*pGridDimensionSize;
      U finalDim = (checkDim - (trueLocalDimension*pGridDimensionSize - trueGlobalDimension));
      std::vector<T> deepBaseCase(finalDim*finalDim,0);
      // manual serialize
      for (U i=0; i<finalDim; i++)
      {
        for (U j=0; j<finalDim; j++)
        {
          deepBaseCase[i*finalDim+j] = cyclicBaseCaseData[i*checkDim+j];
        }
      }
      // Until then, assume a double datatype and simply use LAPACKE_dpotrf. Worry about adding more capabilities later.
      LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', finalDim/*bcDimension*/, &deepBaseCase[0], finalDim/*bcDimension*/);
      // Now, we have L_{11} located inside the "square" vector cyclicBaseCaseData.
      //   We need to call the "move builder" constructor in order to "move" this "rawData" into its own matrix.
      //   Only then, can we call Serializer into the real matrixL. WRONG! We need to find the data we own according to the cyclic rule first!
      //    So it doesn't make any sense to "move" these into Matrices yet.
      // Finally, we need that data for calling the triangular inverse.

      // Next: sequential triangular inverse. Question: does DTRTRI require packed storage or square storage? I think square, so that it can use BLAS-3.
      std::vector<T> deepBaseCaseInv = deepBaseCase;		// true copy because we have to, unless we want to iterate (see below) two different times
      LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', finalDim/*bcDimension*/, &deepBaseCaseInv[0], finalDim/*bcDimension*/);
      // Only truly a "square-to-square" serialization because we store matrixL as a square (no packed storage yet!)

      // Now, before we can serialize into matrixL and matrixLI, we need to save the values that this processor owns according to the cyclic rule.
      // Only then can we serialize.

      // Iterate and pick out. I would like not to have to create any more memory and I would only like to iterate once, not twice, for storeL and storeLI
      //   Use the "overwrite" trick that I have used in CASI code, as well as other places

      // I am going to use a sneaky trick: I will take the vectorData from storeL and storeLI by reference, overwrite its values,
      //   and then "move" them cheaply into new Matrix structures before I call Serialize on them individually.

      // re-serialize with zeros
      std::vector<T> deepBaseCaseFill(checkDim*checkDim,0);
      std::vector<T> deepBaseCaseInvFill(checkDim*checkDim,0);
      // manual serialize
      for (U i=0; i<finalDim; i++)
      {
        for (U j=0; j<finalDim; j++)
        {
          deepBaseCaseFill[i*checkDim+j] = deepBaseCase[i*finalDim+j];
          deepBaseCaseInvFill[i*checkDim+j] = deepBaseCaseInv[i*finalDim+j];
        }
      }

      cyclicToLocalTransformation(
        deepBaseCaseFill, deepBaseCaseInvFill, localDimension, globalDimension, globalDimension/*bcDimension*/, pGridDimensionSize, rankSlice, 'U');
      // "Inject" the first part of these vectors into Matrices (Square Structure is the only option for now)
      //   This is a bit sneaky, since the vector we "move" into the Matrix has a larger size than the Matrix knows, but with the right member
      //    variables, this should be ok.

      Matrix<T,U,MatrixStructureSquare,Distribution> tempR(std::move(deepBaseCaseFill), localDimension, localDimension, globalDimension, globalDimension, true);
      Matrix<T,U,MatrixStructureSquare,Distribution> tempRI(std::move(deepBaseCaseInvFill), localDimension, localDimension, globalDimension, globalDimension, true);

      // Serialize into the existing Matrix data structures owned by the user
//      if (tempRank == 0) { std::cout << "check these 4 numbers - " << matRstartX << "," << matRendX << "," << matRstartY << "," << matRendY << std::endl;}
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixA, tempR, matAstartX, matAendX, matAstartY, matAendY, true);
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixRI, tempRI, matRIstartX, matRIendX, matRIstartY, matRIendY, true);
    }
    else
    {
      std::vector<T>& storeR = cyclicBaseCaseData;

      // Until then, assume a double datatype and simply use LAPACKE_dpotrf. Worry about adding more capabilities later.
      LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', localDimension*pGridDimensionSize/*bcDimension*/, &storeR[0], localDimension*pGridDimensionSize/*bcDimension*/);

      // Now, we have L_{11} located inside the "square" vector cyclicBaseCaseData.
      //   We need to call the "move builder" constructor in order to "move" this "rawData" into its own matrix.
      //   Only then, can we call Serializer into the real matrixL. WRONG! We need to find the data we own according to the cyclic rule first!
      //    So it doesn't make any sense to "move" these into Matrices yet.
      // Finally, we need that data for calling the triangular inverse.

      // Next: sequential triangular inverse. Question: does DTRTRI require packed storage or square storage? I think square, so that it can use BLAS-3.
      std::vector<T> storeRI = storeR;		// true copy because we have to, unless we want to iterate (see below) two different times
      LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', localDimension*pGridDimensionSize/*bcDimension*/, &storeRI[0], localDimension*pGridDimensionSize/*bcDimension*/);

      // Only truly a "square-to-square" serialization because we store matrixL as a square (no packed storage yet!)

      // Now, before we can serialize into matrixL and matrixLI, we need to save the values that this processor owns according to the cyclic rule.
      // Only then can we serialize.

      // Iterate and pick out. I would like not to have to create any more memory and I would only like to iterate once, not twice, for storeL and storeLI
      //   Use the "overwrite" trick that I have used in CASI code, as well as other places

      // I am going to use a sneaky trick: I will take the vectorData from storeL and storeLI by reference, overwrite its values,
      //   and then "move" them cheaply into new Matrix structures before I call Serialize on them individually.

      cyclicToLocalTransformation(
        storeR, storeRI, localDimension, globalDimension, globalDimension/*bcDimension*/, pGridDimensionSize, rankSlice, 'U');

      // "Inject" the first part of these vectors into Matrices (Square Structure is the only option for now)
      //   This is a bit sneaky, since the vector we "move" into the Matrix has a larger size than the Matrix knows, but with the right member
      //    variables, this should be ok.

      Matrix<T,U,MatrixStructureSquare,Distribution> tempR(std::move(storeR), localDimension, localDimension, globalDimension, globalDimension, true);
      Matrix<T,U,MatrixStructureSquare,Distribution> tempRI(std::move(storeRI), localDimension, localDimension, globalDimension, globalDimension, true);

      // Serialize into the existing Matrix data structures owned by the user
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixA, tempR, matAstartX, matAendX, matAstartY, matAendY, true);
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixRI, tempRI, matRIstartX, matRIendX, matRIstartY, matRIendY, true);
    }
    return;
  }

  int rank;
  // use MPI_COMM_WORLD for this p2p communication for transpose, but could use a smaller communicator
  MPI_Comm_rank(commWorld, &rank);
  // globalDimension will always be a power of 2, but localDimension won't
  U localShift = (localDimension>>1);
  // move localShift up to the next power of 2
  localShift = util<T,U>::getNextPowerOf2(localShift);
  U globalShift = (globalDimension>>1);
  bool saveSwitch = isInversePath;
  int saveIndexPrev = baseCaseDimList.size();
  if (inverseCutoffGlobalDimension >= globalDimension)
  {
    if (isInversePath == false)
    {
      baseCaseDimList.push_back(localShift);
    }
    isInversePath = true;
  }
  rFactorUpper(
    matrixA, matrixRI, localShift, trueLocalDimension, bcDimension, globalShift, trueGlobalDimension,
    matAstartX, matAstartX+localShift, matAstartY, matAstartY+localShift,
    matRIstartX, matRIstartX+localShift, matRIstartY, matRIstartY+localShift, transposePartner, MM_id, TS_id,
    commWorld, commInfo3D, isInversePath, baseCaseDimList, inverseCutoffGlobalDimension);

  isInversePath = saveSwitch;
  int saveIndexAfter = baseCaseDimList.size();

  // Regardless of whether or not we need to communicate for the transpose, we still need to serialize into a square buffer
  Matrix<T,U,MatrixStructureUpperTriangular,Distribution> packedMatrix(std::vector<T>(), localShift, localShift, globalShift, globalShift);
  // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
  Serializer<T,U,MatrixStructureSquare,MatrixStructureUpperTriangular>::Serialize(matrixRI, packedMatrix,
    matRIstartX, matRIstartX+localShift, matRIstartY, matRIstartY+localShift);
  util<T,U>::transposeSwap(
    packedMatrix, rank, transposePartner, commWorld);

  blasEngineArgumentPackage_trmm<T> trmmArgs;
  trmmArgs.order = blasEngineOrder::AblasColumnMajor;
  trmmArgs.side = blasEngineSide::AblasLeft;
  trmmArgs.uplo = blasEngineUpLo::AblasUpper;
  trmmArgs.diag = blasEngineDiag::AblasNonUnit;
  trmmArgs.transposeA = blasEngineTranspose::AblasTrans;
  trmmArgs.alpha = 1.;

  if (isInversePath)
  {
    MM3D<T,U,blasEngine>::Multiply(
      packedMatrix, matrixA, 0, localShift, 0, localShift, matAstartX+localShift, matAendX, matAstartY, matAstartY+localShift,
      commWorld, commInfo3D, trmmArgs, false, true, MM_id);
  }
  else
  {
/*
    blasEngineArgumentPackage_gemm<T> trsmArgs;
    trsmArgs.order = blasEngineOrder::AblasColumnMajor;
    trsmArgs.transposeA = blasEngineTranspose::AblasTrans;
    trsmArgs.transposeB = blasEngineTranspose::AblasNoTrans;

    // create a new subvector
    U len = saveIndexAfter - saveIndexPrev;
    std::vector<U> subBaseCaseDimList(len);
    for (U i=saveIndexPrev; i<saveIndexAfter; i++)
    {
      subBaseCaseDimList[i-saveIndexPrev] = baseCaseDimList[i];
    }
    // make extra copy to avoid corrupting matrixA
    // Future optimization: Copy a part of A into matrixAcopy, to avoid excessing copying
    // Note: some of these globalShifts are wrong, but I don't know any easy way to fix them. Everything might still work though.
    Matrix<T,U,MatrixStructureSquare,Distribution> matrixAcopy(std::vector<T>(), matAendX-(matAstartX+localShift), localShift, globalShift, globalShift);
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixA, matrixAcopy,
      matAstartX+localShift, matAendX, matAstartY, matAstartY+localShift);
    // Also need to serialize top-left quadrant of matrixL so that its size matches packedMatrix
    Matrix<T,U,MatrixStructureSquare,Distribution> packedMatrixR(std::vector<T>(), localShift, localShift, globalShift, globalShift);
    // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixR, packedMatrixR,
      matRstartX, matRstartX+localShift, matRstartY, matRstartY+localShift);
    Matrix<T,U,MatrixStructureSquare,Distribution> matrixRcopy(std::vector<T>(), matRendX-(matRstartX+localShift), localShift, globalShift, globalShift);
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixR, matrixRcopy,
      matRstartX+localShift, matRendX, matRstartY, matRstartY+localShift);
    // Swap, same as we did with inverse
    util<T,U>::transposeSwap(
      packedMatrixR, rank, transposePartner, commWorld);
    TRSM3D<T,U,blasEngine>::iSolveLowerRight(
      packedMatrixR, packedMatrix, matrixRcopy, matrixAcopy,
      subBaseCaseDimList, trsmArgs, commWorld, commInfo3D, MM_id, TS_id);

    // Inject back into matrixR
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixR, matrixRcopy,
      matRstartX+localShift, matRendX, matRstartY, matRstartY+localShift, true);
*/
  }

  int pGridDimensionSize;
  MPI_Comm_size(std::get<0>(commInfo3D), &pGridDimensionSize);
  U reverseDimLocal = localDimension-localShift;
  U reverseDimGlobal = reverseDimLocal*pGridDimensionSize;

  Matrix<T,U,MatrixStructureSquare,Distribution> squareR(std::vector<T>(), reverseDimLocal, localShift, reverseDimGlobal, globalShift);
  // NOTE: WE BROKE SQUARE SEMANTICS WITH THIS. CHANGE LATER!
  Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixA, squareR,
    matAstartX+localShift, matAendX, matAstartY, matAstartY+localShift);
  Matrix<T,U,MatrixStructureSquare,Distribution> squareRSwap = squareR;

  util<T,U>::transposeSwap(
    squareRSwap, rank, transposePartner, commWorld);

  blasEngineArgumentPackage_gemm<T> blasArgs;
  blasArgs.order = blasEngineOrder::AblasColumnMajor;
  blasArgs.transposeA = blasEngineTranspose::AblasTrans;
  blasArgs.transposeB = blasEngineTranspose::AblasNoTrans;
  blasArgs.alpha = -1;
  blasArgs.beta = 1;
  MM3D<T,U,blasEngine>::Multiply(
    squareRSwap, squareR, matrixA, 0, reverseDimLocal, 0, localShift, 0, reverseDimLocal, 0, localShift,
    matAstartX+localShift, matAendX, matAstartY+localShift, matAendY, commWorld, commInfo3D, blasArgs, false, false, true, MM_id);

  // Only need to change the argument for matrixA
  saveSwitch = isInversePath;
  if (inverseCutoffGlobalDimension >= globalDimension)
  {
    if (isInversePath == false)
    {
      baseCaseDimList.push_back(localShift);
    }
    isInversePath = true;
  }
  rFactorUpper(
    matrixA, matrixRI, reverseDimLocal, trueLocalDimension, bcDimension, reverseDimGlobal/*globalShift*/, trueGlobalDimension,
    matAstartX+localShift, matAendX, matAstartY+localShift, matAendY,
    matRIstartX+localShift, matRIendX, matRIstartY+localShift, matRIendY, transposePartner, MM_id, TS_id,
    commWorld, commInfo3D, isInversePath, baseCaseDimList, inverseCutoffGlobalDimension);

  isInversePath = saveSwitch;

  // Next step : temp <- R_{12}*RI_{22}
  // We can re-use holdRsyrk as our temporary output matrix.

  if (isInversePath)
  {
    Matrix<T,U,MatrixStructureSquare,Distribution>& tempInverse = squareR;

    blasEngineArgumentPackage_trmm<T> invPackage1;
    invPackage1.order = blasEngineOrder::AblasColumnMajor;
    invPackage1.side = blasEngineSide::AblasRight;
    invPackage1.uplo = blasEngineUpLo::AblasUpper;
    invPackage1.diag = blasEngineDiag::AblasNonUnit;
    invPackage1.transposeA = blasEngineTranspose::AblasNoTrans;
    invPackage1.alpha = 1.;
    MM3D<T,U,blasEngine>::Multiply(
      matrixRI, tempInverse, matRIstartX+localShift, matRIendX, matRIstartY+localShift, matRIendY, 0, reverseDimLocal, 0, localShift, commWorld, commInfo3D, invPackage1, true, false, MM_id);

    // Next step: finish the Triangular inverse calculation
    invPackage1.alpha = -1.;
    invPackage1.side = blasEngineSide::AblasLeft;
    MM3D<T,U,blasEngine>::Multiply(
      matrixRI, tempInverse, matRIstartX, matRIstartX+localShift, matRIstartY, matRIstartY+localShift, 0, reverseDimLocal, 0, localShift,
      commWorld, commInfo3D, invPackage1, true, false, MM_id);
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixRI, tempInverse,
      matRIstartX+localShift, matRIendX, matRIstartY, matRIstartY+localShift, true);
  }
  TAU_FSTOP(CFR3D::rFactorUpper);
}


template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
std::vector<T> CFR3D<T,U,blasEngine>::blockedToCyclicTransformation(
									Matrix<T,U,MatrixStructureSquare,Distribution>& matA,
									U localDimension,
									U globalDimension,
									U bcDimension,
									U matAstartX,
									U matAendX,
									U matAstartY,
									U matAendY,
									int pGridDimensionSize,
									MPI_Comm slice2Dcomm,
                  char dir
								     )
{
  TAU_FSTART(CFR3D::blockedToCyclicTransformation);
  if (dir == 'U')
  {
    Matrix<T,U,MatrixStructureUpperTriangular,Distribution> baseCaseMatrixA(std::vector<T>(), localDimension, localDimension,
      globalDimension, globalDimension);
    Serializer<T,U,MatrixStructureSquare,MatrixStructureUpperTriangular>::Serialize(matA, baseCaseMatrixA, matAstartX,
      matAendX, matAstartY, matAendY);
  //  U aggregDim = localDimension*pGridDimensionSize;
  //  std::vector<T> blockedBaseCaseData(aggregDim*aggregDim);
    U aggregSize = baseCaseMatrixA.getNumElems()*pGridDimensionSize*pGridDimensionSize;
    std::vector<T> blockedBaseCaseData(aggregSize);
    // Note: recv buffer will be larger tha send buffer * pGridDimensionSize**2! This should not crash, but we need this much memory anyway when calling DPOTRF and DTRTRI
    MPI_Allgather(baseCaseMatrixA.getRawData(), baseCaseMatrixA.getNumElems(), MPI_DOUBLE,
      &blockedBaseCaseData[0], baseCaseMatrixA.getNumElems(), MPI_DOUBLE, slice2Dcomm);

    TAU_FSTOP(CFR3D::blockedToCyclicTransformation);
    return util<T,U>::blockedToCyclicSpecial(
      blockedBaseCaseData, localDimension, localDimension, pGridDimensionSize, dir);
  }
  else
  { // dir == 'L'
    Matrix<T,U,MatrixStructureLowerTriangular,Distribution> baseCaseMatrixA(std::vector<T>(), localDimension, localDimension,
      globalDimension, globalDimension);
    Serializer<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular>::Serialize(matA, baseCaseMatrixA, matAstartX,
      matAendX, matAstartY, matAendY);
  //  U aggregDim = localDimension*pGridDimensionSize;
  //  std::vector<T> blockedBaseCaseData(aggregDim*aggregDim);
    U aggregSize = baseCaseMatrixA.getNumElems()*pGridDimensionSize*pGridDimensionSize;
    std::vector<T> blockedBaseCaseData(aggregSize);
    // Note: recv buffer will be larger tha send buffer * pGridDimensionSize**2! This should not crash, but we need this much memory anyway when calling DPOTRF and DTRTRI
    MPI_Allgather(baseCaseMatrixA.getRawData(), baseCaseMatrixA.getNumElems(), MPI_DOUBLE,
      &blockedBaseCaseData[0], baseCaseMatrixA.getNumElems(), MPI_DOUBLE, slice2Dcomm);

    TAU_FSTOP(CFR3D::blockedToCyclicTransformation);
    return util<T,U>::blockedToCyclicSpecial(
      blockedBaseCaseData, localDimension, localDimension, pGridDimensionSize, dir);
  }
}


// This method can be called from Lower and Upper with one tweak, but note that currently, we iterate over the entire square,
//   when we are really only writing to a triangle. So there is a source of optimization here at least in terms of
//   number of flops, but in terms of memory accesses and cache lines, not sure. Note that with this optimization,
//   we may need to separate into two different functions
template<typename T, typename U, template<typename, typename> class blasEngine>
void CFR3D<T,U,blasEngine>::cyclicToLocalTransformation(
								std::vector<T>& storeT,
								std::vector<T>& storeTI,
								U localDimension,
								U globalDimension,
								U bcDimension,
								int pGridDimensionSize,
								int rankSlice,
								char dir
							     )
{
  TAU_FSTART(CFR3D::cyclicToLocalTransformation);
  U writeIndex = 0;
  U rowOffsetWithinBlock = rankSlice / pGridDimensionSize;
  U columnOffsetWithinBlock = rankSlice % pGridDimensionSize;
  U numCyclicBlocksPerRowCol = localDimension/*bcDimension/pGridDimensionSize*/;
  // modify bcDimension
  bcDimension = localDimension*pGridDimensionSize;
  // MACRO loop over all cyclic "blocks"
  for (U i=0; i<numCyclicBlocksPerRowCol; i++)
  {
    // We know which row corresponds to our processor in each cyclic "block"
    // Inner loop over all cyclic "blocks" partitioning up the columns
    // Future improvement: only need to iterate over lower triangular.
    for (U j=0; j<numCyclicBlocksPerRowCol; j++)
    {
      // We know which column corresponds to our processor in each cyclic "block"
      // Future improvement: get rid of the inner if statement and separate out this inner loop into 2 loops
      // Further improvement: use only triangular matrices and then Serialize into a square later?
      U readIndexCol = i*pGridDimensionSize + columnOffsetWithinBlock;
      U readIndexRow = j*pGridDimensionSize + rowOffsetWithinBlock;
      if (((dir == 'L') && (readIndexCol <= readIndexRow)) ||  ((dir == 'U') && (readIndexCol >= readIndexRow)))
      {
        storeT[writeIndex] = storeT[readIndexCol*bcDimension + readIndexRow];
        storeTI[writeIndex] = storeTI[readIndexCol*bcDimension + readIndexRow];
      }
      else
      {
//        break;
//        if (storeT[writeIndex] != 0) std::cout << "val - " << storeT[writeIndex] << std::endl;
        storeT[writeIndex] = 0.;
        storeTI[writeIndex] = 0.;
      }
      writeIndex++;
    }
  }
  TAU_FSTOP(CFR3D::cyclicToLocalTransformation);
}
