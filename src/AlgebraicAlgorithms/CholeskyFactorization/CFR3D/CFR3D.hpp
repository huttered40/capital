/* Author: Edward Hutter */


template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
std::vector<U> CFR3D<T,U,blasEngine>::Factor(
#ifdef TIMER
  pTimer& timer,
#endif
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixT,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixTI,
  U inverseCutOffGlobalDimension,
  char dir,
  int tune,
  MPI_Comm commWorld,
  int MMid,
  int TSid
  )
{
  // Need to split up the commWorld communicator into a 3D grid similar to Summa3D
  int rank,size;
#ifdef TIMER
  size_t index1 = timer.setStartTime("MPI_Comm_rank");
#endif
  MPI_Comm_rank(commWorld, &rank);
#ifdef TIMER
  timer.setEndTime("MPI_Comm_rank", index1);
  size_t index2 = timer.setStartTime("MPI_Comm_size");
#endif
  MPI_Comm_size(commWorld, &size);
#ifdef TIMER
  timer.setEndTime("MPI_Comm_size", index2);
#endif

  int pGridDimensionSize = std::nearbyint(std::pow(size,1./3.));
  int helper = pGridDimensionSize;
  helper *= helper;
  int pGridCoordX = rank%pGridDimensionSize;
  int pGridCoordY = (rank%helper)/pGridDimensionSize;
  int pGridCoordZ = rank/helper;
  int transposePartner = pGridCoordZ*helper + pGridCoordX*pGridDimensionSize + pGridCoordY;

  // Attain the communicator with only processors on the same 2D slice
  MPI_Comm slice2D;
#ifdef TIMER
  size_t index3 = timer.setStartTime("MPI_Comm_split");
#endif
  MPI_Comm_split(commWorld, pGridCoordZ, rank, &slice2D);
#ifdef TIMER
  timer.setEndTime("MPI_Comm_split", index3);
#endif

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
  inverseCutOffGlobalDimension = bcDimension;
  for (int i=0; i<save; i++)
  {
    inverseCutOffGlobalDimension *= 2;
  }

  std::vector<U> baseCaseDimList;

  if (dir == 'L')
  {
    bool isInversePath = (inverseCutOffGlobalDimension >= globalDimension ? true : false);
    if (isInversePath) { baseCaseDimList.push_back(localDimension); }
#ifdef TIMER
    size_t index4 = timer.setStartTime("rFactorLower");
#endif
    rFactorLower(
#ifdef TIMER
      timer,
#endif
      matrixA, matrixT, matrixTI, localDimension, localDimension, bcDimension, globalDimension, globalDimension,
      0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, transposePartner, MMid, TSid, slice2D, commWorld,
        isInversePath, baseCaseDimList, inverseCutOffGlobalDimension);
#ifdef TIMER
    timer.setEndTime("rFactorLower", index4);
#endif
  }
  else if (dir == 'U')
  {
    bool isInversePath = (inverseCutOffGlobalDimension >= globalDimension ? true : false);
    if (isInversePath) { baseCaseDimList.push_back(localDimension); }
#ifdef TIMER
    size_t index4 = timer.setStartTime("rFactorUpper");
#endif
    rFactorUpper(
#ifdef TIMER
      timer,
#endif
      matrixA, matrixT, matrixTI, localDimension, localDimension, bcDimension, globalDimension, globalDimension,
      0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, transposePartner, MMid, TSid, slice2D, commWorld,
        isInversePath, baseCaseDimList, inverseCutOffGlobalDimension);
#ifdef TIMER
    timer.setEndTime("rFactorUpper", index4);
#endif
  }

  return baseCaseDimList;
}

template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CFR3D<T,U,blasEngine>::rFactorLower(
#ifdef TIMER
  pTimer& timer,
#endif
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixL,
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
  U matLstartX,
  U matLendX,
  U matLstartY,
  U matLendY,
  U matLIstartX,
  U matLIendX,
  U matLIstartY,
  U matLIendY,
  U transposePartner,
  int MM_id,
  int TS_id,
  MPI_Comm slice2D,
  MPI_Comm commWorld, 	// We want to pass in commWorld as MPI_COMM_WORLD because we want to pass that into 3D MM
  bool& isInversePath,
  std::vector<U>& baseCaseDimList,
  U inverseCutoffGlobalDimension)
{
  if (globalDimension <= bcDimension)
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

    int rankSlice,sizeSlice,sizeComm;
#ifdef TIMER
    size_t index1 = timer.setStartTime("MPI_Comm_size");
#endif
    MPI_Comm_size(commWorld, &sizeComm);
#ifdef TIMER
    timer.setEndTime("MPI_Comm_size", index1);
    size_t index2 = timer.setStartTime("MPI_Comm_rank");
#endif
    MPI_Comm_rank(slice2D, &rankSlice);
#ifdef TIMER
    timer.setEndTime("MPI_Comm_rank", index2);
    size_t index3 = timer.setStartTime("MPI_Comm_size");
#endif
    MPI_Comm_size(slice2D, &sizeSlice);
#ifdef TIMER
    timer.setEndTime("MPI_Comm_size", index3);
#endif
    int pGridDimensionSize = std::nearbyint(std::pow(sizeComm,1./3.));

    // Should be fast pass-by-value via move semantics
#ifdef TIMER
    size_t index4 = timer.setStartTime("CFR3D::blockedToCyclicTransformation");
#endif
    std::vector<T> cyclicBaseCaseData = blockedToCyclicTransformation(
#ifdef TIMER
      timer,
#endif
      matrixA, localDimension, globalDimension, globalDimension/*bcDimension*/, matAstartX, matAendX,
      matAstartY, matAendY, pGridDimensionSize, slice2D);
#ifdef TIMER
    timer.setEndTime("CFR3D::blockedToCyclicTransformation", index4);
#endif

    // Now, I want to use something similar to a template class for libraries conforming to the standards of LAPACK, such as FLAME.
    //   I want to be able to mix and match.

    if ((matLendX == trueLocalDimension) && (matLendY == trueLocalDimension))
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
#ifdef TIMER
      size_t index5 = timer.setStartTime("LAPACK_DPOTRF");
#endif
      LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', finalDim/*bcDimension*/, &deepBaseCase[0], finalDim/*bcDimension*/);
#ifdef TIMER
      timer.setEndTime("LAPACK_DPOTRF", index5);
#endif
      // Now, we have L_{11} located inside the "square" vector cyclicBaseCaseData.
      //   We need to call the "move builder" constructor in order to "move" this "rawData" into its own matrix.
      //   Only then, can we call Serializer into the real matrixL. WRONG! We need to find the data we own according to the cyclic rule first!
      //    So it doesn't make any sense to "move" these into Matrices yet.
      // Finally, we need that data for calling the triangular inverse.

      // Next: sequential triangular inverse. Question: does DTRTRI require packed storage or square storage? I think square, so that it can use BLAS-3.
      std::vector<T> deepBaseCaseInv = deepBaseCase;		// true copy because we have to, unless we want to iterate (see below) two different times
#ifdef TIMER
      size_t index6 = timer.setStartTime("LAPACK_DTRTRI");
#endif
      LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'L', 'N', finalDim/*bcDimension*/, &deepBaseCaseInv[0], finalDim/*bcDimension*/);
#ifdef TIMER
      timer.setEndTime("LAPACK_DTRTRI", index6);
#endif
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

#ifdef TIMER
      size_t index7 = timer.setStartTime("CFR3D::cyclicToLocalTransformation");
#endif
      cyclicToLocalTransformation(
#ifdef TIMER
        timer,
#endif
        deepBaseCaseFill, deepBaseCaseInvFill, localDimension, globalDimension, globalDimension/*bcDimension*/, pGridDimensionSize, rankSlice, 'L');
#ifdef TIMER
      timer.setEndTime("CFR3D::cyclicToLocalTransformation", index7);
#endif
      // "Inject" the first part of these vectors into Matrices (Square Structure is the only option for now)
      //   This is a bit sneaky, since the vector we "move" into the Matrix has a larger size than the Matrix knows, but with the right member
      //    variables, this should be ok.

      Matrix<T,U,MatrixStructureSquare,Distribution> tempL(std::move(deepBaseCaseFill), localDimension, localDimension, globalDimension, globalDimension, true);
      Matrix<T,U,MatrixStructureSquare,Distribution> tempLI(std::move(deepBaseCaseInvFill), localDimension, localDimension, globalDimension, globalDimension, true);

      // Serialize into the existing Matrix data structures owned by the user
//      if (tempRank == 0) { std::cout << "check these 4 numbers - " << matLstartX << "," << matLendX << "," << matLstartY << "," << matLendY << std::endl;}
#ifdef TIMER
      size_t index8 = timer.setStartTime("Serializer");
#endif
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixL, tempL, matLstartX, matLendX, matLstartY, matLendY, true);
#ifdef TIMER
      timer.setEndTime("Serializer", index8);
      size_t index9 = timer.setStartTime("Serializer");
#endif
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixLI, tempLI, matLIstartX, matLIendX, matLIstartY, matLIendY, true);
#ifdef TIMER
      timer.setEndTime("Serializer", index9);
#endif
    }
    else
    {
      std::vector<T>& storeL = cyclicBaseCaseData;

      // Until then, assume a double datatype and simply use LAPACKE_dpotrf. Worry about adding more capabilities later.
#ifdef TIMER
      size_t index5 = timer.setStartTime("LAPACK_DPOTRF");
#endif
      LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', localDimension*pGridDimensionSize/*bcDimension*/, &storeL[0], localDimension*pGridDimensionSize/*bcDimension*/);
#ifdef TIMER
      timer.setEndTime("LAPACK_DPOTRF", index5);
#endif

      // Now, we have L_{11} located inside the "square" vector cyclicBaseCaseData.
      //   We need to call the "move builder" constructor in order to "move" this "rawData" into its own matrix.
      //   Only then, can we call Serializer into the real matrixL. WRONG! We need to find the data we own according to the cyclic rule first!
      //    So it doesn't make any sense to "move" these into Matrices yet.
      // Finally, we need that data for calling the triangular inverse.

      // Next: sequential triangular inverse. Question: does DTRTRI require packed storage or square storage? I think square, so that it can use BLAS-3.
      std::vector<T> storeLI = storeL;		// true copy because we have to, unless we want to iterate (see below) two different times
#ifdef TIMER
      size_t index6 = timer.setStartTime("LAPACK_DTRTRI");
#endif
      LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'L', 'N', localDimension*pGridDimensionSize/*bcDimension*/, &storeLI[0], localDimension*pGridDimensionSize/*bcDimension*/);
#ifdef TIMER
      timer.setEndTime("LAPACK_DTRTRI", index6);
#endif

      // Only truly a "square-to-square" serialization because we store matrixL as a square (no packed storage yet!)

      // Now, before we can serialize into matrixL and matrixLI, we need to save the values that this processor owns according to the cyclic rule.
      // Only then can we serialize.

      // Iterate and pick out. I would like not to have to create any more memory and I would only like to iterate once, not twice, for storeL and storeLI
      //   Use the "overwrite" trick that I have used in CASI code, as well as other places

      // I am going to use a sneaky trick: I will take the vectorData from storeL and storeLI by reference, overwrite its values,
      //   and then "move" them cheaply into new Matrix structures before I call Serialize on them individually.

#ifdef TIMER
      size_t index7 = timer.setStartTime("CFR3D::cyclicToLocalTransformation");
#endif
      cyclicToLocalTransformation(
#ifdef TIMER
        timer,
#endif
        storeL, storeLI, localDimension, globalDimension, globalDimension/*bcDimension*/, pGridDimensionSize, rankSlice, 'L');
#ifdef TIMER
      timer.setEndTime("CFR3D::cyclicToLocalTransformation", index7);
#endif

      // "Inject" the first part of these vectors into Matrices (Square Structure is the only option for now)
      //   This is a bit sneaky, since the vector we "move" into the Matrix has a larger size than the Matrix knows, but with the right member
      //    variables, this should be ok.

      Matrix<T,U,MatrixStructureSquare,Distribution> tempL(std::move(storeL), localDimension, localDimension, globalDimension, globalDimension, true);
      Matrix<T,U,MatrixStructureSquare,Distribution> tempLI(std::move(storeLI), localDimension, localDimension, globalDimension, globalDimension, true);

      // Serialize into the existing Matrix data structures owned by the user
#ifdef TIMER
      size_t index8 = timer.setStartTime("Serializer");
#endif
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixL, tempL, matLstartX, matLendX, matLstartY, matLendY, true);
#ifdef TIMER
      timer.setEndTime("Serializer", index8);
      size_t index9 = timer.setStartTime("Serializer");
#endif
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixLI, tempLI, matLIstartX, matLIendX, matLIstartY, matLIendY, true);
#ifdef TIMER
      timer.setEndTime("Serializer", index9);
#endif
    }
    return;
  }

  int rank;
#ifdef TIMER
  size_t index1 = timer.setStartTime("MPI_Comm_rank");
#endif
  MPI_Comm_rank(commWorld, &rank);
#ifdef TIMER
  timer.setEndTime("MPI_Comm_rank", index1);
#endif
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
#ifdef TIMER
    timer,
#endif
    matrixA, matrixL, matrixLI, localShift, trueLocalDimension, bcDimension, globalShift, trueGlobalDimension,
    matAstartX, matAstartX+localShift, matAstartY, matAstartY+localShift,
    matLstartX, matLstartX+localShift, matLstartY, matLstartY+localShift,
    matLIstartX, matLIstartX+localShift, matLIstartY, matLIstartY+localShift, transposePartner, MM_id, TS_id,
    slice2D, commWorld, isInversePath, baseCaseDimList, inverseCutoffGlobalDimension);

  isInversePath = saveSwitch;
  int saveIndexAfter = baseCaseDimList.size();

/* Note: the code below might actualy be a bit better than the uncommented-out code below it, because this code takes advantage of the triangular
         structure of packedMatrix, allowing it to communicate half the words in the transpose. Only problem: because the Allgather+MM3D routine
         doesn't currently work for non-square structure matrices, it crashes if I make this triangular.
  // Regardless of whether or not we need to communicate for the transpose, we still need to serialize into a buffer
  Matrix<T,U,MatrixStructureLowerTriangular,Distribution> packedMatrix(std::vector<T>(), localShift, localShift, globalShift, globalShift);
  // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
  Serializer<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular>::Serialize(matrixLI, packedMatrix,
    matLIstartX, matLIstartX+localShift, matLIstartY, matLIstartY+localShift);
*/
  Matrix<T,U,MatrixStructureSquare,Distribution> packedMatrix(std::vector<T>(), localShift, localShift, globalShift, globalShift);
  // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
#ifdef TIMER
  size_t index3 = timer.setStartTime("Serializer");
#endif
  Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixLI, packedMatrix,
    matLIstartX, matLIstartX+localShift, matLIstartY, matLIstartY+localShift);
#ifdef TIMER
  timer.setEndTime("Serializer", index3);
#endif

#ifdef TIMER
  size_t index4 = timer.setStartTime("transposeSwap");
#endif
  util<T,U>::transposeSwap(
#ifdef TIMER
    timer,
#endif
    packedMatrix, rank, transposePartner, commWorld);
#ifdef TIMER
  timer.setEndTime("transposeSwap", index4);
#endif

  blasEngineArgumentPackage_gemm<T> blasArgs;
  blasArgs.order = blasEngineOrder::AblasColumnMajor;
  blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
  blasArgs.transposeB = blasEngineTranspose::AblasTrans;
  blasArgs.alpha = 1.;
  blasArgs.beta = 0.;

  if (isInversePath)
  {
#ifdef TIMER
    size_t index5 = timer.setStartTime("MM3D::MultiplyCut");
#endif
    MM3D<T,U,blasEngine>::Multiply(
#ifdef TIMER
      timer,
#endif
      matrixA, packedMatrix, matrixL, matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY,
      0, localShift, 0, localShift, matLstartX, matLstartX+localShift, matLstartY+localShift, matLendY, commWorld, blasArgs, true, false, true, MM_id);
#ifdef TIMER
    timer.setEndTime("MM3D::MultiplyCut", index5);
#endif
  }
  else
  {
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
    // make extra copy to avoid corrupting matrixA
    // Future optimization: Copy a part of A into matrixAcopy, to avoid excessing copying
    // Note: some of these globalShifts are wrong, but I don't know any easy way to fix them. Everything might still work though.
    Matrix<T,U,MatrixStructureSquare,Distribution> matrixAcopy(std::vector<T>(), localShift, matAendY-(matAstartY+localShift), globalShift, globalShift);
#ifdef TIMER
    size_t index6 = timer.setStartTime("Serializer");
#endif
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixA, matrixAcopy,
      matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY);
#ifdef TIMER
    timer.setEndTime("Serializer", index6);
#endif
    // Also need to serialize top-left quadrant of matrixL so that its size matches packedMatrix
    Matrix<T,U,MatrixStructureSquare,Distribution> packedMatrixL(std::vector<T>(), localShift, localShift, globalShift, globalShift);
    // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
#ifdef TIMER
    size_t index7 = timer.setStartTime("Serializer");
#endif
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixL, packedMatrixL,
      matLstartX, matLstartX+localShift, matLstartY, matLstartY+localShift);
#ifdef TIMER
    timer.setEndTime("Serializer", index7);
#endif
    Matrix<T,U,MatrixStructureSquare,Distribution> matrixLcopy(std::vector<T>(), localShift, matLendY-(matLstartY+localShift), globalShift, globalShift);
#ifdef TIMER
    size_t index8 = timer.setStartTime("Serializer");
#endif
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixL, matrixLcopy,
      matLstartX, matLstartX+localShift, matLstartY+localShift, matLendY);
#ifdef TIMER
    timer.setEndTime("Serializer", index8);
#endif
    // Swap, same as we did with inverse
#ifdef TIMER
    size_t index9 = timer.setStartTime("transposeSwap");
#endif
    util<T,U>::transposeSwap(
#ifdef TIMER
      timer,
#endif
      packedMatrixL, rank, transposePartner, commWorld);
#ifdef TIMER
    timer.setEndTime("transposeSwap", index9);
#endif
#ifdef TIMER
    size_t index10 = timer.setStartTime("TRSM::iSolveUpperLeft");
#endif
    TRSM3D<T,U,blasEngine>::iSolveUpperLeft(
#ifdef TIMER
      timer,
#endif
      matrixLcopy,packedMatrixL, packedMatrix, matrixAcopy,
      subBaseCaseDimList, trsmArgs, commWorld, MM_id, TS_id);
#ifdef TIMER
    timer.setEndTime("TRSM::iSolveUpperLeft", index10);
#endif

    // inject matrixLcopy back into matrixL
#ifdef TIMER
    size_t index11 = timer.setStartTime("Serializer");
#endif
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixL, matrixLcopy,
      matLstartX, matLstartX+localShift, matLstartY+localShift, matLendY, true);
#ifdef TIMER
    timer.setEndTime("Serializer", index11);
#endif
  }

  // Now we need to perform L_{21}L_{21}^T via syrk
  //   Actually, I am havin trouble with SYRK, lets try gemm instead
  // As of January 2017, still having trouble with SYRK.

  // Later optimization: avoid this recalculation at each recursive level, since it will always be the same.
  int sizeWorld;
  MPI_Comm_size(commWorld, &sizeWorld);
  U pGridDimensionSize = ceil(pow(sizeWorld,1./3.));
  U reverseDimLocal = localDimension-localShift;
  U reverseDimGlobal = reverseDimLocal*pGridDimensionSize;


//  Matrix<T,U,MatrixStructureSquare,Distribution> holdLsyrk(std::vector<T>(reverseDimLocal*reverseDimLocal), reverseDimLocal, reverseDimLocal, reverseDimGlobal, reverseDimGlobal, true);
  Matrix<T,U,MatrixStructureSquare,Distribution> holdLsyrk(std::vector<T>(), reverseDimLocal, reverseDimLocal, reverseDimGlobal, reverseDimGlobal);
#ifdef TIMER
  size_t index6 = timer.setStartTime("Serializer");
#endif
  Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixA, holdLsyrk, matAstartX+localShift,
    matAendX, matAstartY+localShift, matAendY);
#ifdef TIMER
  timer.setEndTime("Serializer", index6);
#endif

  Matrix<T,U,MatrixStructureSquare,Distribution> squareL(std::vector<T>(), localShift, reverseDimLocal, globalShift, reverseDimGlobal);
  // NOTE: WE BROKE SQUARE SEMANTICS WITH THIS. CHANGE LATER!
#ifdef TIMER
  size_t index7 = timer.setStartTime("Serializer");
#endif
  Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixL, squareL,
    matLstartX, matLstartX+localShift, matLstartY+localShift, matLendY);
#ifdef TIMER
  timer.setEndTime("Serializer", index7);
#endif
  Matrix<T,U,MatrixStructureSquare,Distribution> squareLSwap = squareL;

#ifdef TIMER
  size_t index8 = timer.setStartTime("transposeSwap");
#endif
  util<T,U>::transposeSwap(
#ifdef TIMER
    timer,
#endif
    squareLSwap, rank, transposePartner, commWorld);
#ifdef TIMER
  timer.setEndTime("transposeSwap", index8);
#endif

  blasArgs.alpha = -1;
  blasArgs.beta = 1;
#ifdef TIMER
  size_t index9 = timer.setStartTime("MM3D::MultiplyCut");
#endif
  MM3D<T,U,blasEngine>::Multiply(
#ifdef TIMER
      timer,
#endif
    squareL, squareLSwap, holdLsyrk, 0, localShift, 0, reverseDimLocal, 0, localShift, 0, reverseDimLocal,
    0, reverseDimLocal, 0, reverseDimLocal, commWorld, blasArgs, false, false, false, MM_id);
#ifdef TIMER
  timer.setEndTime("MM3D::MultiplyCut", index9);
#endif

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
  rFactorLower(/*holdSum*/
#ifdef TIMER
    timer,
#endif
    holdLsyrk, matrixL, matrixLI, reverseDimLocal, trueLocalDimension, bcDimension, reverseDimGlobal/*globalShift*/, trueGlobalDimension,
    0, reverseDimLocal, 0, reverseDimLocal, matLstartX+localShift, matLendX, matLstartY+localShift, matLendY,
    matLIstartX+localShift, matLIendX, matLIstartY+localShift, matLIendY, transposePartner, MM_id, TS_id,
    slice2D, commWorld, isInversePath, baseCaseDimList, inverseCutoffGlobalDimension);

  isInversePath = saveSwitch;


  if (isInversePath)
  {
    // Next step : temp <- L_{21}*LI_{11}
    // We can re-use holdLsyrk as our temporary output matrix.

    Matrix<T,U,MatrixStructureSquare,Distribution>& tempInverse = squareL/*holdLsyrk*/;

    blasEngineArgumentPackage_gemm<T> invPackage1;
    invPackage1.order = blasEngineOrder::AblasColumnMajor;
    invPackage1.transposeA = blasEngineTranspose::AblasNoTrans;
    invPackage1.transposeB = blasEngineTranspose::AblasNoTrans;
    invPackage1.alpha = 1.;
    invPackage1.beta = 0.;
#ifdef TIMER
    size_t index10 = timer.setStartTime("MM3D::MultiplyCut");
#endif
    MM3D<T,U,blasEngine>::Multiply(
#ifdef TIMER
      timer,
#endif
      matrixL, matrixLI,
      tempInverse, matLstartX, matLstartX+localShift, matLstartY+localShift, matLendY, matLIstartX, matLIstartX+localShift, matLIstartY,
        matLIstartY+localShift, 0, localShift, 0, reverseDimLocal, commWorld, invPackage1, true, true, false, MM_id);
#ifdef TIMER
    timer.setEndTime("MM3D::MultiplyCut", index10);
#endif

    // Next step: finish the Triangular inverse calculation
    invPackage1.alpha = -1.;
#ifdef TIMER
    size_t index11 = timer.setStartTime("MM3D::MultiplyCut");
#endif
    MM3D<T,U,blasEngine>::Multiply(
#ifdef TIMER
      timer,
#endif
      matrixLI, tempInverse,
      matrixLI, matLstartX+localShift, matLendX, matLstartY+localShift, matLendY, 0, localShift, 0, reverseDimLocal,
        matLIstartX, matLIstartX+localShift, matLIstartY+localShift, matLIendY, commWorld, invPackage1, true, false, true, MM_id);
#ifdef TIMER
    timer.setEndTime("MM3D::MultiplyCut", index11);
#endif
  }
}


template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CFR3D<T,U,blasEngine>::rFactorUpper(
#ifdef TIMER
                       pTimer& timer,
#endif
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR,
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
                       U matRstartX,
                       U matRendX,
                       U matRstartY,
                       U matRendY,
                       U matRIstartX,
                       U matRIendX,
                       U matRIstartY,
                       U matRIendY,
                       U transposePartner,
                       int MM_id,
                       int TS_id,
                       MPI_Comm slice2D,
                       MPI_Comm commWorld,
                       bool& isInversePath,
                       std::vector<U>& baseCaseDimList,
                       U inverseCutoffGlobalDimension
                     )
{
  if (globalDimension <= bcDimension)
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

    int rankSlice,sizeSlice,sizeComm;
#ifdef TIMER
    size_t index1 = timer.setStartTime("MPI_Comm_size");
#endif
    MPI_Comm_size(commWorld, &sizeComm);
#ifdef TIMER
    timer.setEndTime("MPI_Comm_size", index1);
    size_t index2 = timer.setStartTime("MPI_Comm_rank");
#endif
    MPI_Comm_rank(slice2D, &rankSlice);
#ifdef TIMER
    timer.setEndTime("MPI_Comm_rank", index2);
    size_t index3 = timer.setStartTime("MPI_Comm_size");
#endif
    MPI_Comm_size(slice2D, &sizeSlice);
#ifdef TIMER
    timer.setEndTime("MPI_Comm_size", index3);
#endif
    int pGridDimensionSize = std::nearbyint(std::pow(sizeComm,1./3.));

    // Should be fast pass-by-value via move semantics
#ifdef TIMER
    size_t index4 = timer.setStartTime("CFR3D::blockedToCyclicTransformation");
#endif
    std::vector<T> cyclicBaseCaseData = blockedToCyclicTransformation(
#ifdef TIMER
      timer,
#endif
      matrixA, localDimension, globalDimension, globalDimension/*bcDimension*/, matAstartX, matAendX,
      matAstartY, matAendY, pGridDimensionSize, slice2D);
#ifdef TIMER
    timer.setEndTime("CFR3D::blockedToCyclicTransformation", index4);
#endif

    // Now, I want to use something similar to a template class for libraries conforming to the standards of LAPACK, such as FLAME.
    //   I want to be able to mix and match.

    if ((matRendX == trueLocalDimension) && (matRendY == trueLocalDimension))
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
#ifdef TIMER
      size_t index5 = timer.setStartTime("LAPACK_DPOTRF");
#endif
      LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', finalDim/*bcDimension*/, &deepBaseCase[0], finalDim/*bcDimension*/);
#ifdef TIMER
      timer.setEndTime("LAPACK_DPOTRF", index5);
#endif
      // Now, we have L_{11} located inside the "square" vector cyclicBaseCaseData.
      //   We need to call the "move builder" constructor in order to "move" this "rawData" into its own matrix.
      //   Only then, can we call Serializer into the real matrixL. WRONG! We need to find the data we own according to the cyclic rule first!
      //    So it doesn't make any sense to "move" these into Matrices yet.
      // Finally, we need that data for calling the triangular inverse.

      // Next: sequential triangular inverse. Question: does DTRTRI require packed storage or square storage? I think square, so that it can use BLAS-3.
      std::vector<T> deepBaseCaseInv = deepBaseCase;		// true copy because we have to, unless we want to iterate (see below) two different times
#ifdef TIMER
      size_t index6 = timer.setStartTime("LAPACK_DTRTRI");
#endif
      LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', finalDim/*bcDimension*/, &deepBaseCaseInv[0], finalDim/*bcDimension*/);
#ifdef TIMER
      timer.setEndTime("LAPACK_DTRTRI", index6);
#endif
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

#ifdef TIMER
      size_t index7 = timer.setStartTime("CFR3D::cyclicToLocalTransformation");
#endif
      cyclicToLocalTransformation(
#ifdef TIMER
        timer,
#endif
        deepBaseCaseFill, deepBaseCaseInvFill, localDimension, globalDimension, globalDimension/*bcDimension*/, pGridDimensionSize, rankSlice, 'U');
#ifdef TIMER
      timer.setEndTime("CFR3D::cyclicToLocalTransformation", index7);
#endif
      // "Inject" the first part of these vectors into Matrices (Square Structure is the only option for now)
      //   This is a bit sneaky, since the vector we "move" into the Matrix has a larger size than the Matrix knows, but with the right member
      //    variables, this should be ok.

      Matrix<T,U,MatrixStructureSquare,Distribution> tempR(std::move(deepBaseCaseFill), localDimension, localDimension, globalDimension, globalDimension, true);
      Matrix<T,U,MatrixStructureSquare,Distribution> tempRI(std::move(deepBaseCaseInvFill), localDimension, localDimension, globalDimension, globalDimension, true);

      // Serialize into the existing Matrix data structures owned by the user
//      if (tempRank == 0) { std::cout << "check these 4 numbers - " << matRstartX << "," << matRendX << "," << matRstartY << "," << matRendY << std::endl;}
#ifdef TIMER
      size_t index8 = timer.setStartTime("Serializer");
#endif
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixR, tempR, matRstartX, matRendX, matRstartY, matRendY, true);
#ifdef TIMER
      timer.setEndTime("Serializer", index8);
      size_t index9 = timer.setStartTime("Serializer");
#endif
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixRI, tempRI, matRIstartX, matRIendX, matRIstartY, matRIendY, true);
#ifdef TIMER
      timer.setEndTime("Serializer", index9);
#endif
    }
    else
    {
      std::vector<T>& storeR = cyclicBaseCaseData;

      // Until then, assume a double datatype and simply use LAPACKE_dpotrf. Worry about adding more capabilities later.
#ifdef TIMER
      size_t index5 = timer.setStartTime("LAPACK_DPOTRF");
#endif
      LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', localDimension*pGridDimensionSize/*bcDimension*/, &storeR[0], localDimension*pGridDimensionSize/*bcDimension*/);
#ifdef TIMER
      timer.setEndTime("LAPACK_DPOTRF", index5);
#endif

      // Now, we have L_{11} located inside the "square" vector cyclicBaseCaseData.
      //   We need to call the "move builder" constructor in order to "move" this "rawData" into its own matrix.
      //   Only then, can we call Serializer into the real matrixL. WRONG! We need to find the data we own according to the cyclic rule first!
      //    So it doesn't make any sense to "move" these into Matrices yet.
      // Finally, we need that data for calling the triangular inverse.

      // Next: sequential triangular inverse. Question: does DTRTRI require packed storage or square storage? I think square, so that it can use BLAS-3.
      std::vector<T> storeRI = storeR;		// true copy because we have to, unless we want to iterate (see below) two different times
#ifdef TIMER
      size_t index6 = timer.setStartTime("LAPACK_DTRTRI");
#endif
      LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', localDimension*pGridDimensionSize/*bcDimension*/, &storeRI[0], localDimension*pGridDimensionSize/*bcDimension*/);
#ifdef TIMER
      timer.setEndTime("LAPACK_DTRTRI", index6);
#endif

      // Only truly a "square-to-square" serialization because we store matrixL as a square (no packed storage yet!)

      // Now, before we can serialize into matrixL and matrixLI, we need to save the values that this processor owns according to the cyclic rule.
      // Only then can we serialize.

      // Iterate and pick out. I would like not to have to create any more memory and I would only like to iterate once, not twice, for storeL and storeLI
      //   Use the "overwrite" trick that I have used in CASI code, as well as other places

      // I am going to use a sneaky trick: I will take the vectorData from storeL and storeLI by reference, overwrite its values,
      //   and then "move" them cheaply into new Matrix structures before I call Serialize on them individually.

#ifdef TIMER
      size_t index7 = timer.setStartTime("CFR3D::cyclicToLocalTransformation");
#endif
      cyclicToLocalTransformation(
#ifdef TIMER
        timer,
#endif
        storeR, storeRI, localDimension, globalDimension, globalDimension/*bcDimension*/, pGridDimensionSize, rankSlice, 'U');
#ifdef TIMER
      timer.setEndTime("CFR3D::cyclicToLocalTransformation", index7);
#endif

      // "Inject" the first part of these vectors into Matrices (Square Structure is the only option for now)
      //   This is a bit sneaky, since the vector we "move" into the Matrix has a larger size than the Matrix knows, but with the right member
      //    variables, this should be ok.

      Matrix<T,U,MatrixStructureSquare,Distribution> tempR(std::move(storeR), localDimension, localDimension, globalDimension, globalDimension, true);
      Matrix<T,U,MatrixStructureSquare,Distribution> tempRI(std::move(storeRI), localDimension, localDimension, globalDimension, globalDimension, true);

      // Serialize into the existing Matrix data structures owned by the user
#ifdef TIMER
      size_t index8 = timer.setStartTime("Serializer");
#endif
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixR, tempR, matRstartX, matRendX, matRstartY, matRendY, true);
#ifdef TIMER
      timer.setEndTime("Serializer", index8);
      size_t index9 = timer.setStartTime("Serializer");
#endif
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixRI, tempRI, matRIstartX, matRIendX, matRIstartY, matRIendY, true);
#ifdef TIMER
      timer.setEndTime("Serializer", index9);
#endif
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
#ifdef TIMER
    timer,
#endif
    matrixA, matrixR, matrixRI, localShift, trueLocalDimension, bcDimension, globalShift, trueGlobalDimension,
    matAstartX, matAstartX+localShift, matAstartY, matAstartY+localShift,
    matRstartX, matRstartX+localShift, matRstartY, matRstartY+localShift,
    matRIstartX, matRIstartX+localShift, matRIstartY, matRIstartY+localShift, transposePartner, MM_id, TS_id,
    slice2D, commWorld, isInversePath, baseCaseDimList, inverseCutoffGlobalDimension);

  isInversePath = saveSwitch;
  int saveIndexAfter = baseCaseDimList.size();



/* Note: the code below might actualy be a bit better than the uncommented-out code below it, because this code takes advantage of the triangular
         structure of packedMatrix, allowing it to communicate half the words in the transpose. Only problem: because the Allgather+MM3D routine
         doesn't currently work for non-square structure matrices, it crashes if I make this triangular.
  // Regardless of whether or not we need to communicate for the transpose, we still need to serialize into a square buffer
  Matrix<T,U,MatrixStructureUpperTriangular,Distribution> packedMatrix(std::vector<T>(), localShift, localShift, globalShift, globalShift);
  // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
  Serializer<T,U,MatrixStructureSquare,MatrixStructureUpperTriangular>::Serialize(matrixRI, packedMatrix,
    matRIstartX, matRIstartX+localShift, matRIstartY, matRIstartY+localShift);
*/

  Matrix<T,U,MatrixStructureSquare,Distribution> packedMatrix(std::vector<T>(), localShift, localShift, globalShift, globalShift);
  // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
#ifdef TIMER
  size_t index1 = timer.setStartTime("Serializer");
#endif
  Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixRI, packedMatrix,
    matRIstartX, matRIstartX+localShift, matRIstartY, matRIstartY+localShift);
#ifdef TIMER
  timer.setEndTime("Serializer", index1);
#endif

#ifdef TIMER
  size_t index2 = timer.setStartTime("transposeSwap");
#endif
  util<T,U>::transposeSwap(
#ifdef TIMER
    timer,
#endif
    packedMatrix, rank, transposePartner, commWorld);
#ifdef TIMER
  timer.setEndTime("transposeSwap", index2);
#endif

  blasEngineArgumentPackage_gemm<T> blasArgs;
  blasArgs.order = blasEngineOrder::AblasColumnMajor;
  blasArgs.transposeA = blasEngineTranspose::AblasTrans;
  blasArgs.transposeB = blasEngineTranspose::AblasNoTrans;
  blasArgs.alpha = 1.;
  blasArgs.beta = 0.;

  if (isInversePath)
  {
#ifdef TIMER
    size_t index3 = timer.setStartTime("MM3D::MultiplyCut");
#endif
    MM3D<T,U,blasEngine>::Multiply(
#ifdef TIMER
      timer,
#endif
      packedMatrix, matrixA, matrixR, 0, localShift, 0, localShift, matAstartX+localShift, matAendX, matAstartY, matAstartY+localShift,
      matRstartX+localShift, matRendX, matRstartY, matRstartY+localShift, commWorld, blasArgs, false, true, true, MM_id);
#ifdef TIMER
    timer.setEndTime("MM3D::MultiplyCut", index3);
#endif
  }
  else
  {
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
#ifdef TIMER
    size_t index3 = timer.setStartTime("Serializer");
#endif
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixA, matrixAcopy,
      matAstartX+localShift, matAendX, matAstartY, matAstartY+localShift);
#ifdef TIMER
    timer.setEndTime("Serializer", index3);
#endif
    // Also need to serialize top-left quadrant of matrixL so that its size matches packedMatrix
    Matrix<T,U,MatrixStructureSquare,Distribution> packedMatrixR(std::vector<T>(), localShift, localShift, globalShift, globalShift);
    // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
#ifdef TIMER
    size_t index4 = timer.setStartTime("Serializer");
#endif
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixR, packedMatrixR,
      matRstartX, matRstartX+localShift, matRstartY, matRstartY+localShift);
#ifdef TIMER
    timer.setEndTime("Serializer", index4);
#endif
    Matrix<T,U,MatrixStructureSquare,Distribution> matrixRcopy(std::vector<T>(), matRendX-(matRstartX+localShift), localShift, globalShift, globalShift);
#ifdef TIMER
    size_t index5 = timer.setStartTime("Serializer");
#endif
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixR, matrixRcopy,
      matRstartX+localShift, matRendX, matRstartY, matRstartY+localShift);
#ifdef TIMER
    timer.setEndTime("Serializer", index5);
#endif
    // Swap, same as we did with inverse
#ifdef TIMER
    size_t index6 = timer.setStartTime("transposeSwap");
#endif
    util<T,U>::transposeSwap(
#ifdef TIMER
      timer,
#endif
      packedMatrixR, rank, transposePartner, commWorld);
#ifdef TIMER
    timer.setEndTime("transposeSwap", index6);
    size_t index7 = timer.setStartTime("TRSM3D::iSolveLowerRight");
#endif
    TRSM3D<T,U,blasEngine>::iSolveLowerRight(
#ifdef TIMER
      timer,
#endif
      packedMatrixR, packedMatrix, matrixRcopy, matrixAcopy,
      subBaseCaseDimList, trsmArgs, commWorld, MM_id, TS_id);
#ifdef TIMER
    timer.setEndTime("TRSM3D::iSolveLowerRight", index7);
#endif

    // Inject back into matrixR
#ifdef TIMER
    size_t index8 = timer.setStartTime("Serializer");
#endif
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixR, matrixRcopy,
      matRstartX+localShift, matRendX, matRstartY, matRstartY+localShift, true);
#ifdef TIMER
    timer.setEndTime("Serializer", index8);
#endif
  }

  int sizeWorld;
  MPI_Comm_size(commWorld, &sizeWorld);
  U pGridDimensionSize = ceil(pow(sizeWorld,1./3.));
  U reverseDimLocal = localDimension-localShift;
  U reverseDimGlobal = reverseDimLocal*pGridDimensionSize;

  // Now we need to perform R_{12}^T * R_{12} via syrk
  //   Actually, I am havin trouble with SYRK, lets try gemm instead
  Matrix<T,U,MatrixStructureSquare,Distribution> holdRsyrk(std::vector<T>(/*reverseDimLocal*reverseDimLocal*/), reverseDimLocal, reverseDimLocal, reverseDimGlobal, reverseDimGlobal, true);
#ifdef TIMER
  size_t index3 = timer.setStartTime("Serializer");
#endif
  Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixA, holdRsyrk, matAstartX+localShift,
    matAendX, matAstartY+localShift, matAendY);
#ifdef TIMER
  timer.setEndTime("Serializer", index3);
#endif

  Matrix<T,U,MatrixStructureSquare,Distribution> squareR(std::vector<T>(), reverseDimLocal, localShift, reverseDimGlobal, globalShift);
  // NOTE: WE BROKE SQUARE SEMANTICS WITH THIS. CHANGE LATER!
#ifdef TIMER
  size_t index4 = timer.setStartTime("Serializer");
#endif
  Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixR, squareR,
    matRstartX+localShift, matRendX, matRstartY, matRstartY+localShift);
#ifdef TIMER
  timer.setEndTime("Serializer", index4);
#endif
  Matrix<T,U,MatrixStructureSquare,Distribution> squareRSwap = squareR;

#ifdef TIMER
  size_t index5 = timer.setStartTime("transposeSwap");
#endif
  util<T,U>::transposeSwap(
#ifdef TIMER
    timer,
#endif
    squareRSwap, rank, transposePartner, commWorld);
#ifdef TIMER
  timer.setEndTime("transposeSwap", index5);
#endif

  blasArgs.alpha = -1;
  blasArgs.beta = 1;
#ifdef TIMER
  size_t index6 = timer.setStartTime("MM3D::MultiplyCut");
#endif
  MM3D<T,U,blasEngine>::Multiply(
#ifdef TIMER
    timer,
#endif
    squareRSwap, squareR, holdRsyrk, 0, reverseDimLocal, 0, localShift, 0, reverseDimLocal, 0, localShift,
    0, reverseDimLocal, 0, reverseDimLocal, commWorld, blasArgs, false, false, false, MM_id);
#ifdef TIMER
  timer.setEndTime("MM3D::MultiplyCut", index6);
#endif

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
  rFactorUpper(/*holdSum*/
#ifdef TIMER
    timer,
#endif
    holdRsyrk, matrixR, matrixRI, reverseDimLocal, trueLocalDimension, bcDimension, reverseDimGlobal/*globalShift*/, trueGlobalDimension,
    0, reverseDimLocal, 0, reverseDimLocal, matRstartX+localShift, matRendX, matRstartY+localShift, matRendY,
    matRIstartX+localShift, matRIendX, matRIstartY+localShift, matRIendY, transposePartner, MM_id, TS_id,
    slice2D, commWorld, isInversePath, baseCaseDimList, inverseCutoffGlobalDimension);

  isInversePath = saveSwitch;

  // Next step : temp <- R_{12}*RI_{22}
  // We can re-use holdRsyrk as our temporary output matrix.

  if (isInversePath)
  {
    Matrix<T,U,MatrixStructureSquare,Distribution>& tempInverse = squareR/*holdRsyrk*/;

    blasEngineArgumentPackage_gemm<T> invPackage1;
    invPackage1.order = blasEngineOrder::AblasColumnMajor;
    invPackage1.transposeA = blasEngineTranspose::AblasNoTrans;
    invPackage1.transposeB = blasEngineTranspose::AblasNoTrans;
    invPackage1.alpha = 1.;
    invPackage1.beta = 0.;
#ifdef TIMER
    size_t index7 = timer.setStartTime("MM3D::MultiplyCut");
#endif
    MM3D<T,U,blasEngine>::Multiply(
#ifdef TIMER
      timer,
#endif
      matrixR, matrixRI,tempInverse, matRstartX+localShift, matRendX, matRstartY, matRstartY+localShift,
      matRIstartX+localShift, matRIendX, matRIstartY+localShift, matRIendY, 0, reverseDimLocal, 0, localShift, commWorld, invPackage1, true, true, false, MM_id);
#ifdef TIMER
    timer.setEndTime("MM3D::MultiplyCut", index7);
#endif

    // Next step: finish the Triangular inverse calculation
    invPackage1.alpha = -1.;
#ifdef TIMER
    size_t index8 = timer.setStartTime("MM3D::MultiplyCut");
#endif
    MM3D<T,U,blasEngine>::Multiply(
#ifdef TIMER
      timer,
#endif
      matrixRI, tempInverse,
      matrixRI, matRstartX, matRstartX+localShift, matRstartY, matRstartY+localShift, 0, reverseDimLocal, 0, localShift,
        matRIstartX+localShift, matRIendX, matRIstartY, matRIstartY+localShift, commWorld, invPackage1, true, false, true, MM_id);
#ifdef TIMER
    timer.setEndTime("MM3D::MultiplyCut", index8);
#endif
  }
}


template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
std::vector<T> CFR3D<T,U,blasEngine>::blockedToCyclicTransformation(
#ifdef TIMER
                  pTimer& timer,
#endif
									Matrix<T,U,MatrixStructureSquare,Distribution>& matA,
									U localDimension,
									U globalDimension,
									U bcDimension,
									U matAstartX,
									U matAendX,
									U matAstartY,
									U matAendY,
									int pGridDimensionSize,
									MPI_Comm slice2Dcomm
								     )
{
  Matrix<T,U,MatrixStructureSquare,Distribution> baseCaseMatrixA(std::vector<T>(), localDimension, localDimension,
    globalDimension, globalDimension);
#ifdef TIMER
  size_t index1 = timer.setStartTime("Serializer");
#endif
  Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matA, baseCaseMatrixA, matAstartX,
    matAendX, matAstartY, matAendY);
#ifdef TIMER
  timer.setEndTime("Serializer", index1);
#endif

  U aggregDim = localDimension*pGridDimensionSize;
  std::vector<T> blockedBaseCaseData(aggregDim*aggregDim);
#ifdef TIMER
  size_t index2 = timer.setStartTime("MPI_Allgather");
#endif
  MPI_Allgather(baseCaseMatrixA.getRawData(), baseCaseMatrixA.getNumElems(), MPI_DOUBLE,
    &blockedBaseCaseData[0], baseCaseMatrixA.getNumElems(), MPI_DOUBLE, slice2Dcomm);
#ifdef TIMER
  timer.setEndTime("MPI_Allgather", index2);
#endif

  return util<T,U>::blockedToCyclic(
#ifdef TIMER
    timer,
#endif
    blockedBaseCaseData, localDimension, localDimension, pGridDimensionSize);
}


// This method can be called from Lower and Upper with one tweak, but note that currently, we iterate over the entire square,
//   when we are really only writing to a triangle. So there is a source of optimization here at least in terms of
//   number of flops, but in terms of memory accesses and cache lines, not sure. Note that with this optimization,
//   we may need to separate into two different functions
template<typename T, typename U, template<typename, typename> class blasEngine>
void CFR3D<T,U,blasEngine>::cyclicToLocalTransformation(
#ifdef TIMER
                pTimer& timer,
#endif
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
        storeT[writeIndex] = 0.;
        storeTI[writeIndex] = 0.;
      }
      writeIndex++;
    }
  }
}
