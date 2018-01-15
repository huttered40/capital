/* Author: Edward Hutter */


template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CFR3D<T,U,blasEngine>::Factor(
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixT,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixTI,
  char dir,
  int tune,
  MPI_Comm commWorld )
{
  // Need to split up the commWorld communicator into a 3D grid similar to Summa3D
  int rank,size;
  MPI_Comm_rank(commWorld, &rank);
  MPI_Comm_size(commWorld, &size);

  int pGridDimensionSize = std::nearbyint(std::pow(size,1./3.));
  int helper = pGridDimensionSize;
  helper *= helper;
  int pGridCoordX = rank%pGridDimensionSize;
  int pGridCoordY = (rank%helper)/pGridDimensionSize;
  int pGridCoordZ = rank/helper;
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
/*
  if (rank == 0)
  {
    std::cout << "localDimension - " << localDimension << std::endl;
    std::cout << "globalDimension - " << globalDimension << std::endl;
    std::cout << "bcDimension - " << bcDimension << std::endl;
  }
*/

  if (dir == 'L')
  {
    rFactorLower(matrixA, matrixT, matrixTI, localDimension, localDimension, bcDimension, globalDimension, globalDimension,
      0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, transposePartner, commWorld);
  }
  else if (dir == 'U')
  {
    rFactorUpper(matrixA, matrixT, matrixTI, localDimension, localDimension, bcDimension, globalDimension, globalDimension,
      0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, transposePartner, commWorld);
  }
}

template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CFR3D<T,U,blasEngine>::rFactorLower(
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
  MPI_Comm commWorld )	// We want to pass in commWorld as MPI_COMM_WORLD because we want to pass that into 3D MM
{
  if (globalDimension <= bcDimension)
  {

    int tempRank; MPI_Comm_rank(MPI_COMM_WORLD, &tempRank);
//    if (tempRank == 0) std::cout << "base case with localDimension - " << localDimension << ", globalDimension - " << globalDimension << " and bcDimension - " << bcDimension << std::endl;

    // First: AllGather matrix A so that every processor has the same replicated diagonal square partition of matrix A of dimension bcDimension
    //          Note that processors only want to communicate with those on their same 2D slice, since the matrices are replicated on every slice
    //          Note that before the AllGather, we need to serialize the matrix A into the small square matrix
    // Second: Data will be received in a blocked order due to AllGather semantics, which is not what we want. We need to get back to cyclic again
    //           This is an ugly process, as it was in the last code.
    // Third: Once data is in cyclic format, we call call sequential Cholesky Factorization and Triangular Inverse.
    // Fourth: Save the data that each processor owns according to the cyclic rule.

    int rankWorld, rankSlice;
    int sizeWorld, sizeSlice;
    MPI_Comm slice2D;
    MPI_Comm_rank(commWorld, &rankWorld);
    MPI_Comm_size(commWorld, &sizeWorld);

    U pGridDimensionSize = ceil(pow(sizeWorld,1./3.));
    U helper = pGridDimensionSize;
    helper *= helper;
    int pGridCoordZ = rankWorld/helper;

    // Attain the communicator with only processors on the same 2D slice
    MPI_Comm_split(commWorld, pGridCoordZ, rankWorld, &slice2D);
    MPI_Comm_rank(slice2D, &rankSlice);
    MPI_Comm_size(slice2D, &sizeSlice);

    // Should be fast pass-by-value via move semantics
    std::vector<T> cyclicBaseCaseData = blockedToCyclicTransformation(matrixA, localDimension, globalDimension, globalDimension/*bcDimension*/, matAstartX, matAendX,
      matAstartY, matAendY, pGridDimensionSize, slice2D);

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
/*
      if (tempRank == 0)
      {
        std::cout << "check local A values\n";
        for (int i=0; i<deepBaseCase.size(); i++)
        {
          std::cout << deepBaseCase[i] << " ";
        }
        std::cout << "\n";
      }
*/
      // Until then, assume a double datatype and simply use LAPACKE_dpotrf. Worry about adding more capabilities later.
      LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'L', finalDim/*bcDimension*/, &deepBaseCase[0], finalDim/*bcDimension*/);
/*
      if (tempRank == 0)
      {
        std::cout << "check local CF values\n";
        for (int i=0; i<deepBaseCase.size(); i++)
        {
          std::cout << deepBaseCase[i] << " ";
        }
        std::cout << "\n";
      }
*/
      // Now, we have L_{11} located inside the "square" vector cyclicBaseCaseData.
      //   We need to call the "move builder" constructor in order to "move" this "rawData" into its own matrix.
      //   Only then, can we call Serializer into the real matrixL. WRONG! We need to find the data we own according to the cyclic rule first!
      //    So it doesn't make any sense to "move" these into Matrices yet.
      // Finally, we need that data for calling the triangular inverse.

      // Next: sequential triangular inverse. Question: does DTRTRI require packed storage or square storage? I think square, so that it can use BLAS-3.
      std::vector<T> deepBaseCaseInv = deepBaseCase;		// true copy because we have to, unless we want to iterate (see below) two different times
      LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'L', 'N', finalDim/*bcDimension*/, &deepBaseCaseInv[0], finalDim/*bcDimension*/);
/*
      if (tempRank == 0)
      {
        std::cout << "check local TI values\n";
        for (int i=0; i<deepBaseCaseInv.size(); i++)
        {
          std::cout << deepBaseCaseInv[i] << " ";
        }
        std::cout << "\n";
      }
*/
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

      cyclicToLocalTransformation(deepBaseCaseFill, deepBaseCaseInvFill, localDimension, globalDimension, globalDimension/*bcDimension*/, pGridDimensionSize, rankSlice, 'L');
/*
      if (tempRank == 0)
      {
        std::cout << "check MY TI values\n";
        for (int i=0; i<deepBaseCaseInvFill.size(); i++)
        {
          std::cout << deepBaseCaseInvFill[i] << " ";
        }
        std::cout << "\n";
      }
*/
      // "Inject" the first part of these vectors into Matrices (Square Structure is the only option for now)
      //   This is a bit sneaky, since the vector we "move" into the Matrix has a larger size than the Matrix knows, but with the right member
      //    variables, this should be ok.

      Matrix<T,U,MatrixStructureSquare,Distribution> tempL(std::move(deepBaseCaseFill), localDimension, localDimension, globalDimension, globalDimension, true);
      Matrix<T,U,MatrixStructureSquare,Distribution> tempLI(std::move(deepBaseCaseInvFill), localDimension, localDimension, globalDimension, globalDimension, true);

      // Serialize into the existing Matrix data structures owned by the user
//      if (tempRank == 0) { std::cout << "check these 4 numbers - " << matLstartX << "," << matLendX << "," << matLstartY << "," << matLendY << std::endl;}
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixL, tempL, matLstartX, matLendX, matLstartY, matLendY, true);
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

      cyclicToLocalTransformation(storeL, storeLI, localDimension, globalDimension, globalDimension/*bcDimension*/, pGridDimensionSize, rankSlice, 'L');

      // "Inject" the first part of these vectors into Matrices (Square Structure is the only option for now)
      //   This is a bit sneaky, since the vector we "move" into the Matrix has a larger size than the Matrix knows, but with the right member
      //    variables, this should be ok.

      Matrix<T,U,MatrixStructureSquare,Distribution> tempL(std::move(storeL), localDimension, localDimension, globalDimension, globalDimension, true);
      Matrix<T,U,MatrixStructureSquare,Distribution> tempLI(std::move(storeLI), localDimension, localDimension, globalDimension, globalDimension, true);

      // Serialize into the existing Matrix data structures owned by the user
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixL, tempL, matLstartX, matLendX, matLstartY, matLendY, true);
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixLI, tempLI, matLIstartX, matLIendX, matLIstartY, matLIendY, true);
    }
    MPI_Comm_free(&slice2D);
    return;
  }

  int rank;
  // use MPI_COMM_WORLD for this p2p communication for transpose, but could use a smaller communicator
  MPI_Comm_rank(commWorld, &rank);
  // globalDimension will always be a power of 2, but localDimension won't
  U localShift = (localDimension>>1);
  // We need localShift to be a power of 2
  if ((localShift & (localShift-1)) != 0)
  {
    // move localShift up to the next power of 2
    localShift--;
    localShift |= (localShift >> 1);
    localShift |= (localShift >> 2);
    localShift |= (localShift >> 4);
    localShift |= (localShift >> 8);
    localShift |= (localShift >> 16);
    // corner case: if dealing with 64-bit integers, shift the 32
    localShift |= (localShift >> 32);
    localShift++;
  }
//  std::cout << "localDimension - " << localDimension << " LOCALSHIFT - " << localShift << std::endl;

  U globalShift = (globalDimension>>1);
  rFactorLower(matrixA, matrixL, matrixLI, localShift, trueLocalDimension, bcDimension, globalShift, trueGlobalDimension,
    matAstartX, matAstartX+localShift, matAstartY, matAstartY+localShift,
    matLstartX, matLstartX+localShift, matLstartY, matLstartY+localShift,
    matLIstartX, matLIstartX+localShift, matLIstartY, matLIstartY+localShift, transposePartner, commWorld);

  // Regardless of whether or not we need to communicate for the transpose, we still need to serialize into a square buffer
  Matrix<T,U,MatrixStructureLowerTriangular,Distribution> packedMatrix(std::vector<T>(), localShift, localShift, globalShift, globalShift);
  // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
//  std::cout << "yo, localShift - " << localShift << ", matLIstartX - " << matLIstartX << ", matLIstartX+localShift - " << matLIstartX+localShift << ", matLIstartY - " << matLIstartY << ", matLIstartY+localShift - " << matLIstartY+localShift << "\n";
  Serializer<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular>::Serialize(matrixLI, packedMatrix,
    matLIstartX, matLIstartX+localShift, matLIstartY, matLIstartY+localShift);

  transposeSwap(packedMatrix, rank, transposePartner, commWorld);

  blasEngineArgumentPackage_gemm<T> blasArgs;
  blasArgs.order = blasEngineOrder::AblasColumnMajor;
  blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
  blasArgs.transposeB = blasEngineTranspose::AblasTrans;
  blasArgs.alpha = 1.;
  blasArgs.beta = 0.;
  MM3D<T,U,blasEngine>::Multiply(matrixA, packedMatrix, matrixL, matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY,
      0, localShift, 0, localShift, matLstartX, matLstartX+localShift, matLstartY+localShift, matLendY, commWorld, blasArgs, 0, true, false, true);
/*
  blasEngineArgumentPackage_trmm<T> blasArgs;
  blasArgs.order = blasEngineOrder::AblasColumnMajor;
  blasArgs.side = blasEngineSide::AblasRight;
  blasArgs.uplo = blasEngineUpLo::AblasLower;
  blasArgs.diag = blasEngineDiag::AblasNonUnit;
  blasArgs.transposeA = blasEngineTranspose::AblasTrans;
  blasArgs.alpha = 1.;
  Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixA, matrixL,
    matLstartX, matLstartX+localShift, matLstartY, matLstartY+localShift);
  MM3D<T,U,blasEngine>::Multiply(packedMatrix, matrixL, 0, localShift, 0, localShift, matLstartX, matLstartX+localShift, matLstartY+localShift, matLendY,
      commWorld, blasArgs, 0, false, true);
*/

  // Now we need to perform L_{21}L_{21}^T via syrk
  //   Actually, I am havin trouble with SYRK, lets try gemm instead
  // As of January 2017, still having trouble with SYRK.

  // Later optimization: avoid this recalculation at each recursive level, since it will always be the same.
  int sizeWorld;
  MPI_Comm_size(commWorld, &sizeWorld);
  U pGridDimensionSize = ceil(pow(sizeWorld,1./3.));
  U reverseDimLocal = localDimension-localShift;
  U reverseDimGlobal = reverseDimLocal*pGridDimensionSize;
//  if (rank == 0) std::cout << "reverseDimGlobal - " << reverseDimLocal << " and global - " << reverseDimGlobal << std::endl;
  Matrix<T,U,MatrixStructureSquare,Distribution> holdLsyrk(std::vector<T>(reverseDimLocal*reverseDimLocal), reverseDimLocal, reverseDimLocal, reverseDimGlobal, reverseDimGlobal, true);

  Matrix<T,U,MatrixStructureSquare,Distribution> squareL(std::vector<T>(), localShift, reverseDimLocal, globalShift, reverseDimGlobal);
  // NOTE: WE BROKE SQUARE SEMANTICS WITH THIS. CHANGE LATER!
  Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixL, squareL,
    matLstartX, matLstartX+localShift, matLstartY+localShift, matLendY);
  Matrix<T,U,MatrixStructureSquare,Distribution> squareLSwap = squareL;

  transposeSwap(squareLSwap, rank, transposePartner, commWorld);
/*
  blasEngineArgumentPackage_gemm<T> blasArgsGemm;
  blasArgsGemm.order = blasEngineOrder::AblasColumnMajor;
  blasArgsGemm.transposeA = blasEngineTranspose::AblasNoTrans;
  blasArgsGemm.transposeB = blasEngineTranspose::AblasTrans;
  blasArgsGemm.alpha = 1.;
  blasArgsGemm.beta = 0.;
*/

//  std::cout << "rank " << rank << " has localShift - " << localShift << " and reverseDimLocal - " << reverseDimLocal << std::endl;
  MM3D<T,U,blasEngine>::Multiply(squareL, squareLSwap, holdLsyrk, 0, localShift, 0, reverseDimLocal, 0, localShift, 0, reverseDimLocal,
      0, reverseDimLocal, 0, reverseDimLocal, commWorld, blasArgs, 0, false, false, false);

  // Next step: A_{22} - holdLsyrk.
  Matrix<T,U,MatrixStructureSquare,Distribution> holdSum(std::vector<T>(reverseDimLocal*reverseDimLocal), reverseDimLocal, reverseDimLocal, reverseDimGlobal, reverseDimGlobal, true);
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixAquadrant4(std::vector<T>(), reverseDimLocal, reverseDimLocal, reverseDimGlobal, reverseDimGlobal);
  Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixA, matrixAquadrant4, matAstartX+localShift,
    matAendX, matAstartY+localShift, matAendY);
  
  std::vector<T>& syrkVec = holdLsyrk.getVectorData();
  std::vector<T>& matAVec = matrixAquadrant4.getVectorData();
  std::vector<T>& holdVec = holdSum.getVectorData();
  for (U i=0; i<reverseDimLocal; i++)
  {
    for (U j=0; j<reverseDimLocal; j++)
    {
      U index = i*(reverseDimLocal)+j;
      holdVec[index] = matAVec[index] - syrkVec[index];
    }
  }

  // Only need to change the argument for matrixA
  rFactorLower(holdSum, matrixL, matrixLI, reverseDimLocal, trueLocalDimension, bcDimension, reverseDimGlobal/*globalShift*/, trueGlobalDimension,
    0, reverseDimLocal, 0, reverseDimLocal, matLstartX+localShift, matLendX, matLstartY+localShift, matLendY,
    matLIstartX+localShift, matLIendX, matLIstartY+localShift, matLIendY, transposePartner, commWorld);

  // Next step : temp <- L_{21}*LI_{11}
  // We can re-use holdLsyrk as our temporary output matrix.

  Matrix<T,U,MatrixStructureSquare,Distribution>& tempInverse = squareL/*holdLsyrk*/;
  
  blasEngineArgumentPackage_gemm<T> invPackage1;
  invPackage1.order = blasEngineOrder::AblasColumnMajor;
  invPackage1.transposeA = blasEngineTranspose::AblasNoTrans;
  invPackage1.transposeB = blasEngineTranspose::AblasNoTrans;
  invPackage1.alpha = 1.;
  invPackage1.beta = 0.;
  MM3D<T,U,blasEngine>::Multiply(matrixL, matrixLI,
    tempInverse, matLstartX, matLstartX+localShift, matLstartY+localShift, matLendY, matLIstartX, matLIstartX+localShift, matLIstartY,
      matLIstartY+localShift, 0, localShift, 0, reverseDimLocal, commWorld, invPackage1, 0, true, true, false);

  // Next step: finish the Triangular inverse calculation
  invPackage1.alpha = -1.;
  MM3D<T,U,blasEngine>::Multiply(matrixLI, tempInverse,
    matrixLI, matLstartX+localShift, matLendX, matLstartY+localShift, matLendY, 0, localShift, 0, reverseDimLocal,
      matLIstartX, matLIstartX+localShift, matLIstartY+localShift, matLIendY, commWorld, invPackage1, 0, true, false, true);
}


template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CFR3D<T,U,blasEngine>::rFactorUpper(
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
                       MPI_Comm commWorld
                     )
{
  if (globalDimension <= bcDimension)
  {
    // First: AllGather matrix A so that every processor has the same replicated diagonal square partition of matrix A of dimension bcDimension
    //          Note that processors only want to communicate with those on their same 2D slice, since the matrices are replicated on every slice
    //          Note that before the AllGather, we need to serialize the matrix A into the small square matrix
    // Second: Data will be received in a blocked order due to AllGather semantics, which is not what we want. We need to get back to cyclic again
    //           This is an ugly process, as it was in the last code.
    // Third: Once data is in cyclic format, we call call sequential Cholesky Factorization and Triangular Inverse.
    // Fourth: Save the data that each processor owns according to the cyclic rule.

    int rankWorld, rankSlice;
    int sizeWorld, sizeSlice;
    MPI_Comm slice2D;
    MPI_Comm_rank(commWorld, &rankWorld);
    MPI_Comm_size(commWorld, &sizeWorld);

    U pGridDimensionSize = ceil(pow(sizeWorld,1./3.));
    U helper = pGridDimensionSize;
    helper *= helper;
    int pGridCoordZ = rankWorld/helper;

    // Attain the communicator with only processors on the same 2D slice
    MPI_Comm_split(commWorld, pGridCoordZ, rankWorld, &slice2D);
    MPI_Comm_rank(slice2D, &rankSlice);
    MPI_Comm_size(slice2D, &sizeSlice);

    // Should be fast pass-by-value via move semantics
    std::vector<T> cyclicBaseCaseData = blockedToCyclicTransformation(matrixA, localDimension, globalDimension, globalDimension/*bcDimension*/, matAstartX, matAendX,
      matAstartY, matAendY, pGridDimensionSize, slice2D);

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
/*
      if (tempRank == 0)
      {
        std::cout << "check local A values\n";
        for (int i=0; i<deepBaseCase.size(); i++)
        {
          std::cout << deepBaseCase[i] << " ";
        }
        std::cout << "\n";
      }
*/
      // Until then, assume a double datatype and simply use LAPACKE_dpotrf. Worry about adding more capabilities later.
      LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', finalDim/*bcDimension*/, &deepBaseCase[0], finalDim/*bcDimension*/);
/*
      if (tempRank == 0)
      {
        std::cout << "check local CF values\n";
        for (int i=0; i<deepBaseCase.size(); i++)
        {
          std::cout << deepBaseCase[i] << " ";
        }
        std::cout << "\n";
      }
*/
      // Now, we have L_{11} located inside the "square" vector cyclicBaseCaseData.
      //   We need to call the "move builder" constructor in order to "move" this "rawData" into its own matrix.
      //   Only then, can we call Serializer into the real matrixL. WRONG! We need to find the data we own according to the cyclic rule first!
      //    So it doesn't make any sense to "move" these into Matrices yet.
      // Finally, we need that data for calling the triangular inverse.

      // Next: sequential triangular inverse. Question: does DTRTRI require packed storage or square storage? I think square, so that it can use BLAS-3.
      std::vector<T> deepBaseCaseInv = deepBaseCase;		// true copy because we have to, unless we want to iterate (see below) two different times
      LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', finalDim/*bcDimension*/, &deepBaseCaseInv[0], finalDim/*bcDimension*/);
/*
      if (tempRank == 0)
      {
        std::cout << "check local TI values\n";
        for (int i=0; i<deepBaseCaseInv.size(); i++)
        {
          std::cout << deepBaseCaseInv[i] << " ";
        }
        std::cout << "\n";
      }
*/
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

      cyclicToLocalTransformation(deepBaseCaseFill, deepBaseCaseInvFill, localDimension, globalDimension, globalDimension/*bcDimension*/, pGridDimensionSize, rankSlice, 'U');
/*
      if (tempRank == 0)
      {
        std::cout << "check MY TI values\n";
        for (int i=0; i<deepBaseCaseInvFill.size(); i++)
        {
          std::cout << deepBaseCaseInvFill[i] << " ";
        }
        std::cout << "\n";
      }
*/
      // "Inject" the first part of these vectors into Matrices (Square Structure is the only option for now)
      //   This is a bit sneaky, since the vector we "move" into the Matrix has a larger size than the Matrix knows, but with the right member
      //    variables, this should be ok.

      Matrix<T,U,MatrixStructureSquare,Distribution> tempR(std::move(deepBaseCaseFill), localDimension, localDimension, globalDimension, globalDimension, true);
      Matrix<T,U,MatrixStructureSquare,Distribution> tempRI(std::move(deepBaseCaseInvFill), localDimension, localDimension, globalDimension, globalDimension, true);

      // Serialize into the existing Matrix data structures owned by the user
//      if (tempRank == 0) { std::cout << "check these 4 numbers - " << matRstartX << "," << matRendX << "," << matRstartY << "," << matRendY << std::endl;}
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixR, tempR, matRstartX, matRendX, matRstartY, matRendY, true);
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

      cyclicToLocalTransformation(storeR, storeRI, localDimension, globalDimension, globalDimension/*bcDimension*/, pGridDimensionSize, rankSlice, 'U');

      // "Inject" the first part of these vectors into Matrices (Square Structure is the only option for now)
      //   This is a bit sneaky, since the vector we "move" into the Matrix has a larger size than the Matrix knows, but with the right member
      //    variables, this should be ok.

      Matrix<T,U,MatrixStructureSquare,Distribution> tempR(std::move(storeR), localDimension, localDimension, globalDimension, globalDimension, true);
      Matrix<T,U,MatrixStructureSquare,Distribution> tempRI(std::move(storeRI), localDimension, localDimension, globalDimension, globalDimension, true);

      // Serialize into the existing Matrix data structures owned by the user
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixR, tempR, matRstartX, matRendX, matRstartY, matRendY, true);
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixRI, tempRI, matRIstartX, matRIendX, matRIstartY, matRIendY, true);
    }
    MPI_Comm_free(&slice2D);
    return;
  }

  int rank;
  // use MPI_COMM_WORLD for this p2p communication for transpose, but could use a smaller communicator
  MPI_Comm_rank(commWorld, &rank);
  // globalDimension will always be a power of 2, but localDimension won't
  U localShift = (localDimension>>1);
  // We need localShift to be a power of 2
  if ((localShift & (localShift-1)) != 0)
  {
    // move localShift up to the next power of 2
    localShift--;
    localShift |= (localShift >> 1);
    localShift |= (localShift >> 2);
    localShift |= (localShift >> 4);
    localShift |= (localShift >> 8);
    localShift |= (localShift >> 16);
    // corner case: if dealing with 64-bit integers, shift the 32
    localShift |= (localShift >> 32);
    localShift++;
  }
//  std::cout << "localDimension - " << localDimension << " LOCALSHIFT - " << localShift << std::endl;

  U globalShift = (globalDimension>>1);

  rFactorUpper(matrixA, matrixR, matrixRI, localShift, trueLocalDimension, bcDimension, globalShift, trueGlobalDimension,
    matAstartX, matAstartX+localShift, matAstartY, matAstartY+localShift,
    matRstartX, matRstartX+localShift, matRstartY, matRstartY+localShift,
    matRIstartX, matRIstartX+localShift, matRIstartY, matRIstartY+localShift, transposePartner, commWorld);

  // Regardless of whether or not we need to communicate for the transpose, we still need to serialize into a square buffer
  Matrix<T,U,MatrixStructureUpperTriangular,Distribution> packedMatrix(std::vector<T>(), localShift, localShift, globalShift, globalShift);
  // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
  Serializer<T,U,MatrixStructureSquare,MatrixStructureUpperTriangular>::Serialize(matrixRI, packedMatrix,
    matRIstartX, matRIstartX+localShift, matRIstartY, matRIstartY+localShift);

  transposeSwap(packedMatrix, rank, transposePartner, commWorld);
 
  blasEngineArgumentPackage_gemm<T> blasArgs;
  blasArgs.order = blasEngineOrder::AblasColumnMajor;
  blasArgs.transposeA = blasEngineTranspose::AblasTrans;
  blasArgs.transposeB = blasEngineTranspose::AblasNoTrans;
  blasArgs.alpha = 1.;
  blasArgs.beta = 0.;
  MM3D<T,U,blasEngine>::Multiply(packedMatrix, matrixA, matrixR, 0, localShift, 0, localShift, matAstartX+localShift, matAendX, matAstartY, matAstartY+localShift,
      matRstartX+localShift, matRendX, matRstartY, matRstartY+localShift, commWorld, blasArgs, 0, false, true, true);

  int sizeWorld;
  MPI_Comm_size(commWorld, &sizeWorld);
  U pGridDimensionSize = ceil(pow(sizeWorld,1./3.));
  U reverseDimLocal = localDimension-localShift;
  U reverseDimGlobal = reverseDimLocal*pGridDimensionSize;

  // Now we need to perform R_{12}^T * R_{12} via syrk
  //   Actually, I am havin trouble with SYRK, lets try gemm instead
  Matrix<T,U,MatrixStructureSquare,Distribution> holdRsyrk(std::vector<T>(reverseDimLocal*reverseDimLocal), reverseDimLocal, reverseDimLocal, reverseDimGlobal, reverseDimGlobal, true);

  Matrix<T,U,MatrixStructureSquare,Distribution> squareR(std::vector<T>(), reverseDimLocal, localShift, reverseDimGlobal, globalShift);
  // NOTE: WE BROKE SQUARE SEMANTICS WITH THIS. CHANGE LATER!
  Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixR, squareR,
    matRstartX+localShift, matRendX, matRstartY, matRstartY+localShift);
  Matrix<T,U,MatrixStructureSquare,Distribution> squareRSwap = squareR;

  transposeSwap(squareRSwap, rank, transposePartner, commWorld);

  MM3D<T,U,blasEngine>::Multiply(squareRSwap, squareR, holdRsyrk, 0, reverseDimLocal, 0, localShift, 0, reverseDimLocal, 0, localShift,
      0, reverseDimLocal, 0, reverseDimLocal, commWorld, blasArgs, 0, false, false, false);

  // Next step: A_{22} - holdRsyrk.
  Matrix<T,U,MatrixStructureSquare,Distribution> holdSum(std::vector<T>(reverseDimLocal*reverseDimLocal), reverseDimLocal, reverseDimLocal, reverseDimGlobal, reverseDimGlobal, true);
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixAquadrant4(std::vector<T>(), reverseDimLocal, reverseDimLocal, reverseDimGlobal, reverseDimGlobal);
  Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixA, matrixAquadrant4, matAstartX+localShift,
    matAendX, matAstartY+localShift, matAendY);

  std::vector<T>& syrkVec = holdRsyrk.getVectorData();
  std::vector<T>& matAVec = matrixAquadrant4.getVectorData();
  std::vector<T>& holdVec = holdSum.getVectorData();
  for (U i=0; i<reverseDimLocal; i++)
  {
    for (U j=0; j<reverseDimLocal; j++)
    {
      U index = i*reverseDimLocal+j;
      holdVec[index] = matAVec[index] - syrkVec[index];
    }
  }

  // Only need to change the argument for matrixA
  rFactorUpper(holdSum, matrixR, matrixRI, reverseDimLocal, trueLocalDimension, bcDimension, reverseDimGlobal/*globalShift*/, trueGlobalDimension,
    0, reverseDimLocal, 0, reverseDimLocal, matRstartX+localShift, matRendX, matRstartY+localShift, matRendY,
    matRIstartX+localShift, matRIendX, matRIstartY+localShift, matRIendY, transposePartner, commWorld);

  // Next step : temp <- R_{12}*RI_{22}
  // We can re-use holdRsyrk as our temporary output matrix.

  Matrix<T,U,MatrixStructureSquare,Distribution>& tempInverse = squareR/*holdRsyrk*/;
  
  blasEngineArgumentPackage_gemm<T> invPackage1;
  invPackage1.order = blasEngineOrder::AblasColumnMajor;
  invPackage1.transposeA = blasEngineTranspose::AblasNoTrans;
  invPackage1.transposeB = blasEngineTranspose::AblasNoTrans;
  invPackage1.alpha = 1.;
  invPackage1.beta = 0.;
  MM3D<T,U,blasEngine>::Multiply(matrixR, matrixRI,tempInverse, matRstartX+localShift, matRendX, matRstartY, matRstartY+localShift,
    matRIstartX+localShift, matRIendX, matRIstartY+localShift, matRIendY, 0, reverseDimLocal, 0, localShift, commWorld, invPackage1, 0, true, true, false);

  // Next step: finish the Triangular inverse calculation
  invPackage1.alpha = -1.;
  MM3D<T,U,blasEngine>::Multiply(matrixRI, tempInverse,
    matrixRI, matRstartX, matRstartX+localShift, matRstartY, matRstartY+localShift, 0, reverseDimLocal, 0, localShift,
      matRIstartX+localShift, matRIendX, matRIstartY, matRIstartY+localShift, commWorld, invPackage1, 0, true, false, true);
}


template<typename T, typename U, template<typename, typename> class blasEngine>
template<
  template<typename,typename,template<typename,typename,int> class> class StructureArg,
  template<typename,typename,int> class Distribution>
void CFR3D<T,U,blasEngine>::transposeSwap(
											Matrix<T,U,StructureArg,Distribution>& mat,
											int myRank,
											int transposeRank,
											MPI_Comm commWorld
										     )
{
  if (myRank != transposeRank)
  {
    // Transfer with transpose rank
    MPI_Sendrecv_replace(mat.getRawData(), mat.getNumElems(), MPI_DOUBLE, transposeRank, 0, transposeRank, 0, commWorld, MPI_STATUS_IGNORE);

    // Note: the received data that now resides in mat is NOT transposed, and the Matrix structure is LowerTriangular
    //       This necesitates making the "else" processor serialize its data L11^{-1} from a square to a LowerTriangular,
    //       since we need to make sure that we call a MM::multiply routine with the same Structure, or else segfault.

  }
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
									MPI_Comm slice2Dcomm
								     )
{
  Matrix<T,U,MatrixStructureSquare,Distribution> baseCaseMatrixA(std::vector<T>(), localDimension, localDimension,
    globalDimension, globalDimension);
  Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matA, baseCaseMatrixA, matAstartX,
    matAendX, matAstartY, matAendY);

  U aggregDim = localDimension*pGridDimensionSize;
  std::vector<T> blockedBaseCaseData(aggregDim*aggregDim/*bcDimension*bcDimension*/);
  std::vector<T> cyclicBaseCaseData(aggregDim*aggregDim/*bcDimension*bcDimension*/);
  MPI_Allgather(baseCaseMatrixA.getRawData(), baseCaseMatrixA.getNumElems(), MPI_DOUBLE,
    &blockedBaseCaseData[0], baseCaseMatrixA.getNumElems(), MPI_DOUBLE, slice2Dcomm);

  // Right now, we assume matrixA has Square Structure, if we want to let the user pass in just the unique part via a Triangular Structure,
  //   then we will need to change this.
  //   Note: this operation is just not cache efficient due to hopping around blockedBaseCaseData. Locality is not what we would like,
  //     but not sure it can really be improved here. Something to look into later.
  //   Also: Although (for LAPACKE_dpotrf), we need to allocate a square buffer and fill in (only) the lower (or upper) triangular portion,
  //     in the future, we might want to try different storage patterns from that one paper by Gustavsson

  // Strategy: We will write to cyclicBaseCaseData in proper order BUT will have to hop around blockedBaseCaseData. This should be ok since
  //   reading on modern computer architectures is less expensive via cache misses than writing, and we should not have any compulsory cache misses

  U numCyclicBlocksPerRowCol = localDimension/*bcDimension/pGridDimensionSize*/;
  U writeIndex = 0;
  U recvDataOffset = localDimension*localDimension;
  // MACRO loop over all cyclic "blocks" (dimensionX direction)
  for (U i=0; i<numCyclicBlocksPerRowCol; i++)
  {
    // Inner loop over all columns in a cyclic "block"
    for (U j=0; j<pGridDimensionSize; j++)
    {
      // Inner loop over all cyclic "blocks"
      for (U k=0; k<numCyclicBlocksPerRowCol; k++)
      {
        // Inner loop over all elements along columns
        for (U z=0; z<pGridDimensionSize; z++)
        {
          U readIndex = i*numCyclicBlocksPerRowCol + j*recvDataOffset + k + z*pGridDimensionSize*recvDataOffset;
          cyclicBaseCaseData[writeIndex++] = blockedBaseCaseData[readIndex];
        }
      }
    }
  }

  // Should be quick pass-by-value via move semantics, since we are effectively returning a localvariable that is going to lose its scope anyways,
  //   so the compiler should be smart enough to use the move constructor for the vector in the caller function.
  return cyclicBaseCaseData;
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
        storeT[writeIndex] = 0.;
        storeTI[writeIndex] = 0.;
      }
      writeIndex++;
    }
  }
}
