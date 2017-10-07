/* Author: Edward Hutter */


template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CFR3D<T,U,MatrixStructureSquare,MatrixStructureSquare,blasEngine>::Factor(
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixL,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixLI,
  U dimension,								// Assume this dimension is of each local Matrix that each processor owns. Could change that later.
  MPI_Comm commWorld )
{
  // Need to split up the commWorld communicator into a 3D grid similar to Summa3D
  int rank,size;
  MPI_Comm_rank(commWorld, &rank);
  MPI_Comm_size(commWorld, &size);

  int pGridDimensionSize = ceil(pow(size,1./3.));
  int helper = pGridDimensionSize;
  helper *= helper;
  int pGridCoordX = rank%pGridDimensionSize;
  int pGridCoordY = (rank%helper)/pGridDimensionSize;
  int pGridCoordZ = rank/helper;
  int transposePartner = pGridCoordZ*helper + pGridCoordX*pGridDimensionSize + pGridCoordY;

  U globalDimension = dimension*pGridDimensionSize;
  U bcDimension = globalDimension/(helper*helper);		// Can be tuned later.

  rFactor(matrixA, matrixL, matrixLI, dimension, bcDimension, globalDimension,
    0, dimension, 0, dimension, 0, dimension, 0, dimension, 0, dimension, 0, dimension, transposePartner, commWorld);
}

template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CFR3D<T,U,MatrixStructureSquare,MatrixStructureSquare,blasEngine>::rFactor(
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixL,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixLI,
  U localDimension,
  U bcDimension,
  U globalDimension,
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
  if (globalDimension == bcDimension)
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
    std::vector<T> cyclicBaseCaseData = blockedToCyclicTransformation(matrixA, localDimension, globalDimension, bcDimension, matAstartX, matAendX,
      matAstartY, matAendY, pGridDimensionSize, slice2D);

    // Now, I want to use something similar to a template class for libraries conforming to the standards of LAPACK, such as FLAME.
    //   I want to be able to mix and match.

    std::vector<T>& storeL = cyclicBaseCaseData;

    // Until then, assume a double datatype and simply use LAPACKE_dpotrf. Worry about adding more capabilities later.
    LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', bcDimension, &storeL[0], bcDimension);

    // Now, we have L_{11} located inside the "square" vector cyclicBaseCaseData.
    //   We need to call the "move builder" constructor in order to "move" this "rawData" into its own matrix.
    //   Only then, can we call Serializer into the real matrixL. WRONG! We need to find the data we own according to the cyclic rule first!
    //    So it doesn't make any sense to "move" these into Matrices yet.
    // Finally, we need that data for calling the triangular inverse.

    // Next: sequential triangular inverse. Question: does DTRTRI require packed storage or square storage? I think square, so that it can use BLAS-3.
    std::vector<T> storeLI = storeL;		// true copy because we have to, unless we want to iterate (see below) two different times
    LAPACKE_dtrtri(LAPACK_ROW_MAJOR, 'L', 'N', bcDimension, &storeLI[0], bcDimension);

    // Only truly a "square-to-square" serialization because we store matrixL as a square (no packed storage yet!)

    // Now, before we can serialize into matrixL and matrixLI, we need to save the values that this processor owns according to the cyclic rule.
    // Only then can we serialize.

    // Iterate and pick out. I would like not to have to create any more memory and I would only like to iterate once, not twice, for storeL and storeLI
    //   Use the "overwrite" trick that I have used in CASI code, as well as other places

    // I am going to use a sneaky trick: I will take the vectorData from storeL and storeLI by reference, overwrite its values,
    //   and then "move" them cheaply into new Matrix structures before I call Serialize on them individually.

    U writeIndex = 0;
    U rowOffsetWithinBlock = rankSlice / pGridDimensionSize;
    U columnOffsetWithinBlock = rankSlice % pGridDimensionSize;
    U numCyclicBlocksPerRowCol = bcDimension/pGridDimensionSize;
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
        U readIndexCol = j*pGridDimensionSize + columnOffsetWithinBlock;
        U readIndexRow = i*pGridDimensionSize + rowOffsetWithinBlock;
        if (readIndexCol <= readIndexRow)
        {
          storeL[writeIndex] = storeL[readIndexCol + readIndexRow*bcDimension];
          storeLI[writeIndex] = storeLI[readIndexCol + readIndexRow*bcDimension];
        }
        else
        {
          storeL[writeIndex] = 0.;
          storeLI[writeIndex] = 0.;
        }
        writeIndex++;
      }
    }

    // "Inject" the first part of these vectors into Matrices (Square Structure is the only option for now)
    //   This is a bit sneaky, since the vector we "move" into the Matrix has a larger size than the Matrix knows, but with the right member
    //    variables, this should be ok.

    Matrix<T,U,MatrixStructureSquare,Distribution> tempL(std::move(storeL), localDimension, localDimension, globalDimension, globalDimension, true);
    Matrix<T,U,MatrixStructureSquare,Distribution> tempLI(std::move(storeLI), localDimension, localDimension, globalDimension, globalDimension, true);

    // Serialize into the existing Matrix data structures owned by the user
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixL, tempL, matLstartX, matLendX, matLstartY, matLendY, true);
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixLI, tempLI, matLIstartX, matLIendX, matLIstartY, matLIendY, true);

    return;
  }

  int rank;
  // use MPI_COMM_WORLD for this p2p communication for transpose, but could use a smaller communicator
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  U localShift = (localDimension>>1);
  U globalShift = (globalDimension>>1);
  rFactor(matrixA, matrixL, matrixLI, localShift, bcDimension, globalShift,
    matAstartX, matAstartX+localShift, matAstartY, matAstartY+localShift,
    matLstartX, matLstartX+localShift, matLstartY, matLstartY+localShift,
    matLIstartX, matLIstartX+localShift, matLIstartY, matLIstartY+localShift, transposePartner, commWorld);

  // Regardless of whether or not we need to communicate for the transpose, we still need to serialize into a square buffer
  Matrix<T,U,MatrixStructureLowerTriangular,Distribution> packedMatrix(std::vector<T>(), localShift, localShift, globalShift, globalShift);
  // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
  Serializer<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular>::Serialize(matrixLI, packedMatrix,
    matLIstartX, matLIstartX+localShift, matLIstartY, matLIstartY+localShift);

  transposeSwap(packedMatrix, rank, transposePartner);
 
  blasEngineArgumentPackage_gemm<T> blasArgs;
  blasArgs.order = blasEngineOrder::AblasRowMajor;
  blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
  blasArgs.transposeB = blasEngineTranspose::AblasTrans;
  blasArgs.alpha = 1.;
  blasArgs.beta = 0.;
  SquareMM3D<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular,MatrixStructureSquare,blasEngine>::
    Multiply(matrixA, packedMatrix, matrixL, matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY,
      0, localShift, 0, localShift, matLstartX, matLstartX+localShift, matLstartY+localShift, matLendY, commWorld, blasArgs, true, false, true);


  // Now we need to perform L_{21}L_{21}^T via syrk
  //   Actually, I am havin trouble with SYRK, lets try gemm instead
  Matrix<T,U,MatrixStructureSquare,Distribution> holdLsyrk(std::vector<T>(localShift*localShift), localShift, localShift, globalShift, globalShift, true);

  Matrix<T,U,MatrixStructureSquare,Distribution> squareL(std::vector<T>(), localShift, localShift, globalShift, globalShift);
  Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixL, squareL,
    matLstartX, matLstartX+localShift, matLstartY+localShift, matLendY);
  Matrix<T,U,MatrixStructureSquare,Distribution> squareLSwap = squareL;

  transposeSwap(squareLSwap, rank, transposePartner);

  SquareMM3D<T,U,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare,blasEngine>::
    Multiply(squareL, squareLSwap, holdLsyrk, 0, localShift, 0, localShift, 0, localShift, 0, localShift,
      0, localShift, 0, localShift, commWorld, blasArgs, false, false, false);

  // Next step: A_{22} - holdLsyrk.
  Matrix<T,U,MatrixStructureSquare,Distribution> holdSum(std::vector<T>(localShift*localShift), localShift, localShift, globalShift, globalShift, true);
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixAquadrant4(std::vector<T>(), localShift, localShift, globalShift, globalShift);
  Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixA, matrixAquadrant4, matAstartX+localShift,
    matAendX, matAstartY+localShift, matAendY);
  
  std::vector<T>& syrkVec = holdLsyrk.getVectorData();
  std::vector<T>& matAVec = matrixAquadrant4.getVectorData();
  std::vector<T>& holdVec = holdSum.getVectorData();
  for (U i=0; i<localShift; i++)
  {
    for (U j=0; j<localShift; j++)
    {
      U index = i*localShift+j;
      holdVec[index] = matAVec[index] - syrkVec[index];
    }
  }

  // Only need to change the argument for matrixA
  rFactor(holdSum, matrixL, matrixLI, localShift, bcDimension, globalShift,
    0, localShift, 0, localShift, matLstartX+localShift, matLendX, matLstartY+localShift, matLendY,
    matLIstartX+localShift, matLIendX, matLIstartY+localShift, matLIendY, transposePartner, commWorld);

  // Next step : temp <- L_{21}*LI_{11}
  // We can re-use holdLsyrk as our temporary output matrix.

  Matrix<T,U,MatrixStructureSquare,Distribution>& tempInverse = holdLsyrk;
  
  blasEngineArgumentPackage_gemm<T> invPackage1;
  invPackage1.order = blasEngineOrder::AblasRowMajor;
  invPackage1.transposeA = blasEngineTranspose::AblasNoTrans;
  invPackage1.transposeB = blasEngineTranspose::AblasNoTrans;
  invPackage1.alpha = 1.;
  invPackage1.beta = 0.;
  SquareMM3D<T,U,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare,blasEngine>::Multiply(matrixL, matrixLI,
    tempInverse, matLstartX, matLstartX+localShift, matLstartY+localShift, matLendY, matLIstartX, matLIstartX+localShift, matLIstartY,
      matLIstartY+localShift, 0, localShift, 0, localShift, commWorld, invPackage1, true, true, false);

  // Next step: finish the Triangular inverse calculation
  invPackage1.alpha = -1.;
  SquareMM3D<T,U,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare,blasEngine>::Multiply(matrixLI, tempInverse,
    matrixLI, matLstartX+localShift, matLendX, matLstartY+localShift, matLendY, 0, localShift, 0, localShift,
      matLIstartX, matLIstartX+localShift, matLIstartY+localShift, matLIendY, commWorld, invPackage1, true, true, true);
  
}
  

template<typename T, typename U, template<typename, typename> class blasEngine>
template<
  template<typename,typename,template<typename,typename,int> class> class StructureArg,
  template<typename,typename,int> class Distribution>
void CFR3D<T,U,MatrixStructureSquare,MatrixStructureSquare,blasEngine>::transposeSwap(
											Matrix<T,U,StructureArg,Distribution>& mat,
											int myRank,
											int transposeRank
										     )
{
  if (myRank != transposeRank)
  {
    // Transfer with transpose rank
    MPI_Sendrecv_replace(mat.getRawData(), mat.getNumElems(), MPI_DOUBLE, transposeRank, 0, transposeRank, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Note: the received data that now resides in mat is NOT transposed, and the Matrix structure is LowerTriangular
    //       This necesitates making the "else" processor serialize its data L11^{-1} from a square to a LowerTriangular,
    //       since we need to make sure that we call a MM::multiply routine with the same Structure, or else segfault.

  }
}


template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
std::vector<T> CFR3D<T,U,MatrixStructureSquare,MatrixStructureSquare,blasEngine>::blockedToCyclicTransformation(
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

  std::vector<T> blockedBaseCaseData(bcDimension*bcDimension);
  std::vector<T>cyclicBaseCaseData(bcDimension*bcDimension);
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

  U numCyclicBlocksPerRowCol = bcDimension/pGridDimensionSize;
  U writeIndex = 0;
  U recvDataOffset = localDimension*localDimension;
  // MACRO loop over all cyclic "blocks"
  for (U i=0; i<numCyclicBlocksPerRowCol; i++)
  {
    // Inner loop over all rows in a cyclic "block"
    for (U j=0; j<pGridDimensionSize; j++)
    {
      // Inner loop over all cyclic "blocks" partitioning up the columns
      for (U k=0; k<numCyclicBlocksPerRowCol; k++)
      {
        // Inner loop over all elements within a row of a cyclic "block"
        for (U z=0; z<pGridDimensionSize; z++)
        {
          U readIndex = i*numCyclicBlocksPerRowCol + j*recvDataOffset*pGridDimensionSize + k + z*recvDataOffset;
          cyclicBaseCaseData[writeIndex++] = blockedBaseCaseData[readIndex];
        }
      }
    }
  }

  // Should be quick pass-by-value via move semantics, since we are effectively returning a localvariable that is going to lose its scope anyways,
  //   so the compiler should be smart enough to use the move constructor for the vector in the caller function.
  return cyclicBaseCaseData;
}
