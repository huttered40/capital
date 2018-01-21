/*
  Author: Edward Hutter
*/

template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void RTI3D<T,U,blasEngine>::Invert(
              Matrix<T,U,MatrixStructureSquare,Distribution>& matrixT,
              Matrix<T,U,MatrixStructureSquare,Distribution>& matrixTI,
              char dir,
              MPI_Comm commWorld
            )
{
  U localDimension = matrixT.getNumRowsLocal();
  U globalDimension = matrixT.getNumRowsGlobal();
  // the division below may have a remainder, but I think integer division will be ok, as long as we change the base case condition to be <= and not just ==

  if (dir == 'L')
  {
    // call InvertLower
    InvertLower(matrixT, matrixTI, localDimension, 0, localDimension, 0, localDimension, 0, commWorld);
  }
  else
  {
    // call InverseUpper
    InvertUpper(matrixT, matrixTI, localDimension, 0, commWorld);
  }
}

template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void RTI3D<T,U,blasEngine>::InvertLower(
                  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixL,
                  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixLI,
                  U localDimension,
                  U startX,
                  U endX,
                  U startY,
                  U endY,
                  int key,
                  MPI_Comm commWorld
                )
{
  // check base cases
  int commSize,commRank;
  MPI_Comm_rank(commWorld, &commRank);
  MPI_Comm_size(commWorld, &commSize);

  if (commSize == 1)
  {
    // Invert (no serialization check needed, as we will never be in this case unless P=1
    LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'L', 'N', localDimension, matrixL.getRawData(), localDimension);
    return;
  }

  if (commSize == 8)
  {
    // special case
    sliceExchangeBase(matrixL, matrixLI, localDimension, startX, endX, startY, endY, commWorld, 'L');
    U localShift = (localDimension>>1);
    Matrix<T,U,MatrixStructureSquare,Distribution> tempInverse(std::vector<T>(localShift*localShift), localShift, localShift, localDimension, localDimension, true);
    blasEngineArgumentPackage_gemm<T> blasArgs;
    blasArgs.order = blasEngineOrder::AblasColumnMajor;
    blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
    blasArgs.transposeB = blasEngineTranspose::AblasNoTrans;
    blasArgs.alpha = 1.;
    blasArgs.beta = 0.;
    MM3D<T,U,blasEngine>::Multiply(matrixL, matrixLI, tempInverse, startX, startX+localShift, startY+localShift, endY, startX, startX+localShift, startY, startY+localShift, 0, localShift, 0, localShift, commWorld, blasArgs, true, true, false, 0);
    blasArgs.alpha = -1.;
    MM3D<T,U,blasEngine>::Multiply(matrixLI, tempInverse, matrixLI, startX+localShift, endX, startY+localShift, endY, 0, localShift, 0, localShift, startX, startX+localShift, startY+localShift, endY, commWorld, blasArgs, true, false, true, 0);
    return;
  }

  MPI_Comm sliceComm;
  U localShift = (localDimension>>1);

  if (commSize%2 == 1)
  {
    // Special case. Not implemented yet.
    MPI_Abort(MPI_COMM_WORLD,-1);
  }

  // Divide one of the dimensions and reshuffle data
  int splitDiv = commSize/2;
  if (key == 0)
  {
    // Split along dimension z. No data movement necessary
    int pGridDimensionSize = std::nearbyint(std::pow(commSize,1./3.));
    int helper = pGridDimensionSize;
/*
    helper *= helper;
    int pGridCoordX = commRank%pGridDimensionSize;
    int pGridCoordY = (commRank%helper)/pGridDimensionSize;
    int pGridCoordZ = commRank/helper;
*/
    int color = commRank/splitDiv;
    MPI_Comm_split(commWorld,color,commRank,&sliceComm);
    if (color == 0)
    {
      InvertLower(matrixL, matrixLI, localShift, startX, startX+localShift, startY, startY+localShift, 1, sliceComm);
      // Expectation: these processors have matrixLI_1_1 distributed cyclically
      // Exchange data along z dimension so that data is replicated along the 3D grid, to keep MM3D's invariant correct
      // Future optimization: avoid copying twice, return data that we are about to exchange by fast rvalue!
      Matrix<T,U,MatrixStructureSquare,Distribution> tempT(std::vector<T>(), localShift, localShift, localDimension, localDimension);
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixLI, tempT, startX, startX+localShift, startY, startY+localShift);
      // Now, exchange data via a MPI_Sendrecv_replace
      MPI_Status stat;
      int exchangePartner = (commRank+splitDiv)%commSize;
      MPI_Sendrecv_replace(tempT.getRawData(), tempT.getNumElems(), MPI_DOUBLE, exchangePartner, 0, exchangePartner, 0, commWorld, &stat);
      // perform another Serialization to load into matrixLI
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixLI, tempT, startX+localShift, endX, startY+localShift, endY, true);
    }
    else
    {
      InvertLower(matrixL, matrixLI, localShift, startX+localShift, endX, startY+localShift, endY, 1, sliceComm);
      // Expectation: these processors have matrixLI_2_2 distributed cyclically
      // Exchange data along z dimension so that data is replicated along the 3D grid, to keep MM3D's invariant correct
      // Future optimization: avoid copying twice, return data that we are about to exchange by fast rvalue!
      Matrix<T,U,MatrixStructureSquare,Distribution> tempT(std::vector<T>(), localShift, localShift, localDimension, localDimension);
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixLI, tempT, startX+localShift, endX, startY+localShift, endY);
      // Now, exchange data via a MPI_Sendrecv_replace
      MPI_Status stat;
      int exchangePartner = (commRank+splitDiv)%commSize;
      MPI_Sendrecv_replace(tempT.getRawData(), tempT.getNumElems(), MPI_DOUBLE, exchangePartner, 0, exchangePartner, 0, commWorld, &stat);
      // perform another Serialization to load into matrixLI
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixLI, tempT, startX, startX+localShift, startY, startY+localShift, true);
    }

    // Finish with 2 MM3D calls
    // Future optimization: again, lots of copying. There is definitely a way to remove some of this.
    Matrix<T,U,MatrixStructureSquare,Distribution> tempInverse(std::vector<T>(localShift*localShift), localShift, localShift, localDimension, localDimension, true);
    blasEngineArgumentPackage_gemm<T> blasArgs;
    blasArgs.order = blasEngineOrder::AblasColumnMajor;
    blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
    blasArgs.transposeB = blasEngineTranspose::AblasNoTrans;
    blasArgs.alpha = 1.;
    blasArgs.beta = 0.;
    MM3D<T,U,blasEngine>::Multiply(matrixL, matrixLI, tempInverse, startX, startX+localShift, startY+localShift, endY, startX, startX+localShift, startY, startY+localShift, 0, localShift, 0, localShift, commWorld, blasArgs, true, true, false, 0);
    blasArgs.alpha = -1.;
    MM3D<T,U,blasEngine>::Multiply(matrixLI, tempInverse, matrixLI, startX+localShift, endX, startY+localShift, endY, 0, localShift, 0, localShift, startX, startX+localShift, startY+localShift, endY, commWorld, blasArgs, true, false, true, 0);
    MPI_Comm_free(&sliceComm);
  }
  else if (key == 1)
  {
    int bigDim = std::nearbyint(std::pow(2*commSize,1./3.));
    int smallDim = (bigDim>>1);
    int sliceRank = commRank%(bigDim*bigDim);
    int color = sliceRank/(bigDim*smallDim);
    //Split along dimension y (rows of p-grid)
    MPI_Comm_split(commWorld,color,commRank,&sliceComm);

    if (color == 0)
    {
      InvertLower(matrixL, matrixLI, localShift, startX, startX+localShift, startY, startY+localShift, 2, sliceComm);
      // Expectation: these processors have matrixLI_1_1 distributed cyclically
      // Exchange data along y dimension so that data is replicated along the 3D grid, to keep MM3D's invariant correct
      // Future optimization: avoid copying twice, return data that we are about to exchange by fast rvalue!
      Matrix<T,U,MatrixStructureSquare,Distribution> tempT(std::vector<T>(), localShift, localShift, localDimension, localDimension);
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixLI, tempT, startX, startX+localShift, startY, startY+localShift);
      // Now, exchange data via a MPI_Sendrecv_replace
      MPI_Status stat;
      int exchangePartner = commRank + bigDim*smallDim;
      MPI_Sendrecv_replace(tempT.getRawData(), tempT.getNumElems(), MPI_DOUBLE, exchangePartner, 0, exchangePartner, 0, commWorld, &stat);
      // perform another Serialization to load into matrixLI
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixLI, tempT, startX+localShift, endX, startY+localShift, endY, true);
    }
    else
    {
      InvertLower(matrixL, matrixLI, localShift, startX+localShift, endX, startY+localShift, endY, 2, sliceComm);
      // Expectation: these processors have matrixLI_2_2 distributed cyclically
      // Exchange data along y dimension so that data is replicated along the 3D grid, to keep MM3D's invariant correct
      // Future optimization: avoid copying twice, return data that we are about to exchange by fast rvalue!
      Matrix<T,U,MatrixStructureSquare,Distribution> tempT(std::vector<T>(), localShift, localShift, localDimension, localDimension);
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixLI, tempT, startX+localShift, endX, startY+localShift, endY);
      // Now, exchange data via a MPI_Sendrecv_replace
      MPI_Status stat;
      int exchangePartner = commRank - bigDim*smallDim;
      MPI_Sendrecv_replace(tempT.getRawData(), tempT.getNumElems(), MPI_DOUBLE, exchangePartner, 0, exchangePartner, 0, commWorld, &stat);
      // perform another Serialization to load into matrixLI
      Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixLI, tempT, startX, startX+localShift, startY, startY+localShift, true);
    }

    // Finish with 4 MM3D calls
    // Future optimization: again, lots of copying. There is definitely a way to remove some of this.
    Matrix<T,U,MatrixStructureSquare,Distribution> tempInverse(std::vector<T>(localShift*localShift), localShift, localShift, localDimension, localDimension, true);
    blasEngineArgumentPackage_gemm<T> blasArgs;
    blasArgs.order = blasEngineOrder::AblasColumnMajor;
    blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
    blasArgs.transposeB = blasEngineTranspose::AblasNoTrans;
    blasArgs.alpha = 1.;
    blasArgs.beta = 0.;
    MM3D<T,U,blasEngine>::Multiply(matrixL, matrixLI, tempInverse, startX, startX+localShift, startY+localShift, endY, startX, startX+localShift, startY, startY+localShift, 0, localShift, 0, localShift, commWorld, blasArgs, true, true, false, 0);
    // One more MM3D call to complete the matrix multiplication
    blasArgs.beta = 1;
    MM3D<T,U,blasEngine>::Multiply(matrixL, matrixLI, tempInverse, startX, startX+localShift, startY+localShift, endY, startX, startX+localShift, startY, startY+localShift, 0, localShift, 0, localShift, commWorld, blasArgs, true, true, false, 0, smallDim);

    blasArgs.beta = 0;
    blasArgs.alpha = -1.;
    MM3D<T,U,blasEngine>::Multiply(matrixLI, tempInverse, matrixLI, startX+localShift, endX, startY+localShift, endY, 0, localShift, 0, localShift, startX, startX+localShift, startY+localShift, endY, commWorld, blasArgs, true, false, true, 0);
    blasArgs.beta = 1;
    // One more MM3D call to complete the matrix multiplication
    MM3D<T,U,blasEngine>::Multiply(matrixLI, tempInverse, matrixLI, startX+localShift, endX, startY+localShift, endY, 0, localShift, 0, localShift, startX, startX+localShift, startY+localShift, endY, commWorld, blasArgs, true, false, true, 0, smallDim);

    MPI_Comm_free(&sliceComm);
    return;
  }
  else // key==2
  {
    // Split along dimension x
    //MPI_Comm_split(commWorld,pGridCoordX/splitDiv,commRank,&sliceComm);
    /*
    .. data swap
    .. form up new matrix
    .. recursive call
    */
    //MPI_Comm_free(&sliceComm);
  }
}


template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void RTI3D<T,U,blasEngine>::InvertUpper(
                  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixU,
                  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixUI,
                  U localDimension,
                  int key,
                  MPI_Comm commWorld
                )
{
  // check base cases
  int commSize;
  MPI_Comm_size(commWorld, &commSize);
  if (commSize == 4)
  {
    // Allgather and invert
  }
}


template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void RTI3D<T,U,blasEngine>::sliceExchangeBase(
                  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixT,
                  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixTI,
                  U localDimension,
                  U startX,
                  U endX,
                  U startY,
                  U endY,
                  MPI_Comm commWorld,
                  char dir
                )
{
  // Note: this won't work for Upper yet. Do that when Lower is fully working.

  // Processor grid has 8 processors. We do something special in this case
  int commSize,commRank;
  MPI_Comm_rank(commWorld, &commRank);
  MPI_Comm_size(commWorld, &commSize);
  MPI_Comm sliceComm;
  int pGridDimensionSize = std::nearbyint(std::pow(commSize,1./3.));
  int helper = pGridDimensionSize;
  helper *= helper;
  int pGridCoordX = commRank%pGridDimensionSize;
  int pGridCoordY = (commRank%helper)/pGridDimensionSize;
  int pGridCoordZ = commRank/helper;

  MPI_Comm_split(commWorld,pGridCoordZ,commRank,&sliceComm);
  U localShift = (localDimension>>1);   // will need to be changed in similar way to CFR3D when localDim is not a power of 2
  U numColumns = endX - startX;
  U numRows = endY - startY;

  if (pGridCoordZ == 0)
  {
    // Should be fast pass-by-value via move semantics
    std::vector<T> cyclicBaseCaseData = blockedToCyclicTransformation(matrixT, localShift, localDimension, startX, startX+localShift, startY, startY+localShift, 2, sliceComm);
    // Next: sequential triangular inverse.
    LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'L', 'N', localDimension, &cyclicBaseCaseData[0], localDimension);
    cyclicToLocalTransformation(cyclicBaseCaseData, localShift, localDimension, 2, commRank, 'L');
    // cut away extra data that isn't ours anymore
    cyclicBaseCaseData.resize(localShift*localShift);   // this will need changed for arbitrary dimensions.
    Matrix<T,U,MatrixStructureSquare,Distribution> tempT(std::move(cyclicBaseCaseData), localShift, localShift, localDimension, localDimension, true);
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixTI, tempT, startX, startX+localShift, startY, startY+localShift, true);
    // Now, exchange data via a MPI_Sendrecv_replace
    MPI_Status stat;
    MPI_Sendrecv_replace(tempT.getRawData(), tempT.getNumElems(), MPI_DOUBLE, (commRank+4)%8, 0, (commRank+4)%8, 0, commWorld, &stat);
    // perform another Serialization
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixTI, tempT, startX+localShift, endX, startY+localShift, endY, true);
  }
  else
  {
    // Should be fast pass-by-value via move semantics
    std::vector<T> cyclicBaseCaseData = blockedToCyclicTransformation(matrixT, localShift, localDimension, startX+localShift, endX, startY+localShift, endY, 2, sliceComm);
    // Next: sequential triangular inverse.
    LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'L', 'N', localDimension, &cyclicBaseCaseData[0], localDimension);
    cyclicToLocalTransformation(cyclicBaseCaseData, localShift, localDimension, 2, commRank-4, 'L');
    // cut away extra data that isn't ours anymore
    cyclicBaseCaseData.resize(localShift*localShift);   // this will need changed for arbitrary dimensions.
    Matrix<T,U,MatrixStructureSquare,Distribution> tempT(std::move(cyclicBaseCaseData), localShift, localShift, localDimension, localDimension, true);
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixTI, tempT, startX+localShift, endX, startY+localShift, endY, true);
    // Now, exchange data via a MPI_Sendrecv_replace
    MPI_Status stat;
    MPI_Sendrecv_replace(tempT.getRawData(), tempT.getNumElems(), MPI_DOUBLE, (commRank+4)%8, 0, (commRank+4)%8, 0, commWorld, &stat);
    // perform another Serialization
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixTI, tempT, startX, startX+localShift, startY, startY+localShift, true);
  }
  MPI_Comm_free(&sliceComm);
  return;
}


template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
std::vector<T> RTI3D<T,U,blasEngine>::blockedToCyclicTransformation(
									Matrix<T,U,MatrixStructureSquare,Distribution>& matT,
									U localDimension,
									U globalDimension,
									U matTstartX,
									U matTendX,
									U matTstartY,
									U matTendY,
									int pGridDimensionSize,
									MPI_Comm slice2Dcomm
								     )
{
  Matrix<T,U,MatrixStructureSquare,Distribution> baseCaseMatrixT(std::vector<T>(), localDimension, localDimension,
    globalDimension, globalDimension);
  Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matT, baseCaseMatrixT, matTstartX,
    matTendX, matTstartY, matTendY);

  U aggregDim = localDimension*pGridDimensionSize;
  // Later optimization: Serialize non-packed triangular matrix into a packed triangular matrix before AllGather
  //    then unpack back into a square (In the 4-loop structure below) before LAPACK call
  std::vector<T> blockedBaseCaseData(aggregDim*aggregDim);
  std::vector<T> cyclicBaseCaseData(aggregDim*aggregDim);
  MPI_Allgather(baseCaseMatrixT.getRawData(), baseCaseMatrixT.getNumElems(), MPI_DOUBLE,
    &blockedBaseCaseData[0], baseCaseMatrixT.getNumElems(), MPI_DOUBLE, slice2Dcomm);
  // Right now, we assume matrixA has Square Structure, if we want to let the user pass in just the unique part via a Triangular Structure,
  //   then we will need to change this.
  //   Note: this operation is just not cache efficient due to hopping around blockedBaseCaseData. Locality is not what we would like,
  //     but not sure it can really be improved here. Something to look into later.
  //   Also: Although (for LAPACKE_dpotrf), we need to allocate a square buffer and fill in (only) the lower (or upper) triangular portion,
  //     in the future, we might want to try different storage patterns from that one paper by Gustavsson


  // Strategy: We will write to cyclicBaseCaseData in proper order BUT will have to hop around blockedBaseCaseData. This should be ok since
  //   reading on modern computer architectures is less expensive via cache misses than writing, and we should not have any compulsory cache misses

  U numCyclicBlocksPerRowCol = globalDimension/pGridDimensionSize;
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
void RTI3D<T,U,blasEngine>::cyclicToLocalTransformation(
								std::vector<T>& storeT,
								U localDimension,
								U globalDimension,
								int pGridDimensionSize,
								int rankSlice,
								char dir
							     )
{
  U writeIndex = 0;
  U rowOffsetWithinBlock = rankSlice / pGridDimensionSize;
  U columnOffsetWithinBlock = rankSlice % pGridDimensionSize;
  U numCyclicBlocksPerRowCol = globalDimension/pGridDimensionSize;
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
        storeT[writeIndex] = storeT[readIndexCol*globalDimension + readIndexRow];
      }
      else
      {
        storeT[writeIndex] = 0.;
      }
      writeIndex++;
    }
  }
}
