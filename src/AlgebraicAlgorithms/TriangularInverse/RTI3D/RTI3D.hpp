/*
  Author: Edward Hutter
*/

template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void RTI3D<T,U,blasEngine>::Invert(
              Matrix<T,U,MatrixStructureSquare,Distribution>& matrixT,
              U localDimension,
              char dir,
              MPI_Comm commWorld
            )
{
  if (dir == 'L')
  {
    // call InvertLower
    InvertLower(matrixT, localDimension, 0, commWorld);
  }
  else
  {
    // call InverseUpper
    InvertUpper(matrixT, localDimension, 0, commWorld);
  }
}

template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void RTI3D<T,U,blasEngine>::InvertLower(
                  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixL,
                  U localDimension,
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
    // Invert
    LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'L', 'N', localDimension, matrixL.getRawData(), localDimension);
    return;
  }

  if (commSize == 4)
  {
    // Allgather, Invert, Shuffle data
    // Should be fast pass-by-value via move semantics
    std::vector<T> cyclicBaseCaseData = blockedToCyclicTransformation(matrixL, localDimension, localDimension*2, 2, commWorld);

    // Next: sequential triangular inverse.
    LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'L', 'N', localDimension*2, &cyclicBaseCaseData[0], localDimension*2);

    cyclicToLocalTransformation(cyclicBaseCaseData, localDimension, localDimension*2, 2, commRank, 'L');
    // cut away extra data that isn't ours anymore
    cyclicBaseCaseData.resize(localDimension*localDimension);
    matrixL.getVectorData() = std::move(cyclicBaseCaseData);
    return;
  }

  MPI_Comm sliceComm;
  int pGridDimensionSize = std::nearbyint(std::pow(commSize,1./3.));
  int helper = pGridDimensionSize;
  helper *= helper;
  int pGridCoordX = commRank%pGridDimensionSize;
  int pGridCoordY = (commRank%helper)/pGridDimensionSize;
  int pGridCoordZ = commRank/helper;

  if (commSize == 8)
  {
    // Split into two 2x2 faces based on z dimension of cubic processor grid.
    MPI_Comm_split(commWorld,pGridCoordZ,commRank,&sliceComm);
    InvertLower(matrixL, localDimension, key+1, sliceComm);
    MPI_Comm_free(&sliceComm);
    return;
  }

  // Divide one of the dimensions and reshuffle data
  int splitDiv = pGridDimensionSize/2;
  if (key%3 == 0)
  {
    // Split along dimension z
    MPI_Comm_split(commWorld,pGridCoordZ/splitDiv,commRank,&sliceComm);
    /*
    .. data swap
    .. form up new matrix
    .. recursive call
    */
    MPI_Comm_free(&sliceComm);
  }
  else if (key%3 == 1)
  {
    // Split along dimension y
    MPI_Comm_split(commWorld,pGridCoordY/splitDiv,commRank,&sliceComm);
    /*
    .. data swap
    .. form up new matrix
    .. recursive call
    */
    MPI_Comm_free(&sliceComm);
  }
  else
  {
    // Split along dimension x
    MPI_Comm_split(commWorld,pGridCoordX/splitDiv,commRank,&sliceComm);
    /*
    .. data swap
    .. form up new matrix
    .. recursive call
    */
    MPI_Comm_free(&sliceComm);
  }
}

template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void RTI3D<T,U,blasEngine>::InvertUpper(
                  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixU,
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
std::vector<T> RTI3D<T,U,blasEngine>::blockedToCyclicTransformation(
									Matrix<T,U,MatrixStructureSquare,Distribution>& matT,
									U localDimension,
									U globalDimension,
									int pGridDimensionSize,
									MPI_Comm slice2Dcomm
								     )
{
  // Later optimization: Serialize non-packed triangular matrix into a packed triangular matrix before AllGather
  //    then unpack back into a square (In the 4-loop structure below) before LAPACK call

  std::vector<T> blockedBaseCaseData(globalDimension*globalDimension);
  std::vector<T> cyclicBaseCaseData(globalDimension*globalDimension);
  MPI_Allgather(matT.getRawData(), matT.getNumElems(), MPI_DOUBLE,
    &blockedBaseCaseData[0], matT.getNumElems(), MPI_DOUBLE, slice2Dcomm);

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
