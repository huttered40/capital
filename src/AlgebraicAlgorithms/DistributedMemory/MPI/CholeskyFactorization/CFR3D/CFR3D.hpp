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
    std::cout << "Base case has been reached with " << globalDimension << " " << localDimension << " " << bcDimension << "\n";

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

    Matrix<T,U,MatrixStructureSquare,Distribution> baseCaseMatrixA(std::vector<T>(), localDimension, localDimension,
      globalDimension, globalDimension);
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixA, baseCaseMatrixA, matAstartX,
      matAendX, matAstartY, matAendY);

    std::vector<T> blockedBaseCaseData(bcDimension*bcDimension);
    std::vector<T>cyclicBaseCaseData(bcDimension*bcDimension);
    MPI_Allgather(baseCaseMatrixA.getRawData(), sizeof(T)*baseCaseMatrixA.getNumElems(), MPI_CHAR,
      &blockedBaseCaseData[0], sizeof(T)*baseCaseMatrixA.getNumElems(), MPI_CHAR, slice2D);

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



    // Now, I want to use something similar to a template class for libraries conforming to the standards of LAPACK, such as FLAME.
    //   I want to be able to mix and match.

    std::vector<T>& storeL = cyclicBaseCaseData;
    std::vector<T> storeLI = storeL;

    // Until then, assume a double datatype and simply use LAPACKE_dpotrf. Worry about adding more capabilities later.
    LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', bcDimension, &storeL[0], bcDimension);

    // Now, we have L_{11} located inside the "square" vector cyclicBaseCaseData.
    //   We need to call the "move builder" constructor in order to "move" this "rawData" into its own matrix.
    //   Only then, can we call Serializer into the real matrixL. WRONG! We need to find the data we own according to the cyclic rule first!
    //    So it doesn't make any sense to "move" these into Matrices yet.
    // Finally, we need that data for calling the triangular inverse.

    // Next: sequential triangular inverse. Question: does DTRTRI require packed storage or square storage? I think square, so that it can use BLAS-3.
    LAPACKE_dtrtri(LAPACK_ROW_MAJOR, 'L', 'N', bcDimension, &storeLI[0], bcDimension);

    // Only truly a "square-to-square" serialization because we store matrixL as a square (no packed storage yet!)

    // Now, before we can serialize into matrixL and matrixLI, we need to save the values that this processor owns according to the cyclic rule.
    // Only then can we serialize.

    // Iterate and pick out. I would like not to have to create any more memory and I would only like to iterate once, not twice, for storeL and storeLI
    //   Use the "overwrite" trick that I have used in CASI code, as well as other places

    // I am going to use a sneaky trick: I will take the vectorData from storeL and storeLI by reference, overwrite its values,
    //   and then "move" them cheaply into new Matrix structures before I call Serialize on them individually.

    writeIndex = 0;
    U rowOffsetWithinBlock = rankSlice / pGridDimensionSize;
    U columnOffsetWithinBlock = rankSlice % pGridDimensionSize;
    // MACRO loop over all cyclic "blocks"
    for (U i=0; i<numCyclicBlocksPerRowCol; i++)
    {
      // We know which row corresponds to our processor in each cyclic "block"
      // Inner loop over all cyclic "blocks" partitioning up the columns
      for (U j=0; j<numCyclicBlocksPerRowCol; j++)
      {
        // We know which column corresponds to our processor in each cyclic "block"
        U readIndex = j*pGridDimensionSize + columnOffsetWithinBlock + i*(bcDimension*pGridDimensionSize) + rowOffsetWithinBlock*bcDimension;
        storeL[writeIndex] = cyclicBaseCaseData[readIndex];
        storeLI[writeIndex] = cyclicBaseCaseData[readIndex];
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
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixLI, tempLI, matLstartX, matLendX, matLstartY, matLendY, true);

    return;
  }

  U localShift = (localDimension>>1);
  U globalShift = (globalDimension>>1);
  rFactor(matrixA, matrixL, matrixLI, localShift, bcDimension, globalShift,
    matAstartX, matAstartX+localShift, matAstartY, matAstartY+localShift,
    matLstartX, matLstartX+localShift, matLstartY, matLstartY+localShift,
    matLIstartX, matLIstartX+localShift, matLIstartY, matLIstartY+localShift, transposePartner, commWorld);

  int rank;
  // use MPI_COMM_WORLD for this p2p communication for transpose, but could use a smaller communicator
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  std::cout << "I am rank " << rank << " and Local dimension - " << localDimension << std::endl;

  // Regardless of whether or not we don't need to communicate, we still need to serialize into a square buffer
  if (rank != transposePartner)
  {
    // Serialize the square matrix into the nonzero lower triangular (so avoid sending the zeros in the upper-triangular part of matrix)
    Matrix<T,U,MatrixStructureLowerTriangular,Distribution> packedMatrix(std::vector<T>(), localShift, localShift, globalShift, globalShift);
    // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
    Serializer<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular>::Serialize(matrixLI, packedMatrix,
      matLIstartX, matLIstartX+localShift, matLIstartY, matLIstartY+localShift);
 
    // Transfer with transpose rank
    MPI_Sendrecv_replace(packedMatrix.getRawData(), sizeof(T)*packedMatrix.getNumElems(), MPI_CHAR, transposePartner, 0, transposePartner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Note: the received data that now resides in packedMatrix is NOT transposed, and the Matrix structure is LowerTriangular
    //       This necesitates making the "else" processor serialize its data L11^{-1} from a square to a LowerTriangular,
    //       since we need to make sure that we call a MM::multiply routine with the same Structure, or else segfault.

    // Now, we want to be able to move this "raw, temporary, and anonymous" buffer into a Matrix structure, to be passed into
      // MatrixMultiplication at no cost. We definitely do not want to copy, we want to move!
    // But wait! That buffer is exactly what we need anyway
    // Wait again!! Why do we need to do this? The structure is already LowerTriangular. Just keep it and have the non-transpose processor
    //   serialize his Square matrix structure into a lower triangular

    // Call matrix multiplication that does not cut up matrix B in C <- AB
    //    Need to set up the struct that has useful BLAS info

    // I am using gemm right now, but I might want to use dtrtri or something due to B being triangular at heart
    blasEngineArgumentPackage_gemm<T> blasArgs;
    blasArgs.order = blasEngineOrder::AblasRowMajor;
    blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
    blasArgs.transposeB = blasEngineTranspose::AblasTrans;
    blasArgs.alpha = 1.;
    blasArgs.beta = 1.;
    SquareMM3D<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular,MatrixStructureSquare,blasEngine>::
      Multiply(matrixA, packedMatrix, matrixL, matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY,
        0, localShift, 0, localShift, matLstartX, matLstartX+localShift, matLstartY+localShift, matLendY, commWorld, blasArgs, true, false, true);
  }
  else
  {
    // For processors that are their own transpose within the slice they are on in a 3D processor grid.
    // We want to serialize LI from Square into LowerTriangular so it can match the "transposed" processors that did it to send half the words for one reason
    Matrix<T,U,MatrixStructureLowerTriangular,Distribution> tempLI(std::vector<T>(), localShift, localShift, globalShift, globalShift);
    Serializer<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular>::Serialize(matrixLI, tempLI, matLIstartX,
      matLIstartX+localShift, matLIstartY, matLIstartY+localShift);

    // I am using gemm right now, but I might want to use dtrtri or something due to B being triangular at heart
    blasEngineArgumentPackage_gemm<T> blasArgs;
    blasArgs.order = blasEngineOrder::AblasRowMajor;
    blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
    blasArgs.transposeB = blasEngineTranspose::AblasTrans;
    blasArgs.alpha = 1.;
    blasArgs.beta = 1.;
    SquareMM3D<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular,MatrixStructureSquare,blasEngine>::
      Multiply(matrixA, tempLI, matrixL, matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY,
        0, localShift, 0, localShift, matLstartX, matLstartX+localShift, matLstartY+localShift, matLendY, commWorld, blasArgs, true, false, true);
  }

  // Now we need to perform L_{21}L_{21}^T via syrk

  Matrix<T,U,MatrixStructureSquare,Distribution> holdLsyrk(std::vector<T>(localShift*localShift), localShift, localShift, globalShift, globalShift, true);
  blasEngineArgumentPackage_syrk<T> syrkPackage;
  syrkPackage.order = blasEngineOrder::AblasRowMajor;
  syrkPackage.uplo = blasEngineUpLo::AblasLower;
  syrkPackage.transposeA = blasEngineTranspose::AblasNoTrans;
  syrkPackage.alpha = 1.;
  syrkPackage.beta = 1.;
  SquareMM3D<T,U,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare,blasEngine>::Multiply(matrixL, holdLsyrk,
    matLstartX, matLstartX+localShift, matLstartY+localShift, matLendY, 0, localShift, 0, localShift, commWorld, syrkPackage, true, false);

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
  rFactor(matrixAquadrant4, matrixL, matrixLI, localShift, bcDimension, globalShift,
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
  invPackage1.beta = 1.;
  SquareMM3D<T,U,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare,blasEngine>::Multiply(matrixL, matrixLI,
    tempInverse, matLstartX, matLstartX+localShift, matLstartY+localShift, matLendY, matLIstartX, matLIstartX+localShift, matLIstartY,
      matLIstartY+localShift, 0, localShift, 0, localShift, commWorld, invPackage1, true, true, false);
}
