/* Author: Edward Hutter */

template<typename T, typename U>
std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int> util<T,U>::build3DTopology(
                    MPI_Comm commWorld
                  )
{
  TAU_FSTART(setUpCommunicators);
  int rank,size;
  MPI_Comm_rank(commWorld, &rank);
  MPI_Comm_size(commWorld, &size);

  int pGridDimensionSize = std::nearbyint(std::ceil(pow(size,1./3.)));
  int helper = pGridDimensionSize;
  helper *= helper;
  #ifdef BLUEWATERS
  int pGridCoordZ = rank%pGridDimensionSize;
  int pGridCoordY = rank/helper;
  int pGridCoordX = (rank%helper)/pGridDimensionSize;

  MPI_Comm rowComm, columnComm, sliceComm, depthComm;

  // First, split the 3D Cube processor grid communicator into groups based on what 2D slice they are located on.
  // Then, subdivide further into row groups and column groups
  MPI_Comm_split(commWorld, rank/pGridDimensionSize, rank, &depthComm);
  MPI_Comm_split(commWorld, pGridCoordZ, rank, &sliceComm);
  MPI_Comm_split(sliceComm, pGridCoordY, pGridCoordX, &rowComm);
  MPI_Comm_split(sliceComm, pGridCoordX, pGridCoordY, &columnComm);
  #else
  int pGridCoordX = rank%pGridDimensionSize;
  int pGridCoordY = (rank%helper)/pGridDimensionSize;
  int pGridCoordZ = rank/helper;

  MPI_Comm rowComm, columnComm, sliceComm, depthComm;

  // First, split the 3D Cube processor grid communicator into groups based on what 2D slice they are located on.
  // Then, subdivide further into row groups and column groups
  MPI_Comm_split(commWorld, pGridCoordY+pGridDimensionSize*pGridCoordX, rank, &depthComm);
  MPI_Comm_split(commWorld, pGridCoordZ, rank, &sliceComm);
  MPI_Comm_split(sliceComm, pGridCoordY, pGridCoordX, &rowComm);
  MPI_Comm_split(sliceComm, pGridCoordX, pGridCoordY, &columnComm);
  #endif


  TAU_FSTOP(setUpCommunicators);
  return std::make_tuple(rowComm, columnComm, sliceComm, depthComm, pGridCoordX, pGridCoordY, pGridCoordZ);
}

template<typename T, typename U>
std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm> util<T,U>::buildTunableTopology(
      MPI_Comm commWorld, int pGridDimensionD, int pGridDimensionC)
{
  TAU_FSTART(getTunableCommunicators);
  int worldRank, worldSize, columnRank, cubeRank;
  MPI_Comm_rank(commWorld, &worldRank);
  MPI_Comm_size(commWorld, &worldSize);

  #ifdef BLUEWATERS
  int SubCubeSize = pGridDimensionC*pGridDimensionC*pGridDimensionC;
  int SubCubeSliceSize = pGridDimensionC*pGridDimensionC;
  MPI_Comm sliceComm, rowComm, columnComm, columnContigComm, columnAltComm, depthComm, miniCubeComm;
  MPI_Comm_split(commWorld, worldRank/SubCubeSize, worldRank, &miniCubeComm);
  MPI_Comm_rank(miniCubeComm, &cubeRank);
  int temp1 = (cubeRank%pGridDimensionC) + pGridDimensionC*(cubeRank/SubCubeSliceSize);
  int temp2 = worldRank % SubCubeSliceSize;
  MPI_Comm_split(miniCubeComm, cubeRank/pGridDimensionC, cubeRank, &depthComm);
  MPI_Comm_split(miniCubeComm, temp1, cubeRank, &rowComm);
  // Note: columnComm in the tunable grid is of size d, not size c like in the 3D grid
  MPI_Comm_split(commWorld, temp2, worldRank, &columnComm);
  MPI_Comm_split(commWorld, worldRank%pGridDimensionC, worldRank, &sliceComm);
  MPI_Comm_rank(columnComm, &columnRank);
  MPI_Comm_split(columnComm, columnRank/pGridDimensionC, columnRank, &columnContigComm);
  MPI_Comm_split(columnComm, columnRank%pGridDimensionC, columnRank, &columnAltComm); 
  #else
  int sliceSize = pGridDimensionD*pGridDimensionC;
  MPI_Comm sliceComm, rowComm, columnComm, columnContigComm, columnAltComm, depthComm, miniCubeComm;
  MPI_Comm_split(commWorld, worldRank/sliceSize, worldRank, &sliceComm);
  MPI_Comm_rank(sliceComm, &sliceRank);
  MPI_Comm_split(sliceComm, sliceRank/pGridDimensionC, sliceRank, &rowComm);
  int cubeInt = worldRank%sliceSize;
  MPI_Comm_split(commWorld, cubeInt/(pGridDimensionC*pGridDimensionC), worldRank, &miniCubeComm);
  MPI_Comm_split(commWorld, cubeInt, worldRank, &depthComm);
  MPI_Comm_split(sliceComm, sliceRank%pGridDimensionC, sliceRank, &columnComm);
  MPI_Comm_rank(columnComm, &columnRank);
  MPI_Comm_split(columnComm, columnRank/pGridDimensionC, columnRank, &columnContigComm);
  MPI_Comm_split(columnComm, columnRank%pGridDimensionC, columnRank, &columnAltComm); 
  #endif

  MPI_Comm_free(&columnComm);
  TAU_FSTOP(getTunableCommunicators);
  return std::make_tuple(rowComm, columnContigComm, columnAltComm, depthComm, sliceComm, miniCubeComm);
}

template<typename T, typename U>
void util<T,U>::destroy3DTopology(std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D)
{
  MPI_Comm_free(&std::get<0>(commInfo3D));
  MPI_Comm_free(&std::get<1>(commInfo3D));
  MPI_Comm_free(&std::get<2>(commInfo3D));
  MPI_Comm_free(&std::get<3>(commInfo3D));
}

template<typename T, typename U>
void util<T,U>::destroyTunableTopology(std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm>& commInfoTunable)
{
  MPI_Comm_free(&std::get<0>(commInfoTunable));
  MPI_Comm_free(&std::get<1>(commInfoTunable));
  MPI_Comm_free(&std::get<2>(commInfoTunable));
  MPI_Comm_free(&std::get<3>(commInfoTunable));
  MPI_Comm_free(&std::get<4>(commInfoTunable));
  MPI_Comm_free(&std::get<5>(commInfoTunable));
}

// Note: this method differs from the one below it because blockedData is in packed storage
template<typename T, typename U>
std::vector<T> util<T,U>::blockedToCyclicSpecial(
  std::vector<T>& blockedData, U localDimensionRows, U localDimensionColumns, int pGridDimensionSize, char dir)
{
  TAU_FSTART(Util::blockedToCyclic);
  U aggregNumRows = localDimensionRows*pGridDimensionSize;
  U aggregNumColumns = localDimensionColumns*pGridDimensionSize;
  U aggregSize = aggregNumRows*aggregNumColumns;
  std::vector<T> cyclicData(aggregSize,0);
  U numCyclicBlocksPerRow = localDimensionRows;
  U numCyclicBlocksPerCol = localDimensionColumns;

  if (dir == 'L')
  {
    U writeIndex = 0;
    U recvDataOffset = blockedData.size()/(pGridDimensionSize*pGridDimensionSize);
    U off1 = (-1)*numCyclicBlocksPerRow-1;
    U off3 = pGridDimensionSize*recvDataOffset;
    // MACRO loop over all cyclic "blocks" (dimensionX direction)
    for (U i=0; i<numCyclicBlocksPerCol; i++)
    {
      off1 += (numCyclicBlocksPerRow-i+1);
      // Inner loop over all columns in a cyclic "block"
      for (U j=0; j<pGridDimensionSize; j++)
      {
        U off2 = j*recvDataOffset + off1;
        writeIndex += (i*pGridDimensionSize) + j;
        // Treat first block separately
        for (U z=j; z<pGridDimensionSize; z++)
        {
          U readIndex = off2 + z*off3;
          cyclicData[writeIndex++] = blockedData[readIndex];
        }
        // Inner loop over all cyclic "blocks"
        for (U k=(i+1); k<numCyclicBlocksPerRow; k++)
        {
          U off4 = off2;
          // Inner loop over all elements along columns
          for (U z=0; z<pGridDimensionSize; z++)
          {
            U readIndex = off4 + z*off3 + (k-i);
            cyclicData[writeIndex++] = blockedData[readIndex];
          }
        }
      }
    }
  }
  else /* dir == 'U'*/
  {
    U writeIndex = 0;
    U recvDataOffset = blockedData.size()/(pGridDimensionSize*pGridDimensionSize);
    U off1 = 0;
    U off3 = pGridDimensionSize*recvDataOffset;
    // MACRO loop over all cyclic "blocks" (dimensionX direction)
    for (U i=0; i<numCyclicBlocksPerCol; i++)
    {
      off1 += i;
      // Inner loop over all columns in a cyclic "block"
      for (U j=0; j<pGridDimensionSize; j++)
      {
        U off2 = j*recvDataOffset + off1;
        writeIndex = ((i*pGridDimensionSize)+j)*aggregNumRows;    //  reset each time
        // Inner loop over all cyclic "blocks"
        for (U k=0; k<i; k++)
        {
          U off4 = off2;
          // Inner loop over all elements along columns
          for (U z=0; z<pGridDimensionSize; z++)
          {
            U readIndex = off4 + z*off3 + k;
            cyclicData[writeIndex++] = blockedData[readIndex];
          }
        }
        
        // Special final block
        U off4 = off2;
        // Inner loop over all elements along columns
        for (U z=0; z<=j; z++)
        {
          U readIndex = off4 + z*off3 + i;
          cyclicData[writeIndex++] = blockedData[readIndex];
        }
      }
    }
  }

  // Should be quick pass-by-value via move semantics, since we are effectively returning a localvariable that is going to lose its scope anyways,
  //   so the compiler should be smart enough to use the move constructor for the vector in the caller function.
  TAU_FSTOP(Util::blockedToCyclic);
  return cyclicData;

}

template<typename T, typename U>
std::vector<T> util<T,U>::blockedToCyclic(
  std::vector<T>& blockedData, U localDimensionRows, U localDimensionColumns, int pGridDimensionSize)
{
  TAU_FSTART(Util::blockedToCyclic);
  U aggregNumRows = localDimensionRows*pGridDimensionSize;
  U aggregNumColumns = localDimensionColumns*pGridDimensionSize;
  U aggregSize = aggregNumRows*aggregNumColumns;
  std::vector<T> cyclicData(aggregSize);
  U numCyclicBlocksPerRow = localDimensionRows;
  U numCyclicBlocksPerCol = localDimensionColumns;
  U writeIndex = 0;
  U recvDataOffset = localDimensionRows*localDimensionColumns;
  // MACRO loop over all cyclic "blocks" (dimensionX direction)
  for (U i=0; i<numCyclicBlocksPerCol; i++)
  {
    // Inner loop over all columns in a cyclic "block"
    for (U j=0; j<pGridDimensionSize; j++)
    {
      // Inner loop over all cyclic "blocks"
      for (U k=0; k<numCyclicBlocksPerRow; k++)
      {
        // Inner loop over all elements along columns
        for (U z=0; z<pGridDimensionSize; z++)
        {
          U readIndex = i*numCyclicBlocksPerRow + j*recvDataOffset + k + z*pGridDimensionSize*recvDataOffset;
          cyclicData[writeIndex++] = blockedData[readIndex];
        }
      }
    }
  }

  // Should be quick pass-by-value via move semantics, since we are effectively returning a localvariable that is going to lose its scope anyways,
  //   so the compiler should be smart enough to use the move constructor for the vector in the caller function.
  TAU_FSTOP(Util::blockedToCyclic);
  return cyclicData;

}

template<typename T, typename U>
template<template<typename,typename, template<typename,typename,int> class> class StructureArg,
  template<typename,typename,int> class Distribution>					// Added additional template parameters just for this method
std::vector<T> util<T,U>::getReferenceMatrix(
              Matrix<T,U,StructureArg,Distribution>& myMatrix,
							U key,
							std::tuple<MPI_Comm, int, int, int, int> commInfo
						  )
{
  MPI_Comm sliceComm = std::get<0>(commInfo);
  int pGridCoordX = std::get<1>(commInfo);
  int pGridCoordY = std::get<2>(commInfo);
  int pGridCoordZ = std::get<3>(commInfo);
  int pGridDimensionSize = std::get<4>(commInfo);

  U localNumColumns = myMatrix.getNumColumnsLocal();
  U localNumRows = myMatrix.getNumRowsLocal();
  U globalNumColumns = myMatrix.getNumColumnsGlobal();
  U globalNumRows = myMatrix.getNumRowsGlobal();
/*
  using MatrixType = Matrix<T,U,MatrixStructureSquare,Distribution>;
  MatrixType localMatrix(globalNumColumns, globalNumRows, pGridDimensionSize, pGridDimensionSize);
  localMatrix.DistributeSymmetric(pGridCoordX, pGridCoordY, pGridDimensionSize, pGridDimensionSize, key, true);
*/
  // I first want to check whether or not I want to serialize into a rectangular buffer (I don't care too much about efficiency here,
  //   if I did, I would serialize after the AllGather, but whatever)
  T* matrixPtr = myMatrix.getRawData();
  Matrix<T,U,MatrixStructureRectangle,Distribution> matrixDest(std::vector<T>(), localNumColumns, localNumRows, globalNumColumns, globalNumRows);
  if ((!std::is_same<StructureArg<T,U,Distribution>,MatrixStructureRectangle<T,U,Distribution>>::value)
    && (!std::is_same<StructureArg<T,U,Distribution>,MatrixStructureSquare<T,U,Distribution>>::value))		// compile time if statement. Branch prediction should be correct.
  {
    Serializer<T,U,StructureArg,MatrixStructureRectangle>::Serialize(myMatrix, matrixDest);
    matrixPtr = matrixDest.getRawData();
  }

  U aggregNumRows = localNumRows*pGridDimensionSize;
  U aggregNumColumns = localNumColumns*pGridDimensionSize;
  U localSize = localNumColumns*localNumRows;
  U globalSize = globalNumColumns*globalNumRows;
  U aggregSize = aggregNumRows*aggregNumColumns;
  std::vector<T> blockedMatrix(aggregSize);
//  std::vector<T> cyclicMatrix(aggregSize);
  MPI_Allgather(matrixPtr, localSize, MPI_DATATYPE, &blockedMatrix[0], localSize, MPI_DATATYPE, sliceComm);

  std::vector<T> cyclicMatrix = util<T,U>::blockedToCyclic(
    blockedMatrix, localNumRows, localNumColumns, pGridDimensionSize);

  // In case there are hidden zeros, we will recopy
  if ((globalNumRows%pGridDimensionSize) || (globalNumColumns%pGridDimensionSize))
  {
    U index = 0;
    for (U i=0; i<globalNumColumns; i++)
    {
      for (U j=0; j<globalNumRows; j++)
      {
        cyclicMatrix[index++] = cyclicMatrix[i*aggregNumRows+j];
      }
    }
    // In this case, globalSize < aggregSize
    cyclicMatrix.resize(globalSize);
  }
  return cyclicMatrix;
}

template<typename T, typename U>
template< template<typename,typename,template<typename,typename,int> class> class StructureArg,template<typename,typename,int> class Distribution>
void util<T,U>::transposeSwap(
											Matrix<T,U,StructureArg,Distribution>& mat,
											int myRank,
											int transposeRank,
											MPI_Comm commWorld
										     )
{
  TAU_FSTART(Util::transposeSwap);
  //if (myRank != transposeRank)
  //{
    // Transfer with transpose rank
    MPI_Sendrecv_replace(mat.getRawData(), mat.getNumElems(), MPI_DATATYPE, transposeRank, 0, transposeRank, 0, commWorld, MPI_STATUS_IGNORE);

    // Note: the received data that now resides in mat is NOT transposed, and the Matrix structure is LowerTriangular
    //       This necesitates making the "else" processor serialize its data L11^{-1} from a square to a LowerTriangular,
    //       since we need to make sure that we call a MM::multiply routine with the same Structure, or else segfault.

  //}
  TAU_FSTOP(Util::transposeSwap);
}

template<typename T, typename U>
std::tuple<MPI_Comm, int, int, int, int> util<T,U>::getCommunicatorSlice(
  MPI_Comm commWorld)
{
  TAU_FSTART(Util::getCommunicatorSlice);
  int rank,size;
  MPI_Comm_rank(commWorld, &rank);
  MPI_Comm_size(commWorld, &size);

  int pGridDimensionSize = std::nearbyint(std::pow(size,1./3.));
  
  int helper = pGridDimensionSize;
  helper *= helper;
  #ifdef BLUEWATERS
  int pCoordZ = rank%pGridDimensionSize;
  int pCoordY = rank/helper;
  int pCoordX = (rank%helper)/pGridDimensionSize;
  #else
  int pCoordX = rank%pGridDimensionSize;
  int pCoordY = (rank%helper)/pGridDimensionSize;
  int pCoordZ = rank/helper;
  #endif

  MPI_Comm sliceComm;
  MPI_Comm_split(commWorld, pCoordZ, rank, &sliceComm);
  TAU_FSTOP(Util::getCommunicatorSlice);
  return std::make_tuple(sliceComm, pCoordX, pCoordY, pCoordZ, pGridDimensionSize); 
}

template<typename T, typename U>
U util<T,U>::getNextPowerOf2(U localShift)
{
  TAU_FSTART(Util::getNextPowerOf2);
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
  TAU_FSTOP(Util::getNextPowerOf2);
  return localShift;
}

template<typename T, typename U>
template< template<typename,typename,template<typename,typename,int> class> class StructureArg,
  template<typename,typename,int> class Distribution>
void util<T,U>::removeTriangle(Matrix<T,U,StructureArg,Distribution>& matrix, int pGridCoordX, int pGridCoordY, int pGridDimensionSize, char dir)
{
  U globalDimVert = pGridCoordY;
  U globalDimHoriz = pGridCoordX;
  U localVert = matrix.getNumRowsLocal();
  U localHoriz = matrix.getNumColumnsLocal();
  for (U i=0; i<localHoriz; i++)
  {
    globalDimVert = pGridCoordY;    //   reset
    for (U j=0; j<localVert; j++)
    {
      if ((globalDimVert < globalDimHoriz) && (dir == 'L'))
      {
        matrix.getRawData()[i*localVert + j] = 0;
      }
      if ((globalDimVert > globalDimHoriz) && (dir == 'U'))
      {
        matrix.getRawData()[i*localVert + j] = 0;
      }
      globalDimVert += pGridDimensionSize;
    }
    globalDimHoriz += pGridDimensionSize;
  }
}

template<typename T, typename U>
void util<T,U>::processAveragesFromFile(std::ofstream& fptrAvg, std::string& fileStrTotal, int numFuncs, int numIterations, int rank)
{
  if (rank == 0)
  {
    std::ifstream fptrTotal2(fileStrTotal.c_str());
    //debugging
    if (!fptrTotal2.is_open())
    {
      abort();
    }
    using profileType = std::tuple<std::string,int,double,double,double,double>;
    std::vector<profileType> profileVector(numFuncs, std::make_tuple("",0,0,0,0,0));
    for (int i=0; i<numIterations; i++)
    {
      // read in first item on line: iteration #
      int numIter;
      fptrTotal2 >> numIter;
      for (int j=0; j<numFuncs; j++)
      {
        std::string funcName;
        int numCalls;
        double info1,info2,info3,info4;
        fptrTotal2 >> funcName >> numCalls >> info1 >> info2 >> info3 >> info4;
	// Below statement is for debugging
	std::cout << "check this: " << funcName << " " << numCalls << " " << info1 << " " << info2 << " " << info3 << " " << info4 << std::endl;
        std::get<0>(profileVector[j]) = funcName;
        std::get<1>(profileVector[j]) = numCalls;
        std::get<2>(profileVector[j]) += info1;
        std::get<3>(profileVector[j]) += info2;
        std::get<4>(profileVector[j]) += info3;
        std::get<5>(profileVector[j]) += info4;
      }
    }
    for (int i=0; i<numFuncs; i++)
    {
      if (i>0) fptrAvg << "\t";
      fptrAvg << std::get<0>(profileVector[i]).c_str();
      fptrAvg << "\t" << std::get<1>(profileVector[i]);
      fptrAvg << "\t" << std::get<2>(profileVector[i])/numIterations;
      fptrAvg << "\t" << std::get<3>(profileVector[i])/numIterations;
      fptrAvg << "\t" << std::get<4>(profileVector[i])/numIterations;
      fptrAvg << "\t" << std::get<5>(profileVector[i])/numIterations;
    }
    fptrAvg << std::endl;
    fptrTotal2.close();
  }
}

template<typename T, typename U>
void util<T,U>::InitialGEMM()
{
  // Function must be called before performance testing is done due to MKL implementation of GEMM
  std::vector<T> matrixA(128*128,0.);
  std::vector<T> matrixB(128*128,0.);
  std::vector<T> matrixC(128*128,0.);
  blasEngineArgumentPackage_gemm<T> gemmPack1(blasEngineOrder::AblasColumnMajor, blasEngineTranspose::AblasNoTrans, blasEngineTranspose::AblasNoTrans, 1., 0.);
  cblasEngine<T,U>::_gemm(&matrixA[0], &matrixB[0], &matrixC[0], 128, 128, 128, 128, 128, 128, gemmPack1);
}
