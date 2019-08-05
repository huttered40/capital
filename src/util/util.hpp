/* Author: Edward Hutter */

template<typename MatrixType, typename RefMatrixType, typename LambdaType>
std::pair<typename MatrixType::ScalarType, typename MatrixType::ScalarType>
util::residual_local(MatrixType& Matrix, RefMatrixType& RefMatrix, LambdaType&& Lambda, MPI_Comm slice, size_t sliceX, size_t sliceY, size_t sliceDim){
  using T = typename MatrixType::ScalarType;
  using U = typename MatrixType::DimensionType;
  T error = 0;
  T control = 0;
  U localNumRows = Matrix.getNumRowsLocal();
  U localNumColumns = Matrix.getNumColumnsLocal();
  U globalX = sliceX;
  U globalY = sliceY;
  for (U i=0; i<localNumColumns; i++){
    globalY = sliceY;    // reset
    for (size_t j=0; j<localNumRows; j++){
      auto info = Lambda(Matrix, RefMatrix, i*localNumColumns+j,globalX, globalY);
      error += std::abs(info.first*info.first);
      control += std::abs(info.control*info.control);
      globalY += SquareCommInfo.d;
      //if (rank == 0) std::cout << val << " " << i << " " << j << std::endl;
    }
    globalX += SquareCommInfo.d;
  }
  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DATATYPE, MPI_SUM, slice);
  MPI_Allreduce(MPI_IN_PLACE, &control, 1, MPI_DATATYPE, MPI_SUM, slice);
  error = std::sqrt(error) / std::sqrt(control);
  //MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DATATYPE, MPI_SUM, depthComm);
  return error;


// Note: this method differs from the one below it because blockedData is in packed storage
template<typename T, typename U>
std::vector<T> util::blockedToCyclicSpecial(std::vector<T>& blockedData, U localDimensionRows, U localDimensionColumns, size_t sliceDim, char dir){
  TAU_FSTART(Util::blockedToCyclic);

  U aggregNumRows = localDimensionRows*sliceDim;
  U aggregNumColumns = localDimensionColumns*sliceDim;
  U aggregSize = aggregNumRows*aggregNumColumns;
  std::vector<T> cyclicData(aggregSize,0);
  U numCyclicBlocksPerRow = localDimensionRows;
  U numCyclicBlocksPerCol = localDimensionColumns;

  if (dir == 'L'){
    U writeIndex = 0;
    U recvDataOffset = blockedData.size()/(sliceDim*sliceDim);
    U off1 = (-1)*numCyclicBlocksPerRow-1;
    U off3 = sliceDim*recvDataOffset;
    // MACRO loop over all cyclic "blocks" (dimensionX direction)
    for (U i=0; i<numCyclicBlocksPerCol; i++){
      off1 += (numCyclicBlocksPerRow-i+1);
      // Inner loop over all columns in a cyclic "block"
      for (U j=0; j<sliceDim; j++){
        U off2 = j*recvDataOffset + off1;
        writeIndex += (i*sliceDim) + j;
        // Treat first block separately
        for (U z=j; z<sliceDim; z++){
          U readIndex = off2 + z*off3;
          cyclicData[writeIndex++] = blockedData[readIndex];
        }
        // Inner loop over all cyclic "blocks"
        for (U k=(i+1); k<numCyclicBlocksPerRow; k++){
          U off4 = off2;
          // Inner loop over all elements along columns
          for (U z=0; z<sliceDim; z++){
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
    U recvDataOffset = blockedData.size()/(sliceDim*sliceDim);
    U off1 = 0;
    U off3 = sliceDim*recvDataOffset;
    // MACRO loop over all cyclic "blocks" (dimensionX direction)
    for (U i=0; i<numCyclicBlocksPerCol; i++){
      off1 += i;
      // Inner loop over all columns in a cyclic "block"
      for (U j=0; j<sliceDim; j++){
        U off2 = j*recvDataOffset + off1;
        writeIndex = ((i*sliceDim)+j)*aggregNumRows;    //  reset each time
        // Inner loop over all cyclic "blocks"
        for (U k=0; k<i; k++){
          U off4 = off2;
          // Inner loop over all elements along columns
          for (U z=0; z<sliceDim; z++){
            U readIndex = off4 + z*off3 + k;
            cyclicData[writeIndex++] = blockedData[readIndex];
          }
        }
        
        // Special final block
        U off4 = off2;
        // Inner loop over all elements along columns
        for (U z=0; z<=j; z++){
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
std::vector<T> util::blockedToCyclic(std::vector<T>& blockedData, U localDimensionRows, U localDimensionColumns, size_t sliceDim){
  TAU_FSTART(Util::blockedToCyclic);

  U aggregNumRows = localDimensionRows*sliceDim;
  U aggregNumColumns = localDimensionColumns*sliceDim;
  U aggregSize = aggregNumRows*aggregNumColumns;
  std::vector<T> cyclicData(aggregSize);
  U numCyclicBlocksPerRow = localDimensionRows;
  U numCyclicBlocksPerCol = localDimensionColumns;
  U writeIndex = 0;
  U recvDataOffset = localDimensionRows*localDimensionColumns;
  // MACRO loop over all cyclic "blocks" (dimensionX direction)
  for (U i=0; i<numCyclicBlocksPerCol; i++){
    // Inner loop over all columns in a cyclic "block"
    for (U j=0; j<sliceDim; j++){
      // Inner loop over all cyclic "blocks"
      for (U k=0; k<numCyclicBlocksPerRow; k++){
        // Inner loop over all elements along columns
        for (U z=0; z<sliceDim; z++){
          U readIndex = i*numCyclicBlocksPerRow + j*recvDataOffset + k + z*sliceDim*recvDataOffset;
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

template<typename MatrixType>
std::vector<typename MatrixType::ScalarType>
util::getReferenceMatrix(MatrixType& myMatrix, size_t key, MPI_Comm slice, size_t commDim){
  TAU_FSTART(Util::getReferenceMatrix);

  using T = typename MatrixType::ScalarType;
  using U = typename MatrixType::DimensionType;
  using Structure = typename MatrixType::StructureType;
  using Distribution = typename MatrixType::DistributionType;
  using Offload = typename MatrixType::OffloadType;

  size_t commDim = std::get<4>(commInfo);

  U localNumColumns = myMatrix.getNumColumnsLocal();
  U localNumRows = myMatrix.getNumRowsLocal();
  U globalNumColumns = myMatrix.getNumColumnsGlobal();
  U globalNumRows = myMatrix.getNumRowsGlobal();
/*
  using MatrixType = Matrix<T,U,Square,Distribution>;
  MatrixType localMatrix(globalNumColumns, globalNumRows, commDim, commDim);
  localMatrix.DistributeSymmetric(pGridCoordX, pGridCoordY, commDim, commDim, key, true);
*/
  // I first want to check whether or not I want to serialize into a rectangular buffer (I don't care too much about efficiency here,
  //   if I did, I would serialize after the AllGather, but whatever)
  T* matrixPtr = myMatrix.getRawData();
  Matrix<T,U,Rectangular,Distribution,Offload> matrixDest(std::vector<T>(), localNumColumns, localNumRows, globalNumColumns, globalNumRows);
  if ((!std::is_same<Structure,Rectangular>::value)
    && (!std::is_same<Structure,Square>::value)){
    Serializer<Structure,Rectangular>::Serialize(myMatrix, matrixDest);
    matrixPtr = matrixDest.getRawData();
  }

  U aggregNumRows = localNumRows*commDim;
  U aggregNumColumns = localNumColumns*commDim;
  U localSize = localNumColumns*localNumRows;
  U globalSize = globalNumColumns*globalNumRows;
  U aggregSize = aggregNumRows*aggregNumColumns;
  std::vector<T> blockedMatrix(aggregSize);
//  std::vector<T> cyclicMatrix(aggregSize);
  MPI_Allgather(matrixPtr, localSize, MPI_DATATYPE, &blockedMatrix[0], localSize, MPI_DATATYPE, slice);

  std::vector<T> cyclicMatrix = util::blockedToCyclic(blockedMatrix, localNumRows, localNumColumns, commDim);

  // In case there are hidden zeros, we will recopy
  if ((globalNumRows%commDim) || (globalNumColumns%commDim)){
    U index = 0;
    for (U i=0; i<globalNumColumns; i++){
      for (U j=0; j<globalNumRows; j++){
        cyclicMatrix[index++] = cyclicMatrix[i*aggregNumRows+j];
      }
    }
    // In this case, globalSize < aggregSize
    cyclicMatrix.resize(globalSize);
  }
  TAU_FSTOP(Util::getReferenceMatrix);
  return cyclicMatrix;
}

template<typename MatrixType, typename CommType>
void util::transposeSwap(MatrixType& mat, CommType&& CommInfo){
  TAU_FSTART(Util::transposeSwap);

  size_t SquareFaceSize = CommInfo.c*CommInfo.d;
  size_t transposePartner = CommInfo.x*SquareFaceSize + CommInfo.y*CommInfo.c + CommInfo.z;
  //if (myRank != transposeRank)
  //{
    // Transfer with transpose rank
    MPI_Sendrecv_replace(mat.getRawData(), mat.getNumElems(), MPI_DATATYPE, transposeRank, 0, transposeRank, 0, comm, MPI_STATUS_IGNORE);

    // Note: the received data that now resides in mat is NOT transposed, and the Matrix structure is LowerTriangular
    //       This necesitates making the "else" processor serialize its data L11^{-1} from a square to a LowerTriangular,
    //       since we need to make sure that we call a MM::multiply routine with the same Structure, or else segfault.

  //}
  TAU_FSTOP(Util::transposeSwap);
}

template<typename U>
U util::getNextPowerOf2(U localShift){
  TAU_FSTART(Util::getNextPowerOf2);

  if ((localShift & (localShift-1)) != 0){
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

template<typename MatrixType>
void util::removeTriangle(MatrixType& matrix, size_t sliceX, size_t sliceY, size_t sliceDim, char dir){
  TAU_FSTART(Util::removeTriangle);

  using U = typename MatrixType::DimensionType;

  U globalDimVert = sliceY;
  U globalDimHoriz = sliceX;
  U localVert = matrix.getNumRowsLocal();
  U localHoriz = matrix.getNumColumnsLocal();
  for (U i=0; i<localHoriz; i++){
    globalDimVert = sliceY;    //   reset
    for (U j=0; j<localVert; j++){
      if ((globalDimVert < globalDimHoriz) && (dir == 'L')){
        matrix.getRawData()[i*localVert + j] = 0;
      }
      if ((globalDimVert > globalDimHoriz) && (dir == 'U')){
        matrix.getRawData()[i*localVert + j] = 0;
      }
      globalDimVert += sliceDim;
    }
    globalDimHoriz += sliceDim;
  }
  TAU_FSTOP(Util::removeTriangle);
}

void util::processAveragesFromFile(std::ofstream& fptrAvg, std::string& fileStrTotal, size_t numFuncs, size_t numIterations, size_t rank){
  if (rank == 0){
    std::ifstream fptrTotal2(fileStrTotal.c_str());
    //debugging
    if (!fptrTotal2.is_open()){
      abort();
    }
    using profileType = std::tuple<std::string,size_t,double,double,double,double>;
    std::vector<profileType> profileVector(numFuncs, std::make_tuple("",0,0,0,0,0));
    for (size_t i=0; i<numIterations; i++){
      // read in first item on line: iteration #
      size_t numIter;
      fptrTotal2 >> numIter;
      for (size_t j=0; j<numFuncs; j++){
        std::string funcName;
        size_t numCalls;
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
    for (size_t i=0; i<numFuncs; i++){
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

template<typename T>
void util::InitialGEMM(){
  // Function must be called before performance testing is done due to MKL implementation of GEMM
  std::vector<T> matrixA(128*128,0.);
  std::vector<T> matrixB(128*128,0.);
  std::vector<T> matrixC(128*128,0.);
  blasEngineArgumentPackage_gemm<T> gemmPack1(blasEngineOrder::AblasColumnMajor, blasEngineTranspose::AblasNoTrans, blasEngineTranspose::AblasNoTrans, 1., 0.);
  blasEngine::_gemm(&matrixA[0], &matrixB[0], &matrixC[0], 128, 128, 128, 128, 128, 128, gemmPack1);
}
