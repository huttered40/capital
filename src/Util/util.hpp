/* Author: Edward Hutter */

template<typename T, typename U>
std::vector<T> util<T,U>::blockedToCyclic(std::vector<T>& blockedData, U localDimensionRows, U localDimensionColumns, int pGridDimensionSize)
{
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
  MPI_Allgather(matrixPtr, localSize, MPI_DOUBLE, &blockedMatrix[0], localSize, MPI_DOUBLE, sliceComm);

  std::vector<T> cyclicMatrix = util<T,U>::blockedToCyclic(blockedMatrix, localNumRows, localNumColumns, pGridDimensionSize);

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
  if (myRank != transposeRank)
  {
    // Transfer with transpose rank
    MPI_Sendrecv_replace(mat.getRawData(), mat.getNumElems(), MPI_DOUBLE, transposeRank, 0, transposeRank, 0, commWorld, MPI_STATUS_IGNORE);

    // Note: the received data that now resides in mat is NOT transposed, and the Matrix structure is LowerTriangular
    //       This necesitates making the "else" processor serialize its data L11^{-1} from a square to a LowerTriangular,
    //       since we need to make sure that we call a MM::multiply routine with the same Structure, or else segfault.

  }
}

template<typename T, typename U>
std::tuple<MPI_Comm, int, int, int, int> util<T,U>::getCommunicatorSlice(MPI_Comm commWorld)
{
  int rank,size;
  MPI_Comm_rank(commWorld, &rank);
  MPI_Comm_size(commWorld, &size);

  int pGridDimensionSize = std::nearbyint(std::pow(size,1./3.));
  
  int helper = pGridDimensionSize;
  helper *= helper;
  int pCoordX = rank%pGridDimensionSize;
  int pCoordY = (rank%helper)/pGridDimensionSize;
  int pCoordZ = rank/helper;

  MPI_Comm sliceComm;
  MPI_Comm_split(commWorld, pCoordZ, rank, &sliceComm);
  return std::make_tuple(sliceComm, pCoordX, pCoordY, pCoordZ, pGridDimensionSize); 
}

template<typename T, typename U>
template< template<typename,typename,template<typename,typename,int> class> class StructureArg1,
  template<typename,typename,template<typename,typename,int> class> class StructureArg2,
  template<typename,typename,template<typename,typename,int> class> class StructureArg3,
  template<typename,typename,int> class Distribution>
void util<T,U>::validateResidualParallel(
                        Matrix<T,U,StructureArg1,Distribution>& matrixA,
                        Matrix<T,U,StructureArg2,Distribution>& matrixB,
                        Matrix<T,U,StructureArg3,Distribution>& matrixC,
                        char dir,
                        MPI_Comm commWorld
                      )
{
  int rank,size;
  MPI_Comm_rank(commWorld, &rank);
  MPI_Comm_size(commWorld, &size);

  auto commInfo = getCommunicatorSlice(commWorld);
  MPI_Comm sliceComm = std::get<0>(commInfo);
  U pGridCoordX = std::get<1>(commInfo);
  U pGridCoordY = std::get<2>(commInfo);
  U pGridCoordZ = std::get<3>(commInfo);
  U pGridDimensionSize = std::get<4>(commInfo);
  int helper = pGridDimensionSize;
  helper *= helper;

  if (dir == 'L')
  {
    blasEngineArgumentPackage_gemm<T> blasArgs;
    blasArgs.order = blasEngineOrder::AblasColumnMajor;
    blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
    blasArgs.transposeB = blasEngineTranspose::AblasTrans;
    blasArgs.alpha = 1.;
    blasArgs.beta = -1.;
    MM3D<T,U,cblasEngine>::Multiply(matrixA, matrixB, matrixC, commWorld, blasArgs);
  }
  else if (dir == 'U')
  {
    blasEngineArgumentPackage_gemm<T> blasArgs;
    blasArgs.order = blasEngineOrder::AblasColumnMajor;
    blasArgs.transposeA = blasEngineTranspose::AblasTrans;
    blasArgs.transposeB = blasEngineTranspose::AblasNoTrans;
    blasArgs.alpha = 1.;
    blasArgs.beta = -1.;
    MM3D<T,U,cblasEngine>::Multiply(matrixA, matrixB, matrixC, commWorld, blasArgs);
  }
  else if (dir == 'F')
  {
    blasEngineArgumentPackage_gemm<T> blasArgs;
    blasArgs.order = blasEngineOrder::AblasColumnMajor;
    blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
    blasArgs.transposeB = blasEngineTranspose::AblasNoTrans;
    blasArgs.alpha = 1.;
    blasArgs.beta = -1.;
    MM3D<T,U,cblasEngine>::Multiply(matrixA, matrixB, matrixC, commWorld, blasArgs);
  }
  else if (dir == 'I')
  {
    blasEngineArgumentPackage_gemm<T> blasArgs;
    blasArgs.order = blasEngineOrder::AblasColumnMajor;
    blasArgs.transposeA = blasEngineTranspose::AblasTrans;
    blasArgs.transposeB = blasEngineTranspose::AblasNoTrans;
    blasArgs.alpha = 1.;
    blasArgs.beta = 0;
    MM3D<T,U,cblasEngine>::Multiply(matrixA, matrixB, matrixC, commWorld, blasArgs);
  }
  else
  {
    abort();
  }

  // Now just calculate residual
  T error = 0;
  U localNumRows = matrixC.getNumRowsLocal();
  U localNumColumns = matrixC.getNumColumnsLocal();
  U globalX = pGridCoordX;
  U globalY = pGridCoordY;
  for (U i=0; i<localNumColumns; i++)
  {
    globalY = pGridCoordY;    // reset
    for (int j=0; j<localNumRows; j++)
    {
      T val = 0;
      if ((dir == 'F') || ((dir == 'L') && (globalY >= globalX)) || ((dir == 'U') && (globalY <= globalX)))
      {
        val = matrixC.getRawData()[i*localNumRows+j];
      }
      else if (dir == 'I')
      {
        if (globalX == globalY)
        {
          val = std::abs(1. - matrixC.getRawData()[i*localNumRows+j]);
        }
        else
        {
          val = matrixC.getRawData()[i*localNumRows+j];
        }
        if (matrixC.getRawData()[i*localNumRows+j] > .5) {std::cout << "CHECK THIS at global " << globalX << " " << globalY <<  std::endl;}
      }
      val *= val;
      //if (rank == 0) std::cout << val << " " << i << " " << j << std::endl;
      error += std::abs(val);
      globalY += pGridDimensionSize;
    }
    globalX += pGridDimensionSize;
  }
  error = std::sqrt(error);
  //std::cout << "localError = " << error << std::endl;
  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, sliceComm);
  if (rank == 0) {std::cout << "Residual error = " << error << std::endl;}
  MPI_Comm_free(&sliceComm);
}

template<typename T, typename U>
template< template<typename,typename,template<typename,typename,int> class> class StructureArg,
  template<typename,typename,int> class Distribution>
void util<T,U>::validateOrthogonalityParallel(
                        Matrix<T,U,StructureArg,Distribution>& matrixQ,
                        MPI_Comm commWorld
                      )
{
  int rank,size;
  MPI_Comm_rank(commWorld, &rank);
  MPI_Comm_size(commWorld, &size);

  auto commInfo = getCommunicatorSlice(commWorld);
  MPI_Comm sliceComm = std::get<0>(commInfo);
  U pGridCoordX = std::get<1>(commInfo);
  U pGridCoordY = std::get<2>(commInfo);
  U pGridCoordZ = std::get<3>(commInfo);
  U pGridDimensionSize = std::get<4>(commInfo);
  int helper = pGridDimensionSize;
  helper *= helper;
  int transposePartner = pGridCoordZ*helper + pGridCoordX*pGridDimensionSize + pGridCoordY;

  Matrix<T,U,StructureArg,Distribution> matrixQtrans = matrixQ;
  util<T,U>::transposeSwap(matrixQtrans, rank, transposePartner, commWorld);
  U localNumRows = matrixQtrans.getNumColumnsLocal();
  U localNumColumns = matrixQ.getNumColumnsLocal();
  std::cout << "check these - " << localNumRows << " " << localNumColumns << std::endl;
  U globalNumRows = matrixQtrans.getNumColumnsGlobal();
  U globalNumColumns = matrixQ.getNumColumnsGlobal();
  U numElems = localNumRows*localNumColumns;
  Matrix<T,U,StructureArg,Distribution> matrixI(std::vector<T>(numElems,0), localNumColumns, localNumRows, globalNumColumns, globalNumRows, true);
  util<T,U>::validateResidualParallel(matrixQtrans,matrixQ,matrixI,'I',commWorld);
  MPI_Comm_free(&sliceComm);
}
