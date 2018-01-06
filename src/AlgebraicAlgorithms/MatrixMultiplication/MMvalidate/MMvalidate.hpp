/* Author: Edward Hutter */


static std::tuple<MPI_Comm, int, int, int, int> getCommunicatorSlice(MPI_Comm commWorld)
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

// We enforce that matrixSol must have Square Structure.

template<typename T, typename U, template<typename,typename> class blasEngine>
template<template<typename,typename, template<typename,typename,int> class> class StructureArg,
  template<typename,typename,int> class Distribution>					// Added additional template parameters just for this method
void MMvalidate<T,U,blasEngine>::validateLocal(
            		//.. lets change this up: I want to pass in 2 input matrices, and the outputmatrix I attained. That is simpler for me.
                        //.. then I also need to watch out for serialization, say if I pass in a UT matrix or LT, etc.
		        Matrix<T,U,StructureArg,Distribution>& myMatrix,
                        U localDimensionM,
                        U localDimensionN,
                        U localDimensionK,
                        U globalDimensionM,
                        U globalDimensionN,
                        U globalDimensionK,
                        MPI_Comm commWorld,
                        const blasEngineArgumentPackage_gemm<T>& srcPackage
                      )
{
  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  int myRank,sliceRank;
  MPI_Comm_rank(commWorld, &myRank);

  std::tuple<MPI_Comm, int, int, int, int> commInfo = getCommunicatorSlice(commWorld);
  MPI_Comm sliceComm = std::get<0>(commInfo);
  MPI_Comm_rank(sliceComm, &sliceRank);
  int pGridCoordX = std::get<1>(commInfo);
  int pGridCoordY = std::get<2>(commInfo);
  int pGridCoordZ = std::get<3>(commInfo);
  int pGridDimensionSize = std::get<4>(commInfo);

  // Locally generate each matrix, then AllGather along the slice communicator. Buid the entire matrix. Only then can we feed into LAPACK/BLAS routines
  // Fast pass-by-value via modern C++ move semantics
  std::vector<T> matrixAforEngine = getReferenceMatrix(myMatrix, localDimensionK, localDimensionM, globalDimensionK, globalDimensionM, pGridCoordX*pGridDimensionSize+pGridCoordY, commInfo);
  std::vector<T> matrixBforEngine = getReferenceMatrix(myMatrix, localDimensionN, localDimensionK, globalDimensionN, globalDimensionK, (pGridCoordX*pGridDimensionSize+pGridCoordY)*(-1), commInfo);
  std::vector<T> matrixCforEngine(globalDimensionM*globalDimensionN, 0);	// No matrix needed for this. Only used in BLAS call

  // Assume column-major matrix and no transposes
  blasEngine<T,U>::_gemm(&matrixAforEngine[0], &matrixBforEngine[0], &matrixCforEngine[0], globalDimensionM, globalDimensionN,
    globalDimensionK, globalDimensionM, globalDimensionK, globalDimensionM, srcPackage);

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error = getResidualSquare(myMatrix.getVectorData(), matrixCforEngine, localDimensionM, localDimensionN, globalDimensionM,
    globalDimensionN, commInfo);

  // Now, we need the AllReduce of the error. Very cheap operation in terms of bandwidth cost, since we are only communicating a single double primitive type.
  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, sliceComm);
  error = std::sqrt(error);
  if (sliceRank == 0) {std::cout << "Total error = " << error << std::endl;}

  MPI_Comm_free(&sliceComm);
}

template<typename T, typename U, template<typename,typename> class blasEngine>
template<template<typename,typename, template<typename,typename,int> class> class StructureArg,
  template<typename,typename,int> class Distribution>					// Added additional template parameters just for this method
void MMvalidate<T,U,blasEngine>::validateLocal(
                        Matrix<T,U,StructureArg,Distribution>& myMatrix,
                        U localDimensionM,
                        U localDimensionN,
                        U globalDimensionM,
                        U globalDimensionN,
                        MPI_Comm commWorld,
                        const blasEngineArgumentPackage_trmm<T>& srcPackage
                      )
{
  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  int myRank,sliceRank;
  MPI_Comm_rank(commWorld, &myRank);

  std::tuple<MPI_Comm, int, int, int, int> commInfo = getCommunicatorSlice(commWorld);
  MPI_Comm sliceComm = std::get<0>(commInfo);
  MPI_Comm_rank(sliceComm, &sliceRank);
  int pGridCoordX = std::get<1>(commInfo);
  int pGridCoordY = std::get<2>(commInfo);
  int pGridCoordZ = std::get<3>(commInfo);
  int pGridDimensionSize = std::get<4>(commInfo);

  // Locally generate each matrix, then AllGather along the slice communicator. Buid the entire matrix. Only then can we feed into LAPACK/BLAS routines
  // Fast pass-by-value via modern C++ move semantics
  int localTriDim = (srcPackage.side == blasEngineSide::AblasLeft ? localDimensionM : localDimensionN);
  int globalTriDim = (srcPackage.side == blasEngineSide::AblasLeft ? globalDimensionM : globalDimensionN);
  std::vector<T> matrixAforEngine = getReferenceMatrix(myMatrix, localTriDim, localTriDim, globalTriDim, globalTriDim, pGridCoordX*pGridDimensionSize+pGridCoordY, commInfo);
  std::vector<T> matrixBforEngine = getReferenceMatrix(myMatrix, localDimensionN, localDimensionM, globalDimensionN, globalDimensionM, (pGridCoordX*pGridDimensionSize+pGridCoordY)*(-1), commInfo);

  blasEngine<T,U>::_trmm(&matrixAforEngine[0], &matrixBforEngine[0], globalDimensionM, globalDimensionN,
    (srcPackage.side == blasEngineSide::AblasLeft ? globalDimensionM : globalDimensionN),
    (srcPackage.order == blasEngineOrder::AblasColumnMajor ? globalDimensionM : globalDimensionN), srcPackage);

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error = getResidualSquare(myMatrix.getVectorData(), matrixBforEngine, localDimensionM, localDimensionN, globalDimensionM,
    globalDimensionN, commInfo);

  // Now, we need the AllReduce of the error. Very cheap operation in terms of bandwidth cost, since we are only communicating a single double primitive type.
  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, sliceComm);
  error = std::sqrt(error);
  if (sliceRank == 0) {std::cout << "Total error = " << error << std::endl;}

  MPI_Comm_free(&sliceComm);
}

template<typename T, typename U, template<typename,typename> class blasEngine>
template<template<typename,typename, template<typename,typename,int> class> class StructureArg,
  template<typename,typename,int> class Distribution>					// Added additional template parameters just for this method
void MMvalidate<T,U,blasEngine>::validateLocal(
                        Matrix<T,U,StructureArg,Distribution>& myMatrix,
                        U localDimensionN,
                        U localDimensionK,
                        U globalDimensionN,
                        U globalDimensionK,
                        MPI_Comm commWorld,
                        const blasEngineArgumentPackage_syrk<T>& srcPackage
                      )
{
  // not quite correct yet
  abort();

  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  int myRank,sliceRank;
  MPI_Comm_rank(commWorld, &myRank);

  std::tuple<MPI_Comm, int, int, int, int> commInfo = getCommunicatorSlice(commWorld);
  MPI_Comm sliceComm = std::get<0>(commInfo);
  MPI_Comm_rank(sliceComm, &sliceRank);
  int pGridCoordX = std::get<1>(commInfo);
  int pGridCoordY = std::get<2>(commInfo);
  int pGridCoordZ = std::get<3>(commInfo);
  int pGridDimensionSize = std::get<4>(commInfo);

  // Locally generate each matrix, then AllGather along the slice communicator. Buid the entire matrix. Only then can we feed into LAPACK/BLAS routines
  // Fast pass-by-value via modern C++ move semantics
  std::vector<T> matrixAforEngine = getReferenceMatrix(myMatrix, localDimensionK, localDimensionN, globalDimensionK, globalDimensionN, pGridCoordX*pGridDimensionSize+pGridCoordY, commInfo);
  std::vector<T> matrixBforEngine(globalDimensionN*globalDimensionK);	// Instead of using C for output matrix, lets use B as we did in SquareMM3D

  // Assume column major and no transpose and that the matrix A is square.
  blasEngine<T,U>::_syrk(&matrixAforEngine[0], &matrixBforEngine[0], globalDimensionK, globalDimensionN,
    globalDimensionK, globalDimensionN, srcPackage);

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error = getResidualSquare(myMatrix.getVectorData(), matrixBforEngine, localDimensionN, localDimensionK, globalDimensionN,
    globalDimensionK, commInfo);

  // Now, we need the AllReduce of the error. Very cheap operation in terms of bandwidth cost, since we are only communicating a single double primitive type.
  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, sliceComm);
  error = std::sqrt(error);
  if (sliceRank == 0) {std::cout << "Total error = " << error << std::endl;}

  MPI_Comm_free(&sliceComm);
}
  
template<typename T, typename U, template<typename,typename> class blasEngine>
T MMvalidate<T,U,blasEngine>::getResidualSquare(
		     std::vector<T>& myValues,
		     std::vector<T>& blasValues,
		     U localDimensionM,
		     U localDimensionN,
		     U globalDimensionM,
		     U globalDimensionN,
		     std::tuple<MPI_Comm, int, int, int, int> commInfo
		   )
{
  T error = 0;
  int pCoordX = std::get<1>(commInfo);
  int pCoordY = std::get<2>(commInfo);
  int pGridDimensionSize = std::get<4>(commInfo);
  U myIndex = 0;
  U solIndex = pCoordX *globalDimensionM + pCoordY;

  for (U i=0; i<localDimensionN; i++)
  {
    U saveCountRef = solIndex;
    for (U j=0; j<localDimensionM; j++)
    {
      T errorSquare = std::abs(myValues[myIndex] - blasValues[solIndex]);
      //std::cout << errorSquare << " " << myValues[myIndex] << " " << blasValues[solIndex] << std::endl;
      errorSquare *= errorSquare;
      error += errorSquare;
      solIndex += pGridDimensionSize;
      myIndex++;
    }
    solIndex = saveCountRef + pGridDimensionSize*globalDimensionM;
  }

  //error = std::sqrt(error);
  //std::cout << "Processor residual error - " << error << std::endl;
  return error;
}


template<typename T, typename U, template<typename,typename> class blasEngine>
template<template<typename,typename, template<typename,typename,int> class> class StructureArg,
  template<typename,typename,int> class Distribution>					// Added additional template parameters just for this method
std::vector<T> MMvalidate<T,U,blasEngine>::getReferenceMatrix(
								Matrix<T,U,StructureArg,Distribution>& myMatrix,
								U localNumColumns,
								U localNumRows,
								U globalNumColumns,
								U globalNumRows,
								U key,
								std::tuple<MPI_Comm, int, int, int, int> commInfo
							     )
{
  //.. new idea here: I don't want to re-generate A and B. I just want to AllGather what I originally made in test.cpp, and then get my cyclic part.

  MPI_Comm sliceComm = std::get<0>(commInfo);
  int pGridCoordX = std::get<1>(commInfo);
  int pGridCoordY = std::get<2>(commInfo);
  int pGridCoordZ = std::get<3>(commInfo);
  int pGridDimensionSize = std::get<4>(commInfo);

  using MatrixType = Matrix<T,U,StructureArg,Distribution>;
  MatrixType localMatrix(localNumColumns, localNumRows, globalNumColumns, globalNumRows);
  localMatrix.DistributeRandom(pGridCoordX, pGridCoordY, pGridDimensionSize, pGridDimensionSize, key);

  // debugging
  localMatrix.print();
  std::cout << "\n\n\n\n";

  std::vector<T> blockedMatrix(globalNumColumns*globalNumRows);
  std::vector<T> cyclicMatrix(globalNumColumns*globalNumRows);
  U localSize = localNumColumns*localNumRows;
  MPI_Allgather(localMatrix.getRawData(), localSize, MPI_DOUBLE, &blockedMatrix[0], localSize, MPI_DOUBLE, sliceComm);

  U numCyclicBlocksPerRow = globalNumRows/pGridDimensionSize;
  U numCyclicBlocksPerCol = globalNumColumns/pGridDimensionSize;
  U writeIndex = 0;
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
          U readIndex = i*numCyclicBlocksPerRow + j*localSize + k + z*pGridDimensionSize*localSize;
          cyclicMatrix[writeIndex++] = blockedMatrix[readIndex];
        }
      }
    }
  }

  return cyclicMatrix;
}
