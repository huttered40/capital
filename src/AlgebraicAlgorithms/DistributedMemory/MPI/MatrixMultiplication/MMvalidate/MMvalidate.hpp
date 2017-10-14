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
template<template<typename,typename,int> class Distribution>
void MMvalidate<T,U,blasEngine>::validateLocal(
                        Matrix<T,U,MatrixStructureSquare,Distribution>& myMatrix,
                        U localDimensionX,
                        U localDimensionY,
                        U localDimensionZ,
                        U globalDimensionX,
                        U globalDimensionY,
                        U globalDimensionZ,
                        MPI_Comm commWorld,
                        const blasEngineArgumentPackage_gemm<T>& srcPackage
                      )
{
  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  int myRank;
  MPI_Comm_rank(commWorld, &myRank);

  std::tuple<MPI_Comm, int, int, int, int> commInfo = getCommunicatorSlice(commWorld);
  MPI_Comm sliceComm = std::get<0>(commInfo);
  int pGridCoordX = std::get<1>(commInfo);
  int pGridCoordY = std::get<2>(commInfo);
  int pGridCoordZ = std::get<3>(commInfo);
  int pGridDimensionSize = std::get<4>(commInfo);

  // Locally generate each matrix, then AllGather along the slice communicator. Buid the entire matrix. Only then can we feed into LAPACK/BLAS routines
  // Fast pass-by-value via modern C++ move semantics
  std::vector<T> matrixAforEngine = getReferenceMatrix(myMatrix, localDimensionX, localDimensionY, globalDimensionX, globalDimensionY, pGridCoordX*pGridDimensionSize+pGridCoordY, commInfo);
  std::vector<T> matrixBforEngine = getReferenceMatrix(myMatrix, localDimensionZ, localDimensionX, globalDimensionZ, globalDimensionX, (pGridCoordX*pGridDimensionSize+pGridCoordY)*(-1), commInfo);
  std::vector<T> matrixCforEngine(globalDimensionY*globalDimensionZ, 0);	// No matrix needed for this. Only used in BLAS call

  // Assume column-major matrix and no transposes
  blasEngine<T,U>::_gemm(&matrixAforEngine[0], &matrixBforEngine[0], &matrixCforEngine[0], globalDimensionY, globalDimensionZ,
    globalDimensionX, globalDimensionY, globalDimensionX, globalDimensionY, srcPackage);

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error = getResidualSquare(myMatrix.getVectorData(), matrixCforEngine, localDimensionX, localDimensionY, localDimensionZ, globalDimensionX,
    globalDimensionY, globalDimensionZ, commInfo);

  // Now, we need the AllReduce of the error. Very cheap operation in terms of bandwidth cost, since we are only communicating a single double primitive type.
  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, sliceComm);
  if (myRank == 0) {std::cout << "Total error = " << error << std::endl;}

  MPI_Comm_free(&sliceComm);
}

template<typename T, typename U, template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void MMvalidate<T,U,blasEngine>::validateLocal(
                        Matrix<T,U,MatrixStructureSquare,Distribution>& myMatrix,
                        U localDimensionX,
                        U localDimensionY,
                        U localDimensionZ,
                        U globalDimensionX,
                        U globalDimensionY,
                        U globalDimensionZ,
                        MPI_Comm commWorld,
                        const blasEngineArgumentPackage_trmm<T>& srcPackage
                      )
{
  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  int myRank;
  MPI_Comm_rank(commWorld, &myRank);

  std::tuple<MPI_Comm, int, int, int, int> commInfo = getCommunicatorSlice(commWorld);
  MPI_Comm sliceComm = std::get<0>(commInfo);
  int pGridCoordX = std::get<1>(commInfo);
  int pGridCoordY = std::get<2>(commInfo);
  int pGridCoordZ = std::get<3>(commInfo);
  int pGridDimensionSize = std::get<4>(commInfo);

  // Locally generate each matrix, then AllGather along the slice communicator. Buid the entire matrix. Only then can we feed into LAPACK/BLAS routines
  // Fast pass-by-value via modern C++ move semantics
  std::vector<T> matrixAforEngine = getReferenceMatrix(myMatrix, localDimensionX, localDimensionY, globalDimensionX, globalDimensionY, pGridCoordX*pGridDimensionSize+pGridCoordY, commInfo);
  std::vector<T> matrixBforEngine = getReferenceMatrix(myMatrix, localDimensionZ, localDimensionX, globalDimensionZ, globalDimensionX, (pGridCoordX*pGridDimensionSize+pGridCoordY)*(-1), commInfo);


  blasEngine<T,U>::_trmm(&matrixAforEngine[0], &matrixBforEngine[0], (srcPackage.side == blasEngineSide::AblasLeft ? globalDimensionX : globalDimensionY),
    (srcPackage.side == blasEngineSide::AblasLeft ? globalDimensionZ : globalDimensionX), (srcPackage.side == blasEngineSide::AblasLeft ? globalDimensionY : globalDimensionX),
    (srcPackage.side == blasEngineSide::AblasLeft ? globalDimensionX : globalDimensionY), srcPackage);

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error = getResidualSquare(myMatrix.getVectorData(), matrixBforEngine, localDimensionX, localDimensionY, localDimensionZ, globalDimensionX,
    globalDimensionY, globalDimensionZ, commInfo);

  // Now, we need the AllReduce of the error. Very cheap operation in terms of bandwidth cost, since we are only communicating a single double primitive type.
  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, sliceComm);
  if (myRank == 0) {std::cout << "Total error = " << error << std::endl;}

  MPI_Comm_free(&sliceComm);
}

template<typename T, typename U, template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void MMvalidate<T,U,blasEngine>::validateLocal(
                        Matrix<T,U,MatrixStructureSquare,Distribution>& myMatrix,
                        U localDimensionX,
                        U localDimensionY,
                        U localDimensionZ,
                        U globalDimensionX,
                        U globalDimensionY,
                        U globalDimensionZ,
                        MPI_Comm commWorld,
                        const blasEngineArgumentPackage_syrk<T>& srcPackage
                      )
{
  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  int myRank;
  MPI_Comm_rank(commWorld, &myRank);

  std::tuple<MPI_Comm, int, int, int, int> commInfo = getCommunicatorSlice(commWorld);
  MPI_Comm sliceComm = std::get<0>(commInfo);
  int pGridCoordX = std::get<1>(commInfo);
  int pGridCoordY = std::get<2>(commInfo);
  int pGridCoordZ = std::get<3>(commInfo);
  int pGridDimensionSize = std::get<4>(commInfo);

  // Locally generate each matrix, then AllGather along the slice communicator. Buid the entire matrix. Only then can we feed into LAPACK/BLAS routines
  // Fast pass-by-value via modern C++ move semantics
  std::vector<T> matrixAforEngine = getReferenceMatrix(myMatrix, localDimensionX, localDimensionY, globalDimensionX, globalDimensionY, pGridCoordX*pGridDimensionSize+pGridCoordY, commInfo);
  std::vector<T> matrixBforEngine(globalDimensionZ*globalDimensionX);	// Instead of using C for output matrix, lets use B as we did in SquareMM3D

  // Assume column major and no transpose and that the matrix A is square.
  blasEngine<T,U>::_syrk(&matrixAforEngine[0], &matrixBforEngine[0], globalDimensionX, globalDimensionY,
    globalDimensionX, globalDimensionY, srcPackage);

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error = getResidualSquare(myMatrix.getVectorData(), matrixBforEngine, localDimensionX, localDimensionY, localDimensionZ, globalDimensionX,
    globalDimensionY, globalDimensionZ, commInfo);

  // Now, we need the AllReduce of the error. Very cheap operation in terms of bandwidth cost, since we are only communicating a single double primitive type.
  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, sliceComm);
  if (myRank == 0) {std::cout << "Total error = " << error << std::endl;}

  MPI_Comm_free(&sliceComm);
}
  
template<typename T, typename U, template<typename,typename> class blasEngine>
T MMvalidate<T,U,blasEngine>::getResidualSquare(
		     std::vector<T>& myValues,
		     std::vector<T>& blasValues,
		     U localDimensionX,
		     U localDimensionY,
		     U localDimensionZ,
		     U globalDimensionX,
		     U globalDimensionY,
	   	     U globalDimensionZ,
		     std::tuple<MPI_Comm, int, int, int, int> commInfo
		   )
{
  T error = 0;
  int pCoordX = std::get<1>(commInfo);
  int pCoordY = std::get<2>(commInfo);
  int pGridDimensionSize = std::get<4>(commInfo);
  U myIndex = 0;
  U solIndex = pCoordX *globalDimensionY + pCoordY;

  for (U i=0; i<localDimensionX; i++)
  {
    U saveCountRef = solIndex;
    for (U j=0; j<localDimensionY; j++)
    {
      T errorSquare = abs(myValues[myIndex] - blasValues[solIndex]);
      //std::cout << errorSquare << " " << myValues[myIndex] << " " << blasValues[solIndex] << std::endl;
      errorSquare *= errorSquare;
      error += errorSquare;
      solIndex += pGridDimensionSize;
      myIndex++;
    }
    solIndex = saveCountRef + pGridDimensionSize*globalDimensionY;
  }

  error = std::sqrt(error);
  return error;
}


template<typename T, typename U, template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
std::vector<T> MMvalidate<T,U,blasEngine>::getReferenceMatrix(
                        					Matrix<T,U,MatrixStructureSquare,Distribution>& myMatrix,
								U localNumColumns,
								U localNumRows,
								U globalNumColumns,
								U globalNumRows,
								U key,
								std::tuple<MPI_Comm, int, int, int, int> commInfo
							     )
{
  MPI_Comm sliceComm = std::get<0>(commInfo);
  int pGridCoordX = std::get<1>(commInfo);
  int pGridCoordY = std::get<2>(commInfo);
  int pGridCoordZ = std::get<3>(commInfo);
  int pGridDimensionSize = std::get<4>(commInfo);

  using MatrixType = Matrix<T,U,MatrixStructureSquare,Distribution>;
  MatrixType localMatrix(localNumColumns, localNumRows, globalNumColumns, globalNumRows);
  localMatrix.DistributeRandom(pGridCoordX, pGridCoordY, pGridDimensionSize, pGridDimensionSize, key);

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
