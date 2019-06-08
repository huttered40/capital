/* Author: Edward Hutter */

template<typename MatrixAType, typename MatrixBType, typename MatrixCType>
void MMvalidate::validateLocal(MatrixAType& matrixA, MatrixBType& matrixB, MatrixCType& matrixC, MPI_Comm commWorld,
                               const blasEngineArgumentPackage_gemm<typename MatrixAType::ScalarType>& srcPackage){
  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  int myRank,sliceRank;
  MPI_Comm_rank(commWorld, &myRank);

  std::tuple<MPI_Comm,size_t,size_t,size_t,size_t> commInfo = util::getCommunicatorSlice(commWorld);
  MPI_Comm sliceComm = std::get<0>(commInfo);
  MPI_Comm_rank(sliceComm, &sliceRank);
  size_t pGridCoordX = std::get<1>(commInfo);
  size_t pGridCoordY = std::get<2>(commInfo);
  size_t pGridDimensionSize = std::get<4>(commInfo);

  // Locally generate each matrix, then AllGather along the slice communicator. Buid the entire matrix. Only then can we feed into LAPACK/BLAS routines
  // Fast pass-by-value via modern C++ move semantics
  U localDimensionM = matrixA.getNumRowsLocal();
  U localDimensionN = matrixB.getNumColumnsLocal();
  U globalDimensionM = matrixA.getNumRowsGlobal();
  U globalDimensionN = matrixB.getNumColumnsGlobal();
  U globalDimensionK = matrixA.getNumColumnsGlobal();
  std::vector<T> matrixAforEngine = util::getReferenceMatrix(matrixA, pGridCoordX*pGridDimensionSize+pGridCoordY, commInfo);
  std::vector<T> matrixBforEngine = util::getReferenceMatrix(matrixB, (pGridCoordX*pGridDimensionSize+pGridCoordY)*(-1), commInfo);
  // Note: If I am comparing with srcPackage.beta = 1, then this test should fail, since matrixC is started at 0.
  std::vector<T> matrixCforEngine(globalDimensionM*globalDimensionN, 0);	// No matrix needed for this. Only used in BLAS call

  // Assume column-major matrix and no transposes
  blasEngine::_gemm(&matrixAforEngine[0], &matrixBforEngine[0], &matrixCforEngine[0], globalDimensionM, globalDimensionN,
    globalDimensionK, globalDimensionM, globalDimensionK, globalDimensionM, srcPackage);

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error = getResidual(matrixC.getVectorData(), matrixCforEngine, localDimensionM, localDimensionN, globalDimensionM,globalDimensionN, commInfo);

  // Now, we need the AllReduce of the error. Very cheap operation in terms of bandwidth cost, since we are only communicating a single double primitive type.
  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, sliceComm);
  error = std::sqrt(error);
  if (sliceRank == 0) {std::cout << "Total error = " << error << std::endl;}

  MPI_Comm_free(&sliceComm);
}

template<typename MatrixAType, typename MatrixBinType, typename MatrixBoutType>
void MMvalidate::validateLocal(MatrixAType& matrixA, MatrixBinType& matrixBin, MatrixBoutType& matrixBout,
                               MPI_Comm commWorld, const blasEngineArgumentPackage_trmm<typename MatrixAType::ScalarType>& srcPackage){
  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  int myRank,sliceRank;
  MPI_Comm_rank(commWorld, &myRank);

  std::tuple<MPI_Comm,size_t,size_t,size_t,size_t> commInfo = util::getCommunicatorSlice(commWorld);
  MPI_Comm sliceComm = std::get<0>(commInfo);
  MPI_Comm_rank(sliceComm, &sliceRank);
  size_t pGridCoordX = std::get<1>(commInfo);
  size_t pGridCoordY = std::get<2>(commInfo);
  size_t pGridCoordZ = std::get<3>(commInfo);
  size_t pGridDimensionSize = std::get<4>(commInfo);

  U localDimensionM = matrixBin.getNumRowsLocal();
  U localDimensionN = matrixBin.getNumColumnsLocal();
  U globalDimensionM = matrixBin.getNumRowsGlobal();
  U globalDimensionN = matrixBin.getNumColumnsGlobal();
  // Locally generate each matrix, then AllGather along the slice communicator. Buid the entire matrix. Only then can we feed into LAPACK/BLAS routines
  // Fast pass-by-value via modern C++ move semantics
  U localTriDim = (srcPackage.side == blasEngineSide::AblasLeft ? localDimensionM : localDimensionN);
  U globalTriDim = (srcPackage.side == blasEngineSide::AblasLeft ? globalDimensionM : globalDimensionN);
  std::vector<T> matrixAforEngine = util::getReferenceMatrix(matrixA, pGridCoordX*pGridDimensionSize+pGridCoordY, commInfo);
  std::vector<T> matrixBforEngine = util::getReferenceMatrix(matrixBin, (pGridCoordX*pGridDimensionSize+pGridCoordY)*(-1), commInfo);

  blasEngine::_trmm(&matrixAforEngine[0], &matrixBforEngine[0], globalDimensionM, globalDimensionN,
    (srcPackage.side == blasEngineSide::AblasLeft ? globalDimensionM : globalDimensionN),
    (srcPackage.order == blasEngineOrder::AblasColumnMajor ? globalDimensionM : globalDimensionN), srcPackage);

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error = getResidual(matrixBout.getVectorData(), matrixBforEngine, localDimensionM, localDimensionN, globalDimensionM, globalDimensionN, commInfo);

  // Now, we need the AllReduce of the error. Very cheap operation in terms of bandwidth cost, since we are only communicating a single double primitive type.
  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, sliceComm);
  error = std::sqrt(error);
  if (sliceRank == 0) {std::cout << "Total error = " << error << std::endl;}

  MPI_Comm_free(&sliceComm);
}

  
template<typename T, typename U>
T MMvalidate::getResidual(std::vector<T>& myValues, std::vector<T>& blasValues,
                          U localDimensionM, U localDimensionN, U globalDimensionM, U globalDimensionN, std::tuple<MPI_Comm,size_t,size_t,size_t,size_t> commInfo){
  T error = 0;
  size_t pCoordX = std::get<1>(commInfo);
  size_t pCoordY = std::get<2>(commInfo);
  size_t pGridDimensionSize = std::get<4>(commInfo);
  U myIndex = 0;
  U solIndex = pCoordX *globalDimensionM + pCoordY;

  // We want to truncate this to use only the data that we own
  U trueDimensionM = globalDimensionM/pGridDimensionSize + ((pCoordY < (globalDimensionM%pGridDimensionSize)) ? 1 : 0);
  U trueDimensionN = globalDimensionN/pGridDimensionSize + ((pCoordX < (globalDimensionN%pGridDimensionSize)) ? 1 : 0);
  for (U i=0; i<trueDimensionN; i++){
    U saveCountRef = solIndex;
    U saveCountMy = myIndex;
    for (U j=0; j<trueDimensionM; j++){
      T errorSquare = std::abs(myValues[myIndex] - blasValues[solIndex]);
      //if ((pCoordX == 0) && (pCoordY == 0) && (pCoordZ == 0)) std::cout << errorSquare << " " << myValues[myIndex] << " " << blasValues[solIndex] << " at global index - " << solIndex << " And local index - " << myIndex << " and localDimensionM - " << localDimensionM << " And trueDimensionM - " << trueDimensionM << " And div - " << globalDimensionM/pGridDimensionSize << " and pCoordX - " << pCoordX << std::endl;
      errorSquare *= errorSquare;
      error += errorSquare;
      solIndex += pGridDimensionSize;
      myIndex++;
    }
    solIndex = saveCountRef + pGridDimensionSize*globalDimensionM;
    myIndex = saveCountMy + localDimensionM;
  }

  //error = std::sqrt(error);
  //std::cout << "Processor residual error - " << error << std::endl;
  return error;
}
