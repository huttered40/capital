/* Author: Edward Hutter */

namespace cholesky{
template<typename MatrixAType, typename MatrixSolType>
void CFvalidate::validateLocal(MatrixAType& matrixA, MatrixSolType& matrixSol, char dir, MPI_Comm commWorld){
  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixSolType::DimensionType;

  int myRank;
  MPI_Comm_rank(commWorld, &myRank);

  auto commInfo = util::getCommunicatorSlice(commWorld);
  MPI_Comm sliceComm = std::get<0>(commInfo);
  size_t pGridCoordX = std::get<1>(commInfo);
  size_t pGridCoordY = std::get<2>(commInfo);
  size_t pGridCoordZ = std::get<3>(commInfo);
  size_t pGridDimensionSize = std::get<4>(commInfo);

  U localDimension = matrixSol.getNumRowsLocal();
  U globalDimension = matrixSol.getNumRowsGlobal();
  std::vector<T> globalMatrixA = util::getReferenceMatrix(matrixA, pGridCoordX*pGridDimensionSize+pGridCoordY, commInfo);

  // for ease in finding Frobenius Norm
  for (U i=0; i<globalDimension; i++){
    for (U j=0; j<globalDimension; j++){
      if ((dir == 'L') && (i>j)) globalMatrixA[i*globalDimension+j] = 0;
      if ((dir == 'U') && (j>i)) globalMatrixA[i*globalDimension+j] = 0;
    }
  }

  if (dir == 'L'){
    lapackEngineArgumentPackage_potrf potrfArgs(blasEngineOrder::AblasColumnMajor, blasEngineUpLo::AblasLower);
    lapackEngine::_potrf(&globalMatrixA[0],globalDimension,globalDimension,potrfArgs);
  } else{
    lapackEngineArgumentPackage_potrf potrfArgs(blasEngineOrder::AblasColumnMajor, blasEngineUpLo::AblasUpper);
    lapackEngine::_potrf(&globalMatrixA[0],globalDimension,globalDimension,potrfArgs);
  }

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error = (dir == 'L' ? getResidualTriangleLower(matrixSol.getVectorData(), globalMatrixA, localDimension, globalDimension, commInfo)
              : getResidualTriangleUpper(matrixSol.getVectorData(), globalMatrixA, localDimension, globalDimension, commInfo));

  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DATATYPE, MPI_SUM, sliceComm);
  if (myRank == 0) {std::cout << "Total error = " << error << std::endl;}

// Forget testing the inverse.
/*
  myTimer.setStartTime();
  LAPACKE_dtrtri(LAPACK_COL_MAJOR, dir, 'N', globalDimension, &globalMatrixA[0], globalDimension);
  myTimer.setEndTime();
  myTimer.printParallelTime(1e-9, MPI_COMM_WORLD, "LAPACK Triangular Inverse (dtrtri)");

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error2 = (dir == 'L' ? getResidualTriangleLower(matrixSol_TI.getVectorData(), globalMatrixA, localDimension, globalDimension, commInfo)
               : getResidualTriangleUpper(matrixSol_TI.getVectorData(), globalMatrixA, localDimension, globalDimension, commInfo));

  // Now, we need the AllReduce of the error. Very cheap operation in terms of bandwidth cost, since we are only communicating a single double primitive type.
  MPI_Allreduce(MPI_IN_PLACE, &error2, 1, MPI_DATATYPE, MPI_SUM, sliceComm);
  if (myRank == 0) {std::cout << "Total error = " << error2 << std::endl;}
*/
  MPI_Comm_free(&sliceComm);
}

template<typename MatrixAType, typename MatrixTriType>
typename MatrixAType::ScalarType CFvalidate::validateParallel(MatrixAType& matrixA, MatrixTriType& matrixTri,
                               char dir, MPI_Comm commWorld, std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D){
  int rank,size;
  MPI_Comm_rank(commWorld, &rank);
  MPI_Comm_size(commWorld, &size);

  auto commInfo = util::getCommunicatorSlice(commWorld);
  size_t pGridCoordX = std::get<1>(commInfo);
  size_t pGridCoordY = std::get<2>(commInfo);
  size_t pGridCoordZ = std::get<3>(commInfo);
  size_t pGridDimensionSize = std::get<4>(commInfo);
  size_t helper = pGridDimensionSize;
  helper *= helper;

  size_t transposePartner = pGridCoordX*helper + pGridCoordY*pGridDimensionSize + pGridCoordZ;
  util::removeTriangle(matrixTri, pGridCoordX, pGridCoordY, pGridDimensionSize, dir);
  MatrixTriType matrixTriTrans = matrixTri;
  util::transposeSwap(matrixTriTrans, rank, transposePartner, MPI_COMM_WORLD);
  std::string str = "Residual: ";
  return validator::validateResidualParallel((dir == 'L' ? matrixTri : matrixTriTrans), (dir == 'L' ? matrixTriTrans : matrixTri), matrixA, dir, commWorld, commInfo3D, MPI_COMM_WORLD, str);
}

// We only test the lower triangular for now. The matrices are stored with square structure though.
template<typename T, typename U>
T CFvalidate::getResidualTriangleLower(std::vector<T>& myValues, std::vector<T>& lapackValues, U localDimension, U globalDimension, std::tuple<MPI_Comm,size_t,size_t,size_t,size_t> commInfo){
  T error = 0;
  size_t pCoordX = std::get<1>(commInfo);
  size_t pCoordY = std::get<2>(commInfo);
  size_t pCoordZ = std::get<3>(commInfo);
  bool isRank1 = false;
  if ((pCoordY == 0) && (pCoordX == 0) && (pCoordZ == 0)){
    isRank1 = true;
  }
  size_t pGridDimensionSize = std::get<4>(commInfo);

  U myIndex = 0;
  U solIndex = pCoordY + pCoordX*globalDimension;
  // We want to truncate this to use only the data that we own
  U trueDimensionM = globalDimension/pGridDimensionSize + ((pCoordY < (globalDimension%pGridDimensionSize)) ? 1 : 0);
  U trueDimensionN = globalDimension/pGridDimensionSize + ((pCoordX < (globalDimension%pGridDimensionSize)) ? 1 : 0);

  for (U i=0; i<trueDimensionN; i++){
    U saveCountRef = solIndex;
    U saveCountMy = myIndex;
    for (U j=0; j<trueDimensionM; j++){
      if (i>j) {solIndex+=pGridDimensionSize; myIndex++; continue;}
      T errorSquare = std::abs(myValues[myIndex] - lapackValues[solIndex]);
      //if (isRank1) std::cout << errorSquare << " " << myValues[myIndex] << " " << lapackValues[solIndex] << " " << i << " " << j << " " << myIndex << " " << std::endl;
      errorSquare *= errorSquare;
      error += errorSquare;
      solIndex += pGridDimensionSize;
      myIndex++;
    }
    solIndex = saveCountRef + pGridDimensionSize*globalDimension;
    myIndex = saveCountMy + localDimension;
  }

  error = std::sqrt(error);
  //if (isRank1) std::cout << "Total error - " << error << "\n\n\n";
  return error;		// return 2-norm
}

// We only test the lower triangular for now. The matrices are stored with square structure though.
template<typename T, typename U>
T CFvalidate::getResidualTriangleUpper(std::vector<T>& myValues, std::vector<T>& lapackValues, U localDimension, U globalDimension, std::tuple<MPI_Comm,size_t,size_t,size_t,size_t> commInfo){
  T error = 0;
  size_t pCoordX = std::get<1>(commInfo);
  size_t pCoordY = std::get<2>(commInfo);
  size_t pCoordZ = std::get<3>(commInfo);
  bool isRank1 = false;
  if ((pCoordY == 0) && (pCoordX == 0) && (pCoordZ == 0)){
    isRank1 = true;
  }

  size_t pGridDimensionSize = std::get<4>(commInfo);
  U myIndex = 0;
  U solIndex = pCoordX*globalDimension + pCoordY;

  // We want to truncate this to use only the data that we own
  U trueDimensionM = globalDimension/pGridDimensionSize + ((pCoordY < (globalDimension%pGridDimensionSize)) ? 1 : 0);
  U trueDimensionN = globalDimension/pGridDimensionSize + ((pCoordX < (globalDimension%pGridDimensionSize)) ? 1 : 0);
  for (U i=0; i<trueDimensionN; i++){
    U saveCountRef = solIndex;
    U saveCountMy = myIndex;
    for (U j=0; j<trueDimensionM; j++){
      if (i<j) {solIndex+=pGridDimensionSize; myIndex++; continue;}
      T errorSquare = std::abs(myValues[myIndex] - lapackValues[solIndex]);
      //if (isRank1) std::cout << errorSquare << " " << myValues[myIndex] << " " << lapackValues[solIndex] << " i - " << i << ", j - " << j << std::endl;
      errorSquare *= errorSquare;
      error += errorSquare;
      solIndex += pGridDimensionSize;
      myIndex++;
    }
    solIndex = saveCountRef + pGridDimensionSize*globalDimension;
    myIndex = saveCountMy + localDimension;
  }

  error = std::sqrt(error);
  //if (isRank1) std::cout << "Total error - " << error << "\n\n\n";
  return error;		// return 2-norm
}
}
