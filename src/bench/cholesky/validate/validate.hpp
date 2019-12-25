/* Author: Edward Hutter */

namespace cholesky{

template<typename AlgType>
template<typename MatrixAType, typename MatrixTriType, typename CommType>
typename MatrixAType::ScalarType validate<AlgType>::invoke(MatrixAType& A, MatrixTriType& Tri, char dir, CommType&& CommInfo){

  using T = typename MatrixAType::ScalarType;

  util::remove_triangle(Tri, CommInfo.x, CommInfo.y, CommInfo.d, dir);
  MatrixTriType TriTrans = Tri;
  util::transpose(TriTrans, std::forward<CommType>(CommInfo));
  MatrixAType saveMatA = A;

  if (dir == 'L'){
    blas::ArgPack_gemm<T> blasArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasNoTrans, blas::Transpose::AblasTrans, 1., -1.);
    matmult::summa::invoke(Tri, TriTrans, A, std::forward<CommType>(CommInfo), blasArgs);
    auto Lambda = [](auto&& matrix, auto&& ref, size_t index, size_t sliceX, size_t sliceY){
      using T = typename std::remove_reference_t<decltype(matrix)>::ScalarType;
      T val=0;
      T control=0;
      if (sliceY >= sliceX){
        val = matrix.data()[index];
        control = ref.data()[index];
      }
      return std::make_pair(val,control);
    };
    return util::residual_local(A, saveMatA, std::move(Lambda), CommInfo.slice, CommInfo.x, CommInfo.y, CommInfo.d, CommInfo.d);
  }
  else if (dir == 'U'){
    blas::ArgPack_gemm<T> blasArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasTrans, blas::Transpose::AblasNoTrans, 1., -1.);
    matmult::summa::invoke(TriTrans, Tri, A, std::forward<CommType>(CommInfo), blasArgs);
    auto Lambda = [](auto&& matrix, auto&& ref, size_t index, size_t sliceX, size_t sliceY){
      using T = typename std::remove_reference_t<decltype(matrix)>::ScalarType;
      T val=0;
      T control=0;
      if (sliceY <= sliceX){
        val = matrix.data()[index];
        control = ref.data()[index];
      }
      return std::make_pair(val,control);
    };
    return util::residual_local(A, saveMatA, std::move(Lambda), CommInfo.slice, CommInfo.x, CommInfo.y, CommInfo.d, CommInfo.d);
  }
  return 0.;	// prevent compiler complaints
}

/*
template<typename AlgType>
template<typename MatrixAType, typename MatrixSolType>
void validate<AlgType>::validateLocal(MatrixAType& matrixA, MatrixSolType& matrixSol, char dir, MPI_Comm commWorld){
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
//
//  myTimer.setStartTime();
//  LAPACKE_dtrtri(LAPACK_COL_MAJOR, dir, 'N', globalDimension, &globalMatrixA[0], globalDimension);
//  myTimer.setEndTime();
//  myTimer.printParallelTime(1e-9, MPI_COMM_WORLD, "LAPACK Triangular Inverse (dtrtri)");
//
//  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
//  T error2 = (dir == 'L' ? getResidualTriangleLower(matrixSol_TI.getVectorData(), globalMatrixA, localDimension, globalDimension, commInfo)
//               : getResidualTriangleUpper(matrixSol_TI.getVectorData(), globalMatrixA, localDimension, globalDimension, commInfo));
//
//  // Now, we need the AllReduce of the error. Very cheap operation in terms of bandwidth cost, since we are only communicating a single double primitive type.
//  MPI_Allreduce(MPI_IN_PLACE, &error2, 1, MPI_DATATYPE, MPI_SUM, sliceComm);
//  if (myRank == 0) {std::cout << "Total error = " << error2 << std::endl;}

  MPI_Comm_free(&sliceComm);
}


// We only test the lower triangular for now. The matrices are stored with square structure though.
template<typename AlgType>
template<typename T, typename U, typename CommType>
T validate<AlgType>::getResidualTriangleLower(std::vector<T>& myValues, std::vector<T>& lapackValues, U localDimension, U globalDimension, CommType&& CommInfo){
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
template<typename AlgType>
template<typename T, typename U, typename CommType>
T validate<AlgType>::getResidualTriangleUpper(std::vector<T>& myValues, std::vector<T>& lapackValues, U localDimension, U globalDimension, CommType&& CommInfo){
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
*/
}
