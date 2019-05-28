/* Author: Edward Hutter */

/* Validation against sequential BLAS/LAPACK constructs */
template<typename MatrixAType, typename MatrixQType, typename MatrixRType>
void QRvalidate::validateLocal1D(MatrixAType& matrixA, MatrixQType& matrixQ, MatrixRType& matrixR, MPI_Comm commWorld){
  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  int myRank,numPEs;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_rank(commWorld, &myRank);

  U globalDimensionM = matrixA.getNumRowsGlobal();
  U globalDimensionN = matrixA.getNumColumnsGlobal();
  U localDimensionM = matrixA.getNumRowsLocal();

  std::vector<T> globalMatrixA = getReferenceMatrix1D(matrixA, globalDimensionN, globalDimensionM, localDimensionM, myRank, commWorld);
  // Assume row-major
  std::vector<T> tau(globalDimensionN);
  std::vector<T> matQ = globalMatrixA;		// true copy

  lapackEngineArgumentPackage_geqrf geqrfArgs(blasEngineOrder::AblasColumnMajor);
  lapackEngineArgumentPackage_orgqr orgqrArgs(blasEngineOrder::AblasColumnMajor);
  lapackEngine::_geqrf(&matQ[0], &tau[0], globalDimensionM, globalDimensionN, globalDimensionM, geqrfArgs);
  lapackEngine::_orgqr(&matQ[0], &tau[0], globalDimensionM, globalDimensionN, globalDimensionN, globalDimensionM, orgqrArgs);

  // Q is in globalMatrixA now
  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error = getResidual1D_RowCyclic(matrixQ.getVectorData(), matQ, globalDimensionN, globalDimensionM, localDimensionM, commWorld);

  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DATATYPE, MPI_SUM, commWorld);
  if (myRank == 0) {std::cout << "Total error of myQ - solQ is " << error << std::endl;}

  // Now generate R using Q and A
  std::vector<T> matR(globalDimensionN*globalDimensionN,0);
  blasEngineArgumentPackage_gemm<T> gemmArgs(blasEngineOrder::AblasColumnMajor, blasEngineTranspose::AblasTrans, blasEngineTranspose::AblasNoTrans, 1., 0.);
  blasEngine::_gemm(&matQ[0], &globalMatrixA[0], &matR[0], globalDimensionN, globalDimensionN, globalDimensionM, globalDimensionM, globalDimensionM, globalDimensionN, gemmArgs);

  // Need to set up error2 for matrix R, but do that later
  T error2 = getResidual1D_Full(matrixR.getVectorData(), matR, globalDimensionN, globalDimensionM, commWorld);
  MPI_Allreduce(MPI_IN_PLACE, &error2, 1, MPI_DATATYPE, MPI_SUM, commWorld);
  if (myRank == 0) {std::cout << "Total error of myR - solR is " << error2 << std::endl;}

  // generate A_computed = matrixQ*matrixR and compare against original A
  T error3 = getResidual1D(matrixA.getVectorData(), matrixQ.getVectorData(), matrixR.getVectorData(), globalDimensionN, globalDimensionM, localDimensionM, commWorld);
  MPI_Allreduce(MPI_IN_PLACE, &error3, 1, MPI_DATATYPE, MPI_SUM, commWorld);
  if (myRank == 0) {std::cout << "Total residual error is " << error3 << std::endl;}

  T error4 = testOrthogonality1D(matrixQ.getVectorData(), globalDimensionN, globalDimensionM, localDimensionM, commWorld);
  MPI_Allreduce(MPI_IN_PLACE, &error4, 1, MPI_DATATYPE, MPI_SUM, commWorld);
  if (myRank == 0) {std::cout << "Deviation from orthogonality is " << error4 << std::endl;}

  return;
}


/* Validation against sequential BLAS/LAPACK constructs */
template<typename MatrixAType, typename MatrixQType, typename MatrixRType>
std::pair<typename MatrixAType::ScalarType,typename MatrixAType::ScalarType>
QRvalidate::validateParallel3D(MatrixAType& matrixA, MatrixQType& matrixQ, MatrixRType& matrixR, MPI_Comm commWorld,
                               std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D){
  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  // generate A_computed = matrixQ*matrixR and compare against original A
  int size; MPI_Comm_size(commWorld, &size);
  size_t pGridDimensionSize = std::nearbyint(std::pow(size,1./3.));
  util::removeTriangle(matrixR, std::get<4>(commInfo3D), std::get<5>(commInfo3D), pGridDimensionSize, 'U');
  std::string str1 = "Residual: ";
  T error1 = validator::validateResidualParallel(matrixQ, matrixR, matrixA, 'F', commWorld, commInfo3D, str1);
  std::string str2 = "Deviation from orthogonality: ";
  T error2 = validator::validateOrthogonalityParallel(matrixQ, commWorld, commInfo3D, str2);
  return std::make_pair(error1,error2);
}


/* Validation against sequential BLAS/LAPACK constructs */
template<typename MatrixAType, typename MatrixQType, typename MatrixRType>
std::pair<typename MatrixAType::ScalarType,typename MatrixAType::ScalarType>
QRvalidate::validateParallelTunable(MatrixAType& matrixA, MatrixQType& matrixQ, MatrixRType& matrixR, size_t gridDimensionD, size_t gridDimensionC, MPI_Comm commWorld,
                                    std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm>& commInfoTunable){
  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  MPI_Comm miniCubeComm = std::get<5>(commInfoTunable);
  auto commInfo3D = util::build3DTopology(miniCubeComm);
  MPI_Comm columnAltComm = std::get<2>(commInfoTunable);
  int size; MPI_Comm_size(miniCubeComm, &size);
  size_t pGridDimensionSize = std::nearbyint(std::pow(size,1./3.));
  util::removeTriangle(matrixR, std::get<4>(commInfo3D), std::get<5>(commInfo3D), pGridDimensionSize, 'U');
  std::string str1 = "Residual: ";
  T error1 = validator::validateResidualParallel(matrixQ, matrixR, matrixA, 'F', miniCubeComm, commInfo3D, MPI_COMM_WORLD, str1);
  std::string str2 = "Deviation from orthogonality: ";
  T error2 = validator::validateOrthogonalityParallel(matrixQ, miniCubeComm, commInfo3D, columnAltComm, str2);
  util::destroy3DTopology(commInfo3D);
  return std::make_pair(error1,error2);
}


/* Used for comparing a matrix owned among processors in a 1D row-cyclic manner to a full matrix */
template<typename T, typename U>
T QRvalidate::getResidual1D_RowCyclic(std::vector<T>& myMatrix, std::vector<T>& solutionMatrix, U globalDimensionX, U globalDimensionY, U localDimensionY, MPI_Comm commWorld){
  int numPEs, myRank;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_rank(commWorld, &myRank);

  T error = 0;
  for (U i=0; i<globalDimensionX; i++){
    for (U j=0; j<localDimensionY; j++){
      U myIndex = i*localDimensionY+j;
      U solIndex = i*globalDimensionY+(j*numPEs+myRank);
/*
      if (std::abs(myMatrix[myIndex] + solutionMatrix[solIndex]) <= 1e-12)
      {
        T errorSquare = std::abs(myMatrix[myIndex] + solutionMatrix[solIndex]);
        errorSquare *= errorSquare;
        //error += errorSquare;
        continue;
      }
*/
      T errorSquare = std::abs(myMatrix[myIndex] - solutionMatrix[solIndex]);
      //if (myRank==0) std::cout << errorSquare << " " << myMatrix[myIndex] << " " << solutionMatrix[solIndex] << " i - " << i << ", j - " << j << std::endl;
      errorSquare *= errorSquare;
      error += errorSquare;
    }
  }

  error = std::sqrt(error);
  return error;
}

template<typename T, typename U>
T QRvalidate::testOrthogonality1D(std::vector<T>& myQ, U globalDimensionX, U globalDimensionY, U localDimensionY, MPI_Comm commWorld){
  int numPEs, myRank;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_rank(commWorld, &myRank);

  // generate Q^T*Q and the compare against 0s and 1s, implicely forming the Identity matrix
  std::vector<T> myI(globalDimensionX*globalDimensionX,0);
  blasEngineArgumentPackage_syrk<T> syrkArgs(blasEngineOrder::AblasColumnMajor, blasEngineUpLo::AblasUpper, blasEngineTranspose::AblasTrans, 1., 0.);
  blasEngine::_gemm(&myQ[0], &myI[0], globalDimensionX, localDimensionY, localDimensionY, globalDimensionX, syrkArgs);

  // To complete the sum (rightward movement of Matvecs), perform an AllReduce
  MPI_Allreduce(MPI_IN_PLACE, &myI[0], globalDimensionX*globalDimensionX, MPI_DATATYPE, MPI_SUM, commWorld);

  T error = 0;
  for (U i=0; i<globalDimensionX; i++){
    for (U j=0; j<globalDimensionX; j++){
      U myIndex = i*globalDimensionX+j;
      T errorSquare = 0;
      // To avoid inner-loop if statements, I could separate out this inner loop, but its not necessary right now
      if (i==j){
        errorSquare = std::abs(myI[myIndex] - 1.);
      }
      else{
        errorSquare = std::abs(myI[myIndex] - 0.);
      }
      errorSquare *= errorSquare;
      error += errorSquare;
    }
  }

  error = std::sqrt(error);
  return error;
}


// generate A_computed = myQ*myR and compare against original A
template<typename T, typename U>
T QRvalidate::getResidual1D(std::vector<T>& myA, std::vector<T>& myQ, std::vector<T>&myR, U globalDimensionX, U globalDimensionY, U localDimensionY, MPI_Comm commWorld){
  int numPEs, myRank;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_rank(commWorld, &myRank);

  std::vector<T> computedA(globalDimensionX*localDimensionY,0);
  blasEngineArgumentPackage_gemm<T> gemmArgs(blasEngineOrder::AblasColumnMajor, blasEngineTranspose::AblasNoTrans, blasEngineTranspose::AblasNoTrans, 1., 0.);
  blasEngine::_gemm(&myQ[0], &myR[0], &computedA[0], localDimensionY, globalDimensionX, globalDimensionX, localDimensionY, globalDimensionX, localDimensionY, gemmArgs);

  U trueDimensionY = globalDimensionY/numPEs + ((myRank < (globalDimensionY%numPEs)) ? 1 : 0);
  T error = 0;
  for (U i=0; i<globalDimensionX; i++){
    for (U j=0; j<trueDimensionY; j++){
      U myIndex = i*localDimensionY+j;
      T errorSquare = std::abs(myA[myIndex] - computedA[myIndex]);
      //`if (myRank==0) std::cout << errorSquare << " " << myA[myIndex] << " " << computedA[myIndex] << " i - " << i << ", j - " << j << std::endl;
      errorSquare *= errorSquare;
      error += errorSquare;
    }
  }

  error = std::sqrt(error);
  return error;
}


/* Used for comparing a full matrix to a full matrix */
template<typename T, typename U>
T QRvalidate::getResidual1D_Full(std::vector<T>& myMatrix, std::vector<T>& solutionMatrix, U globalDimensionX, U globalDimensionY, MPI_Comm commWorld){
  // Matrix R is owned by every processor
  int numPEs, myRank;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_rank(commWorld, &myRank);

  T error = 0;
  for (U i=0; i<globalDimensionX; i++){
    for (U j=0; j<(i+1); j++){
      U myIndex = i*globalDimensionX+j;
/*
      if (std::abs(myMatrix[myIndex] + solutionMatrix[myIndex]) <= 1e-12)
      {
        T errorSquare = std::abs(myMatrix[myIndex] + solutionMatrix[myIndex]);
        errorSquare *= errorSquare;
        error += errorSquare;
        continue;
      }
*/
      T errorSquare = std::abs(myMatrix[myIndex] - solutionMatrix[myIndex]);
      //if (myRank==0) std::cout << errorSquare << " " << myMatrix[myIndex] << " " << solutionMatrix[myIndex] << " i - " << i << ", j - " << j << std::endl;
      errorSquare *= errorSquare;
      error += errorSquare;
    }
  }

  error = std::sqrt(error);
  return error;
}

/*
template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
T QRvalidate<T,U>::testOrthogonality3D(Matrix<T,U,MatrixStructureRectangle,Distribution>& myQ,
                               U globalDimensionM, U globalDimensionN, MPI_Comm commWorld){
  int myRank;
  MPI_Comm_rank(commWorld, &myRank);
  auto commInfo3D = util::getCommunicatorSlice(commWorld);

  int pGridDimensionSize;
  MPI_Comm sliceComm = std::get<0>(commInfo3D);
  int pGridCoordX = std::get<1>(commInfo3D);
  int pGridCoordY = std::get<2>(commInfo3D);
  int pGridCoordZ = std::get<3>(commInfo3D);
  pGridDimensionSize = std::get<4>(commInfo3D);

  U localDimensionM = globalDimensionM/pGridDimensionSize;
  U localDimensionN = globalDimensionN/pGridDimensionSize;
  // Need to multiply Q^T by Q, then we get back the result, and we compare against an implicit Identity matrix
  
  // First, I need to reform these matrices in order to call MM3D
  Matrix<T,U,MatrixStructureSquare,Distribution> myI(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN,
    globalDimensionN, globalDimensionN, true);

  blasEngineArgumentPackage_gemm<double> blasArgs;
  blasArgs.order = blasEngineOrder::AblasColumnMajor;
  blasArgs.transposeA = blasEngineTranspose::AblasTrans;
  blasArgs.transposeB = blasEngineTranspose::AblasNoTrans;
  blasArgs.alpha = 1.;
  blasArgs.beta = 0.;

  // First, we need an explicit transpose
  int helper = pGridDimensionSize*pGridDimensionSize;
  int transposePartner = pGridCoordZ*helper + pGridCoordX*pGridDimensionSize + pGridCoordY;
  std::vector<T> Qvector = myQ.getVectorData();							// This is a straight copy, because we can't corrupt Q's data
  //Matrix<T,U,MatrixStructureRectangle,Distribution> QT = myQ;
  MPI_Sendrecv_replace(&Qvector[0], localDimensionN*localDimensionM, MPI_DATATYPE, transposePartner, 0,
    transposePartner, 0, commWorld, MPI_STATUS_IGNORE);

  Matrix<T,U,MatrixStructureRectangle,Distribution> QT(std::move(Qvector), localDimensionM, localDimensionN, globalDimensionM, globalDimensionN, true);
  MM3D<T,U,cblasEngine>::Multiply(QT, myQ, myI, commWorld, blasArgs, 0);

  //if (myRank == 0) myQ.print();

  T error = 0;
  U myIndex = 0;
  U implicitIndexX = pGridCoordX;
  U implicitIndexY = pGridCoordY;

  for (U i=0; i<localDimensionN; i++)
  {
    U saveCountRef = implicitIndexY;
    for (U j=0; j<localDimensionN; j++)
    {
      T errorSquare = 0;
      if (implicitIndexX == implicitIndexY)
      {
        errorSquare = std::abs(myI.getRawData()[myIndex] - 1);
        //if (myRank == 6) {std::cout << myI.getRawData()[myIndex] << " " << 1 << std::endl;}
      }
      else
      {
        errorSquare = std::abs(myI.getRawData()[myIndex] - 0);
        //if (myRank == 6) {std::cout << myI.getRawData()[myIndex] << " " << 0 << std::endl;}
      }
      errorSquare *= errorSquare;
      error += errorSquare;
      //if (myRank == 6) {std::cout << "current error = " << error << std::endl;}
      implicitIndexY += pGridDimensionSize;
      myIndex++;
    }
    implicitIndexY = saveCountRef;
    implicitIndexX += pGridDimensionSize;
  }

  error = std::sqrt(error);
  //std::cout << myRank << " has local error - " << error << std::endl;

  MPI_Comm_free(&sliceComm);
  return error;		// return 2-norm
}
*/

template<typename MatrixType>
std::vector<typename MatrixType::ScalarType>
QRvalidate::getReferenceMatrix1D(MatrixType& myMatrix, typename MatrixType::DimensionType globalDimensionX, typename MatrixType::DimensionType globalDimensionY,
                                 typename MatrixType::DimensionType localDimensionY, size_t key, MPI_Comm commWorld){

  using T = typename MatrixType::ScalarType;
  using U = typename MatrixType::DimensionType;

  int numPEs, myRank;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_rank(commWorld, &myRank);

  MatrixType localMatrix(globalDimensionX, globalDimensionY, 1, numPEs);
  localMatrix.DistributeRandom(0, myRank, 1, numPEs, key);

  U globalSize = globalDimensionX*localMatrix.getNumRowsLocal()*numPEs/*globalDimensionY*/;
  std::vector<T> blockedMatrix(globalSize);
  std::vector<T> cyclicMatrix(globalSize);
  U localSize = localMatrix.getNumRowsLocal()*localMatrix.getNumColumnsLocal();
  MPI_Allgather(localMatrix.getRawData(), localSize, MPI_DATATYPE, &blockedMatrix[0], localSize, MPI_DATATYPE, commWorld);

  U writeIndex = 0;
  // MACRO loop over all columns
  for (U i=0; i<globalDimensionX; i++){
    // Inner loop over "block"s
    for (U j=0; j<localDimensionY; j++){
      // Inner loop over all rows in a "block"
      for (size_t k=0; k<numPEs; k++){
        U readIndex = i*localDimensionY + j + k*localSize;
        cyclicMatrix[writeIndex++] = blockedMatrix[readIndex];
      }
    }
  }

  return cyclicMatrix;
}
