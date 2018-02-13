/* Author: Edward Hutter */

/* Validation against sequential BLAS/LAPACK constructs */
template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
void QRvalidate<T,U>::validateLocal1D(
                        Matrix<T,U,MatrixStructureRectangle,Distribution>& matrixA,
                        Matrix<T,U,MatrixStructureRectangle,Distribution>& myQ,
                        Matrix<T,U,MatrixStructureSquare,Distribution>& myR,
                        MPI_Comm commWorld
                      )
{
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
  std::vector<T> matrixQ = globalMatrixA;		// true copy
  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, globalDimensionM, globalDimensionN, &matrixQ[0], globalDimensionM, &tau[0]);
  LAPACKE_dorgqr(LAPACK_COL_MAJOR, globalDimensionM, globalDimensionN, globalDimensionN, &matrixQ[0],
    globalDimensionM, &tau[0]);

  // Q is in globalMatrixA now
  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error = getResidual1D_RowCyclic(myQ.getVectorData(), matrixQ, globalDimensionN, globalDimensionM, localDimensionM, commWorld);

  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, commWorld);
  if (myRank == 0) {std::cout << "Total error of myQ - solQ is " << error << std::endl;}

  // Now generate R using Q and A
  std::vector<T> matrixR(globalDimensionN*globalDimensionN,0);
  // For right now, I will just use cblas, but Note that I should template this class with blasEngine
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, globalDimensionN, globalDimensionN, globalDimensionM,
    1., &matrixQ[0], globalDimensionM, &globalMatrixA[0], globalDimensionM, 0., &matrixR[0], globalDimensionN);

  // Need to set up error2 for matrix R, but do that later
  T error2 = getResidual1D_Full(myR.getVectorData(), matrixR, globalDimensionN, globalDimensionM, commWorld);
  MPI_Allreduce(MPI_IN_PLACE, &error2, 1, MPI_DOUBLE, MPI_SUM, commWorld);
  if (myRank == 0) {std::cout << "Total error of myR - solR is " << error2 << std::endl;}

  // generate A_computed = myQ*myR and compare against original A
  T error3 = getResidual1D(matrixA.getVectorData(), myQ.getVectorData(), myR.getVectorData(), globalDimensionN, globalDimensionM, localDimensionM, commWorld);
  MPI_Allreduce(MPI_IN_PLACE, &error3, 1, MPI_DOUBLE, MPI_SUM, commWorld);
  if (myRank == 0) {std::cout << "Total residual error is " << error3 << std::endl;}

  T error4 = testOrthogonality1D(myQ.getVectorData(), globalDimensionN, globalDimensionM, localDimensionM, commWorld);
  MPI_Allreduce(MPI_IN_PLACE, &error4, 1, MPI_DOUBLE, MPI_SUM, commWorld);
  if (myRank == 0) {std::cout << "Deviation from orthogonality is " << error4 << std::endl;}

  return;
}


/* Validation against sequential BLAS/LAPACK constructs */
template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
void QRvalidate<T,U>::validateLocal3D(
                        Matrix<T,U,MatrixStructureRectangle,Distribution>& matrixA,
                        Matrix<T,U,MatrixStructureRectangle,Distribution>& myQ,
                        Matrix<T,U,MatrixStructureSquare,Distribution>& myR,
                        MPI_Comm commWorld
                      )
{
  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  int myRank,numPEs;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_rank(commWorld, &myRank);

  auto commInfo3D = getCommunicatorSlice(commWorld);

  // Simple asignments like these don't need pass-by-reference. Remember the new pass-by-value semantics are efficient anyways
  MPI_Comm sliceComm = std::get<0>(commInfo3D);
  int pCoordX = std::get<1>(commInfo3D);
  int pCoordY = std::get<2>(commInfo3D);
  int pCoordZ = std::get<3>(commInfo3D);
  int pGridDimensionSize = std::get<4>(commInfo3D);

  // Remember, we are assuming that the matrix is square here
  U globalDimensionM = matrixA.getNumRowsGlobal();
  U globalDimensionN = matrixA.getNumColumnsGlobal();
  U localDimensionM = matrixA.getNumRowsLocal();
  U localDimensionN = matrixA.getNumColumnsLocal();

  // generate A_computed = myQ*myR and compare against original A
  T error1 = getResidual3D(matrixA, myQ, myR, globalDimensionM, globalDimensionN, commWorld);
  MPI_Allreduce(MPI_IN_PLACE, &error1, 1, MPI_DOUBLE, MPI_SUM, sliceComm);
  if (myRank == 0) {std::cout << "Total residual error is " << error1 << std::endl;}

/*
  T error2 = testOrthogonality3D(myQ, globalDimensionM, globalDimensionN, commWorld);
  MPI_Allreduce(MPI_IN_PLACE, &error2, 1, MPI_DOUBLE, MPI_SUM, sliceComm);
  if (myRank == 0) {std::cout << "Deviation from orthogonality is " << error2 << std::endl;}
*/
  MPI_Comm_free(&sliceComm);
  return;
}


/* Validation against sequential BLAS/LAPACK constructs */
template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
void QRvalidate<T,U>::validateLocalTunable(
                        Matrix<T,U,MatrixStructureRectangle,Distribution>& matrixA,
                        Matrix<T,U,MatrixStructureRectangle,Distribution>& myQ,
                        Matrix<T,U,MatrixStructureSquare,Distribution>& myR,
                        int gridDimensionD,
                        int gridDimensionC,
                        MPI_Comm commWorld
                      )
{
  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  auto tunableCommunicators = getTunableCommunicators(commWorld, gridDimensionD, gridDimensionC);

  // Simple asignments like these don't need pass-by-reference. Remember the new pass-by-value semantics are efficient anyways
  MPI_Comm sliceComm = std::get<4>(tunableCommunicators);
  int myRank;
  MPI_Comm_rank(commWorld, &myRank);

  // generate A_computed = myQ*myR and compare against original A
  U globalDimensionM = matrixA.getNumRowsGlobal();
  U globalDimensionN = matrixA.getNumColumnsGlobal();
  T error1 = getResidualTunable(matrixA, myQ, myR, globalDimensionM, globalDimensionN, gridDimensionD, gridDimensionC, commWorld, tunableCommunicators);
  MPI_Allreduce(MPI_IN_PLACE, &error1, 1, MPI_DOUBLE, MPI_SUM, sliceComm);
  if (myRank == 0) {std::cout << "Total residual error is " << error1 << std::endl;}

  MPI_Comm_free(&std::get<0>(tunableCommunicators));
  MPI_Comm_free(&std::get<1>(tunableCommunicators));
  MPI_Comm_free(&std::get<2>(tunableCommunicators));
  MPI_Comm_free(&std::get<3>(tunableCommunicators));
  MPI_Comm_free(&std::get<4>(tunableCommunicators));
  return;
}


/* Used for comparing a matrix owned among processors in a 1D row-cyclic manner to a full matrix */
template<typename T, typename U>
T QRvalidate<T,U>::getResidual1D_RowCyclic(std::vector<T>& myMatrix, std::vector<T>& solutionMatrix, U globalDimensionX, U globalDimensionY, U localDimensionY, MPI_Comm commWorld)
{
  int numPEs, myRank;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_rank(commWorld, &myRank);

  T error = 0;
  for (U i=0; i<globalDimensionX; i++)
  {
    for (U j=0; j<localDimensionY; j++)
    {
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
T QRvalidate<T,U>::testOrthogonality1D(std::vector<T>& myQ, U globalDimensionX, U globalDimensionY, U localDimensionY, MPI_Comm commWorld)
{
  int numPEs, myRank;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_rank(commWorld, &myRank);

  // generate Q^T*Q and the compare against 0s and 1s, implicely forming the Identity matrix
  std::vector<T> myI(globalDimensionX*globalDimensionX,0);
  // Again, for now, lets just use cblas, but I can encapsulate it into blasEngine later
  cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, globalDimensionX, localDimensionY, 1., &myQ[0],
    localDimensionY, 0., &myI[0], globalDimensionX);

  // To complete the sum (rightward movement of Matvecs), perform an AllReduce
  MPI_Allreduce(MPI_IN_PLACE, &myI[0], globalDimensionX*globalDimensionX, MPI_DOUBLE, MPI_SUM, commWorld);

  T error = 0;
  for (U i=0; i<globalDimensionX; i++)
  {
    for (U j=0; j<globalDimensionX; j++)
    {
      U myIndex = i*globalDimensionX+j;
      T errorSquare = 0;
      // To avoid inner-loop if statements, I could separate out this inner loop, but its not necessary right now
      if (i==j)
      {
        errorSquare = std::abs(myI[myIndex] - 1.);
      }
      else
      {
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
T QRvalidate<T,U>::getResidual1D(std::vector<T>& myA, std::vector<T>& myQ, std::vector<T>&myR, U globalDimensionX, U globalDimensionY, U localDimensionY, MPI_Comm commWorld)
{
  int numPEs, myRank;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_rank(commWorld, &myRank);

  std::vector<T> computedA(globalDimensionX*localDimensionY,0);
  // Again, for now, lets just use cblas, but I can encapsulate it into blasEngine later
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, localDimensionY, globalDimensionX, globalDimensionX,
    1., &myQ[0], localDimensionY, &myR[0], globalDimensionX, 0., &computedA[0], localDimensionY);

  U trueDimensionY = globalDimensionY/numPEs + ((myRank < (globalDimensionY%numPEs)) ? 1 : 0);
  T error = 0;
  for (U i=0; i<globalDimensionX; i++)
  {
    for (U j=0; j<trueDimensionY; j++)
    {
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
T QRvalidate<T,U>::getResidual1D_Full(std::vector<T>& myMatrix, std::vector<T>& solutionMatrix, U globalDimensionX, U globalDimensionY, MPI_Comm commWorld)
{
  // Matrix R is owned by every processor
  int numPEs, myRank;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_rank(commWorld, &myRank);

  T error = 0;
  for (U i=0; i<globalDimensionX; i++)
  {
    for (U j=0; j<(i+1); j++)
    {
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


template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
T QRvalidate<T,U>::getResidual3D(Matrix<T,U,MatrixStructureRectangle,Distribution>& myA,
                         Matrix<T,U,MatrixStructureRectangle,Distribution>& myQ,
                         Matrix<T,U,MatrixStructureSquare,Distribution>& myR,
                         U globalDimensionM, U globalDimensionN, MPI_Comm commWorld)
{
  auto commInfo3D = getCommunicatorSlice(commWorld);

  // Simple asignments like these don't need pass-by-reference. Remember the new pass-by-value semantics are efficient anyways
  int pGridDimensionSize;
  MPI_Comm sliceComm = std::get<0>(commInfo3D);
  int pGridCoordX = std::get<1>(commInfo3D);
  int pGridCoordY = std::get<2>(commInfo3D);
  int pGridCoordZ = std::get<3>(commInfo3D);
  pGridDimensionSize = std::get<4>(commInfo3D);
  bool isRank1 = false;
  if ((pGridCoordY == 0) && (pGridCoordX == 0) && (pGridCoordZ == 0))
  {
    isRank1 = true;
  }

  U localDimensionM = myQ.getNumRowsLocal();
  U localDimensionN = myQ.getNumColumnsLocal();

  Matrix<T,U,MatrixStructureRectangle,Distribution> testA = myA;			// Just copy here. No big deal because it will be overwritten soon
  blasEngineArgumentPackage_gemm<double> blasArgs;
  blasArgs.order = blasEngineOrder::AblasColumnMajor;
  blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
  blasArgs.transposeB = blasEngineTranspose::AblasNoTrans;
  blasArgs.alpha = 1.;
  blasArgs.beta = 0.;
  MM3D<T,U,cblasEngine>::Multiply(myQ, myR, testA, commWorld, blasArgs, 0);

  // Now we can just iterate over myA and testA and compare
  T error = 0;
  U myIndex = 0;
  for (U i=0; i<localDimensionN; i++)
  {
    for (U j=0; j<localDimensionM; j++)
    {
      T errorSquare = 0;
      errorSquare = std::abs(myA.getRawData()[myIndex] - testA.getRawData()[myIndex]);
      //if (isRank1) std::cout << errorSquare << " " << myA.getRawData()[myIndex] << " " << testA.getRawData()[myIndex] << " " << i << " " << j << " " << myIndex << " " << std::endl;
      errorSquare *= errorSquare;
      error += errorSquare;
      myIndex++;
    }
  }

  error = std::sqrt(error);

  MPI_Comm_free(&sliceComm);
  return error;  
}


template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
T QRvalidate<T,U>::testOrthogonality3D(Matrix<T,U,MatrixStructureRectangle,Distribution>& myQ,
                               U globalDimensionM, U globalDimensionN, MPI_Comm commWorld)
{
  int myRank;
  MPI_Comm_rank(commWorld, &myRank);
  auto commInfo3D = getCommunicatorSlice(commWorld);

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
  MPI_Sendrecv_replace(&Qvector[0], localDimensionN*localDimensionM, MPI_DOUBLE, transposePartner, 0,
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


template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
T QRvalidate<T,U>::getResidualTunable(Matrix<T,U,MatrixStructureRectangle,Distribution>& myA,
                         Matrix<T,U,MatrixStructureRectangle,Distribution>& myQ,
                         Matrix<T,U,MatrixStructureSquare,Distribution>& myR,
                         U globalDimensionM, U globalDimensionN, int gridDimensionD, int gridDimensionC, MPI_Comm commWorld,
                         std::tuple<MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm> tunableCommunicators)
{
  MPI_Comm miniCubeComm = std::get<5>(tunableCommunicators);

  U localDimensionM = myA.getNumRowsLocal();//globalDimensionM/gridDimensionD;
  U localDimensionN = myA.getNumColumnsLocal();//globalDimensionN/gridDimensionC;

  Matrix<T,U,MatrixStructureRectangle,Distribution> testA = myA;			// Just copy here. No big deal because it will be overwritten soon
  blasEngineArgumentPackage_gemm<double> blasArgs;
  blasArgs.order = blasEngineOrder::AblasColumnMajor;
  blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
  blasArgs.transposeB = blasEngineTranspose::AblasNoTrans;
  blasArgs.alpha = 1.;
  blasArgs.beta = 0.;
  MM3D<T,U,cblasEngine>::Multiply(myQ, myR, testA, miniCubeComm, blasArgs, 0);

  // Now we can just iterate over myA and testA and compare
  T error = 0;
  U myIndex = 0;
  for (U i=0; i<localDimensionM; i++)
  {
    for (U j=0; j<localDimensionN; j++)
    {
      T errorSquare = 0;
      errorSquare = std::abs(myA.getRawData()[myIndex] - testA.getRawData()[myIndex]);
      errorSquare *= errorSquare;
      error += errorSquare;
      myIndex++;
    }
  }

  error = std::sqrt(error);
  return error;  
}

template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
std::vector<T> QRvalidate<T,U>::getReferenceMatrix1D(
                        				Matrix<T,U,MatrixStructureRectangle,Distribution>& myMatrix,
							U globalDimensionX,
							U globalDimensionY,
							U localDimensionY,
							U key,
							MPI_Comm commWorld
						  )
{
  int numPEs, myRank;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_rank(commWorld, &myRank);

  using MatrixType = Matrix<T,U,MatrixStructureRectangle,Distribution>;
  MatrixType localMatrix(globalDimensionX, globalDimensionY, 1, numPEs);
  localMatrix.DistributeRandom(0, myRank, 1, numPEs, key);

  U globalSize = globalDimensionX*localMatrix.getNumRowsLocal()*numPEs/*globalDimensionY*/;
  std::vector<T> blockedMatrix(globalSize);
  std::vector<T> cyclicMatrix(globalSize);
  U localSize = localMatrix.getNumRowsLocal()*localMatrix.getNumColumnsLocal();
  MPI_Allgather(localMatrix.getRawData(), localSize, MPI_DOUBLE, &blockedMatrix[0], localSize, MPI_DOUBLE, commWorld);

  U writeIndex = 0;
  // MACRO loop over all columns
  for (U i=0; i<globalDimensionX; i++)
  {
    // Inner loop over "block"s
    for (U j=0; j<localDimensionY; j++)
    {
      // Inner loop over all rows in a "block"
      for (U k=0; k<numPEs; k++)
      {
        U readIndex = i*localDimensionY + j + k*localSize;
        cyclicMatrix[writeIndex++] = blockedMatrix[readIndex];
      }
    }
  }

  return cyclicMatrix;
}
