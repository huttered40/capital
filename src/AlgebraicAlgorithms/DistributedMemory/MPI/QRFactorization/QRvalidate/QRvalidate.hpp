/* Author: Edward Hutter */

/*  Already defined in MMvalidate
static std::tuple<MPI_Comm, int, int, int, int> getCommunicatorSlice(MPI_Comm commWorld)
{
  int rank,size;
  MPI_Comm_rank(commWorld, &rank);
  MPI_Comm_size(commWorld, &size);

  int pGridDimensionSize = ceil(pow(size,1./3.));
  
  int helper = pGridDimensionSize;
  helper *= helper;
  int pCoordX = rank%pGridDimensionSize;
  int pCoordY = (rank%helper)/pGridDimensionSize;
  int pCoordZ = rank/helper;

  MPI_Comm sliceComm;
  MPI_Comm_split(commWorld, pCoordZ, rank, &sliceComm);
  return std::make_tuple(sliceComm, pCoordX, pCoordY, pCoordZ, pGridDimensionSize); 
}
*/


/* Validation against sequential BLAS/LAPACK constructs */
template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
void QRvalidate<T,U>::validateLocal1D(
                        Matrix<T,U,MatrixStructureRectangle,Distribution>& matrixA,
                        Matrix<T,U,MatrixStructureRectangle,Distribution>& myQ,
                        Matrix<T,U,MatrixStructureSquare,Distribution>& myR,
                        U globalDimensionX,
                        U globalDimensionY,
                        MPI_Comm commWorld
                      )
{
  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  int myRank,numPEs;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_rank(commWorld, &myRank);
  U localDimensionY = globalDimensionY/numPEs;

  std::vector<T> globalMatrixA = getReferenceMatrix1D(matrixA, globalDimensionX, globalDimensionY, localDimensionY, myRank, commWorld);

  // Assume row-major
  std::vector<T> tau(globalDimensionX);
  std::vector<T> matrixQ = globalMatrixA;		// true copy
  LAPACKE_dgeqrf(LAPACK_COL_MAJOR, globalDimensionY, globalDimensionX, &matrixQ[0], globalDimensionY, &tau[0]);
  LAPACKE_dorgqr(LAPACK_COL_MAJOR, globalDimensionY, globalDimensionX, globalDimensionX, &matrixQ[0],
    globalDimensionY, &tau[0]);

  // Q is in globalMatrixA now

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error = getResidual1D_RowCyclic(myQ.getVectorData(), matrixQ, globalDimensionX, globalDimensionY, commWorld);

  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, commWorld);
  if (myRank == 0) {std::cout << "Total error of myQ - solQ is " << error << std::endl;}

  // Now generate R using Q and A
  std::vector<T> matrixR(globalDimensionX*globalDimensionX,0);
  // For right now, I will just use cblas, but Note that I should template this class with blasEngine
  cblas_dgemm(CblasColMajor, CblasTrans, CblasNoTrans, globalDimensionX, globalDimensionX, globalDimensionY,
    1., &matrixQ[0], globalDimensionY, &globalMatrixA[0], globalDimensionY, 0., &matrixR[0], globalDimensionX);

  // Need to set up error2 for matrix R, but do that later
  T error2 = getResidual1D_Full(myR.getVectorData(), matrixR, globalDimensionX, globalDimensionY, commWorld);
  MPI_Allreduce(MPI_IN_PLACE, &error2, 1, MPI_DOUBLE, MPI_SUM, commWorld);
  if (myRank == 0) {std::cout << "Total error of myR - solR is " << error2 << std::endl;}

  // generate A_computed = myQ*myR and compare against original A
  T error3 = getResidual1D(matrixA.getVectorData(), myQ.getVectorData(), myR.getVectorData(), globalDimensionX, globalDimensionY, commWorld);
  MPI_Allreduce(MPI_IN_PLACE, &error3, 1, MPI_DOUBLE, MPI_SUM, commWorld);
  if (myRank == 0) {std::cout << "Total residual error is " << error3 << std::endl;}

  T error4 = testOrthogonality1D(myQ.getVectorData(), globalDimensionX, globalDimensionY, commWorld);
  MPI_Allreduce(MPI_IN_PLACE, &error4, 1, MPI_DOUBLE, MPI_SUM, commWorld);
  if (myRank == 0) {std::cout << "Deviation from orthogonality is " << error4 << std::endl;}

  return;
}


/* Validation against sequential BLAS/LAPACK constructs */
template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
void QRvalidate<T,U>::validateLocal3D(
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
                        Matrix<T,U,MatrixStructureSquare,Distribution>& myQ,
                        Matrix<T,U,MatrixStructureSquare,Distribution>& myR,
                        U globalDimensionX,
                        U globalDimensionY,
                        MPI_Comm commWorld
                      )
{
  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  int myRank,numPEs;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_rank(commWorld, &myRank);

  auto commInfo3D = setUpCommunicators(commWorld);

  // Simple asignments like these don't need pass-by-reference. Remember the new pass-by-value semantics are efficient anyways
  int pGridDimensionSize;
  MPI_Comm rowComm = std::get<0>(commInfo3D);
  MPI_Comm sliceComm = std::get<2>(commInfo3D);
  MPI_Comm_size(rowComm, &pGridDimensionSize);

  auto tup = getCommunicatorSlice(commWorld);
  int pCoordX = std::get<1>(tup);
  int pCoordY = std::get<2>(tup);
  int pCoordZ = std::get<3>(tup);
  

  // Remember, we are assuming that the matrix is square here
  U localDimension = globalDimensionX/pGridDimensionSize;

  // Can generate matrix later
  std::vector<T> globalMatrixA = getReferenceMatrix3D(matrixA, localDimension, globalDimensionY, pCoordX*pGridDimensionSize+pCoordY, tup); 
  std::vector<T> testMat(globalDimensionX*globalDimensionY,0);
  cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, globalDimensionX, globalDimensionY, 1., &globalMatrixA[0],
    globalDimensionY, 0., &testMat[0], globalDimensionX);
  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', globalDimensionX, &testMat[0], globalDimensionX);
  LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', globalDimensionX, &testMat[0], globalDimensionX);
  std::vector<T> testMatQ(globalDimensionX*globalDimensionX);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, globalDimensionX, globalDimensionX, globalDimensionY,
    1., &globalMatrixA[0], globalDimensionY, &testMat[0], globalDimensionY, 0., &testMatQ[0], globalDimensionX);
  cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, globalDimensionX, globalDimensionY, 1., &testMatQ[0],
    globalDimensionY, 0., &testMat[0], globalDimensionX);
  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', globalDimensionX, &testMat[0], globalDimensionX);
  LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', globalDimensionX, &testMat[0], globalDimensionX);
  std::vector<T> testMatQ2(globalDimensionX*globalDimensionX);
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, globalDimensionX, globalDimensionX, globalDimensionY,
    1., &testMatQ[0], globalDimensionY, &testMat[0], globalDimensionY, 0., &testMatQ2[0], globalDimensionX);
  std::vector<T> testMatI(globalDimensionX*globalDimensionX);
  cblas_dsyrk(CblasColMajor, CblasUpper, CblasTrans, globalDimensionX, globalDimensionY, 1., &testMatQ2[0],
    globalDimensionY, 0., &testMatI[0], globalDimensionX);

  // Now compare deviation from orthogonality
  T error = 0;
  for (int i=0; i<globalDimensionX; i++)
  {
    for (int j=0; j<globalDimensionX; j++)
    {
      if (i==j) {error += std::abs(testMatI[i*globalDimensionX+j] - 1);}
      else {error += std::abs(testMatI[i*globalDimensionX+j]);}
    }
  }

  if (myRank == 0) std::cout << "True deviation from orthogonality - " << error << std::endl;

  Matrix<T,U,MatrixStructureSquare,Distribution> matI(std::move(testMatI), globalDimensionX, globalDimensionX, globalDimensionX, globalDimensionX, true);
  if (myRank == 0)
  {
    std::cout << "Printing true I\n";
    matI.print();
  }

  // Now, for debugging 3D, lets run this into a gemm and see what happens

  // For now, until I talk with Edgar, lets just have a residual test and an orthogonality test

  // generate A_computed = myQ*myR and compare against original A
  T error1 = getResidual3D(matrixA, myQ, myR, globalDimensionX, globalDimensionY, commWorld);
  MPI_Allreduce(MPI_IN_PLACE, &error1, 1, MPI_DOUBLE, MPI_SUM, sliceComm);
  if (myRank == 0) {std::cout << "Total residual error is " << error1 << std::endl;}

  T error2 = testOrthogonality3D(myQ, globalDimensionX, globalDimensionY, commWorld);
  MPI_Allreduce(MPI_IN_PLACE, &error2, 1, MPI_DOUBLE, MPI_SUM, sliceComm);
  if (myRank == 0) {std::cout << "Deviation from orthogonality is " << error2 << std::endl;}

  return;
}


/* Used for comparing a matrix owned among processors in a 1D row-cyclic manner to a full matrix */
template<typename T, typename U>
T QRvalidate<T,U>::getResidual1D_RowCyclic(std::vector<T>& myMatrix, std::vector<T>& solutionMatrix, U globalDimensionX, U globalDimensionY, MPI_Comm commWorld)
{
  int numPEs, myRank;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_rank(commWorld, &myRank);
  U localDimensionY = globalDimensionY/numPEs;

  T error = 0;
  for (U i=0; i<globalDimensionX; i++)
  {
    for (U j=0; j<localDimensionY; j++)
    {
      U myIndex = i*localDimensionY+j;
      U solIndex = i*globalDimensionY+(j*numPEs+myRank);
      if (std::abs(myMatrix[myIndex] + solutionMatrix[solIndex]) <= 1e-12)
      {
        T errorSquare = std::abs(myMatrix[myIndex] + solutionMatrix[solIndex]);
        errorSquare *= errorSquare;
        //error += errorSquare;
        continue;
      }
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
T QRvalidate<T,U>::testOrthogonality1D(std::vector<T>& myQ, U globalDimensionX, U globalDimensionY, MPI_Comm commWorld)
{
  int numPEs, myRank;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_rank(commWorld, &myRank);
  U localDimensionY = globalDimensionY/numPEs;

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
T QRvalidate<T,U>::getResidual1D(std::vector<T>& myA, std::vector<T>& myQ, std::vector<T>&myR, U globalDimensionX, U globalDimensionY, MPI_Comm commWorld)
{
  int numPEs, myRank;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_rank(commWorld, &myRank);
  U localDimensionY = globalDimensionY/numPEs;

  std::vector<T> computedA(globalDimensionX*localDimensionY,0);
  // Again, for now, lets just use cblas, but I can encapsulate it into blasEngine later
  cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, localDimensionY, globalDimensionX, globalDimensionX,
    1., &myQ[0], localDimensionY, &myR[0], globalDimensionX, 0., &computedA[0], localDimensionY);

  T error = 0;
  for (U i=0; i<globalDimensionX; i++)
  {
    for (U j=0; j<localDimensionY; j++)
    {
      U myIndex = i*localDimensionY+j;
      T errorSquare = std::abs(myA[myIndex] - computedA[myIndex]);
      //if (myRank==0) std::cout << errorSquare << " " << myQ[myIndex] << " " << solQ[solIndex] << " i - " << i << ", j - " << j << std::endl;
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
    for (U j=i; j<globalDimensionX; j++)
    {
      U myIndex = i*globalDimensionX+j;
      if (std::abs(myMatrix[myIndex] + solutionMatrix[myIndex]) <= 1e-12)
      {
        T errorSquare = std::abs(myMatrix[myIndex] + solutionMatrix[myIndex]);
        errorSquare *= errorSquare;
        error += errorSquare;
        continue;
      }
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
T QRvalidate<T,U>::testComputedQR(std::vector<T>& myR, std::vector<T>& solR, U globalDimensionX, U globalDimensionY, MPI_Comm commWorld)
{
  // Add if needed. This would be myQ*myR - LAPACK-generated Q*R. Talk to Edgar about this on Friday.
  // Now, we should check my QR against A
/* Below is if we wanted to multiply the LAPACK-generated matrices Q and R to reform A, but now I don't think that test is useful.
  std::vector<T> matrixAtemp(globalDimensionX*globalDimensionY);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, globalDimensionY, globalDimensionX, globalDimensionX,
    1., &matrixQ[0], globalDimensionX, &matrixR[0], globalDimensionX, 0., &matrixAtemp[0], globalDimensionX);
*/
}


template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
T QRvalidate<T,U>::getResidual3D(Matrix<T,U,MatrixStructureSquare,Distribution>& myA,
                         Matrix<T,U,MatrixStructureSquare,Distribution>& myQ,
                         Matrix<T,U,MatrixStructureSquare,Distribution>& myR,
                         U globalDimensionX, U globalDimensionY, MPI_Comm commWorld)
{
  auto commInfo3D = setUpCommunicators(commWorld);

  // Simple asignments like these don't need pass-by-reference. Remember the new pass-by-value semantics are efficient anyways
  int pGridDimensionSize;
  MPI_Comm rowComm = std::get<0>(commInfo3D);
  MPI_Comm columnComm = std::get<1>(commInfo3D);
  MPI_Comm sliceComm = std::get<2>(commInfo3D);
  MPI_Comm depthComm = std::get<3>(commInfo3D);
  int pGridCoordX = std::get<4>(commInfo3D);
  int pGridCoordY = std::get<5>(commInfo3D);
  int pGridCoordZ = std::get<6>(commInfo3D);
  MPI_Comm_size(rowComm, &pGridDimensionSize);

  U localDimensionY = globalDimensionY/pGridDimensionSize;
  U localDimensionX = globalDimensionX/pGridDimensionSize;

  return 0;  
}


template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
T QRvalidate<T,U>::testOrthogonality3D(Matrix<T,U,MatrixStructureSquare,Distribution>& myQ,
                               U globalDimensionX, U globalDimensionY, MPI_Comm commWorld)
{
  int myRank;
  MPI_Comm_rank(commWorld, &myRank);

  auto commInfo3D = setUpCommunicators(commWorld);

  // Simple asignments like these don't need pass-by-reference. Remember the new pass-by-value semantics are efficient anyways
  int pGridDimensionSize;
  MPI_Comm rowComm = std::get<0>(commInfo3D);
  MPI_Comm columnComm = std::get<1>(commInfo3D);
  MPI_Comm sliceComm = std::get<2>(commInfo3D);
  MPI_Comm depthComm = std::get<3>(commInfo3D);
  int pGridCoordX = std::get<4>(commInfo3D);
  int pGridCoordY = std::get<5>(commInfo3D);
  int pGridCoordZ = std::get<6>(commInfo3D);
  MPI_Comm_size(rowComm, &pGridDimensionSize);

  U localDimensionY = globalDimensionY/pGridDimensionSize;
  U localDimensionX = globalDimensionX/pGridDimensionSize;
  // Need to multiply Q^T by Q, then we get back the result, and we compare against an implicit Identity matrix
  
  // First, I need to reform these matrices in order to call MM3D
  Matrix<T,U,MatrixStructureSquare,Distribution> myI(std::vector<T>(localDimensionX*localDimensionY,0), localDimensionX, localDimensionY,
    globalDimensionX, globalDimensionY, true);

  blasEngineArgumentPackage_gemm<double> blasArgs;
  blasArgs.order = blasEngineOrder::AblasColumnMajor;
  blasArgs.transposeA = blasEngineTranspose::AblasTrans;
  blasArgs.transposeB = blasEngineTranspose::AblasNoTrans;
  blasArgs.alpha = 1.;
  blasArgs.beta = 0.;

  // First, we need an explicit transpose
  int helper = pGridDimensionSize*pGridDimensionSize;
  int transposePartner = pGridCoordZ*helper + pGridCoordX*pGridDimensionSize + pGridCoordY;
  Matrix<T,U,MatrixStructureSquare,Distribution> QT = myQ;
  MPI_Sendrecv_replace(QT.getRawData(), localDimensionX*localDimensionY, MPI_DOUBLE, transposePartner, 0,
    transposePartner, 0, commWorld, MPI_STATUS_IGNORE);
  SquareMM3D<T,U,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare,cblasEngine>::Multiply(QT, myQ, myI, localDimensionX, localDimensionY, localDimensionX, commWorld, blasArgs);
/*
  if (myRank == 3)
  {
    myI.print();
  }
*/

  T error = 0;
  U myIndex = 0;
  U implicitIndexX = pGridCoordX;
  U implicitIndexY = pGridCoordY;

  for (U i=0; i<localDimensionX; i++)
  {
    U saveCountRef = implicitIndexY;
    for (U j=0; j<localDimensionY; j++)
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
  return error;		// return 2-norm
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
  MatrixType localMatrix(globalDimensionX, localDimensionY, globalDimensionX, globalDimensionY);
  localMatrix.DistributeRandom(0, myRank, 1, numPEs, key);

  U globalSize = globalDimensionX*globalDimensionY;
  std::vector<T> blockedMatrix(globalSize);
  std::vector<T> cyclicMatrix(globalSize);
  U localSize = globalDimensionX*localDimensionY;
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


template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
std::vector<T> QRvalidate<T,U>::getReferenceMatrix3D(
                        				Matrix<T,U,MatrixStructureSquare,Distribution>& myMatrix,
							U localDimension,
							U globalDimension,
							U key,
							std::tuple<MPI_Comm, int, int, int, int> commInfo
						  )
{
  MPI_Comm sliceComm = std::get<0>(commInfo);
  int pGridCoordX = std::get<1>(commInfo);
  int pGridCoordY = std::get<2>(commInfo);
  int pGridCoordZ = std::get<3>(commInfo);
  int pGridDimensionSize = std::get<4>(commInfo);

  // Remember, assume this algorithm works on square matrices for now
  using MatrixType = Matrix<T,U,MatrixStructureSquare,Distribution>;
  MatrixType localMatrix(localDimension, localDimension, globalDimension, globalDimension);
  localMatrix.DistributeRandom(pGridCoordX, pGridCoordY, pGridDimensionSize, pGridDimensionSize, key);

  U globalSize = globalDimension*globalDimension;
  std::vector<T> blockedMatrix(globalSize);
  std::vector<T> cyclicMatrix(globalSize);
  U localSize = localDimension*localDimension;
  MPI_Allgather(localMatrix.getRawData(), localSize, MPI_DOUBLE, &blockedMatrix[0], localSize, MPI_DOUBLE, sliceComm);

  U numCyclicBlocksPerRow = globalDimension/pGridDimensionSize;
  U numCyclicBlocksPerCol = globalDimension/pGridDimensionSize;
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
