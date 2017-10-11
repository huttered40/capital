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

  int myRank;
  MPI_Comm_rank(commWorld, &myRank);

  using globalMatrixType = Matrix<T,U,MatrixStructureRectangle,Distribution>;
  globalMatrixType globalMatrixA(globalDimensionX,globalDimensionY,globalDimensionX,globalDimensionY);
  globalMatrixA.DistributeRandom(0, 0, 1, 1);		// Hardcode so that the Distributer thinks we own the entire matrix.

  // Assume row-major
  std::vector<T> tau(globalDimensionX);
  std::vector<T> matrixQ = globalMatrixA.getVectorData();		// true copy
  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, globalDimensionY, globalDimensionX, &matrixQ[0], globalDimensionX, &tau[0]);
  LAPACKE_dorgqr(LAPACK_ROW_MAJOR, globalDimensionY, globalDimensionX, globalDimensionX, &matrixQ[0],
    globalDimensionX, &tau[0]);

  // Q is in globalMatrixA now

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error = getResidual1D_RowCyclic(myQ.getVectorData(), matrixQ, globalDimensionX, globalDimensionY, commWorld);

  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, commWorld);
  if (myRank == 0) {std::cout << "Total error of myQ - solQ is " << error << std::endl;}

  // Now generate R using Q and A
  std::vector<T> matrixR(globalDimensionX*globalDimensionX,0);
  // For right now, I will just use cblas, but Note that I should template this class with blasEngine
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, globalDimensionX, globalDimensionX, globalDimensionY,
    1., &matrixQ[0], globalDimensionX, globalMatrixA.getRawData(), globalDimensionX, 0., &matrixR[0], globalDimensionX);

  // Need to set up error2 for matrix R, but do that later
  T error2 = getResidual1D_Full(myR.getVectorData(), matrixR, globalDimensionX, globalDimensionY, commWorld);
  MPI_Allreduce(MPI_IN_PLACE, &error2, 1, MPI_DOUBLE, MPI_SUM, commWorld);
  if (myRank == 0) {std::cout << "Total error of myR - solR is " << error2 << std::endl;}

  // Now, we should check my original A against computed QR and use the getResidual1D_Q to check, since A and Q are of same shape
  std::vector<T> matrixAtemp(globalDimensionX*globalDimensionY);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, globalDimensionY, globalDimensionX, globalDimensionX,
    1., &matrixQ[0], globalDimensionX, &matrixR[0], globalDimensionX, 0., &matrixAtemp[0], globalDimensionX);

  // Hold on, this is the wrong test. I need to be comparing my QR - LAPACK QR.
  T error3 = getResidual1D_RowCyclic(matrixA.getVectorData(), matrixAtemp, globalDimensionX, globalDimensionY, commWorld);
  MPI_Allreduce(MPI_IN_PLACE, &error3, 1, MPI_DOUBLE, MPI_SUM, commWorld);
  if (myRank == 0) {std::cout << "Total error of myQ - solQ is " << error3 << std::endl;}

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
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixSol_Q,
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixSol_R,
                        U globalDimensionX,
                        U globalDimensionY,
                        MPI_Comm commWorld
                      )
{
  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  using globalMatrixType = Matrix<T,U,MatrixStructureRectangle,Distribution>;
  globalMatrixType globalMatrixA(globalDimensionX,globalDimensionY,globalDimensionX,globalDimensionY);
  globalMatrixA.DistributeRandom(0, 0, 1, 1);		// Hardcode so that the Distributer thinks we own the entire matrix.

  auto commInfo = getCommunicatorSlice(commWorld);
  MPI_Comm sliceComm = std::get<0>(commInfo);

  // Assume row-major
  std::vector<T> tau(globalDimensionX);
  std::vector<T> matrixQ = globalMatrixA.getVectorData();		// true copy
  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, globalDimensionY, globalDimensionX, &matrixQ[0], globalDimensionX, &tau[0]);
  LAPACKE_dorgqr(LAPACK_ROW_MAJOR, globalDimensionY, globalDimensionX, globalDimensionX, &matrixQ[0],
    globalDimensionX, &tau[0]);

  // Q is in globalMatrixA now

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error = getResidual1D_Q(matrixSol_Q.getVectorData(), matrixQ, globalDimensionX, globalDimensionY, commWorld);

  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, sliceComm);

  // Now generate R using Q and A
  std::vector<T> matrixR(globalDimensionX*globalDimensionX,0);
  // For right now, I will just use cblas, but Note that I should template this class with blasEngine
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasNoTrans, globalDimensionX, globalDimensionX, globalDimensionY,
    1., &matrixQ[0], globalDimensionX, globalMatrixA.getRawData(), globalDimensionX, 0., &matrixR[0], globalDimensionX);

  // Need to set up error2 for matrix R, but do that later
  T error2 = getResidual1D_R(matrixSol_R.getVectorData(), matrixR, globalDimensionX, globalDimensionY, commWorld);

  // Now, we should check my original A against computed QR and use the getResidual1D_Q to check, since A and Q are of same shape
  std::vector<T> matrixAtemp(globalDimensionX*globalDimensionY);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, globalDimensionY, globalDimensionX, globalDimensionX,
    1., &matrixQ[0], globalDimensionX, &matrixR[0], globalDimensionX, 0., &matrixAtemp[0], globalDimensionX);

  T error3 = getResidual1D_Q(matrixA.getVectorData(), matrixAtemp, globalDimensionX, globalDimensionY, commWorld);

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
  for (U i=0; i<localDimensionY; i++)
  {
    for (U j=0; j<globalDimensionX; j++)
    {
      U myIndex = i*globalDimensionX+j;
      U solIndex = (i*numPEs+myRank)*globalDimensionX+j;
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
  cblas_dsyrk(CblasRowMajor, CblasUpper, CblasTrans, globalDimensionX, localDimensionY, 1., &myQ[0],
    globalDimensionX, 0., &myI[0], globalDimensionX);

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

template<typename T, typename U>
T QRvalidate<T,U>::testResidual1D(std::vector<T>& myQ, std::vector<T>& solQ, U globalDimensionX, U globalDimensionY, MPI_Comm commWorld)
{
  int numPEs, myRank;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_rank(commWorld, &myRank);
  U localDimensionY = globalDimensionY/numPEs;

  T error = 0;
  for (U i=0; i<localDimensionY; i++)
  {
    for (U j=0; j<globalDimensionX; j++)
    {
      U myIndex = i*globalDimensionX+j;
      U solIndex = (i*numPEs+myRank)*globalDimensionX+j;
      if (std::abs(myQ[myIndex] + solQ[solIndex]) <= 1e-12)
      {
        T errorSquare = std::abs(myQ[myIndex] + solQ[solIndex]);
        errorSquare *= errorSquare;
        //error += errorSquare;
        continue;
      }
      T errorSquare = std::abs(myQ[myIndex] - solQ[solIndex]);
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
T testComputedQR(std::vector<T>& myR, std::vector<T>& solR, U globalDimensionX, U globalDimensionY, MPI_Comm commWorld)
{

}
