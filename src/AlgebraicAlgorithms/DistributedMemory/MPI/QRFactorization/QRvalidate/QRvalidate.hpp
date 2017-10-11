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


template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
std::pair<T,T> QRvalidate<T,U>::validateLocal1D(
                        Matrix<T,U,MatrixStructureRectangle,Distribution>& matrixA,
                        Matrix<T,U,MatrixStructureRectangle,Distribution>& matrixSol_Q,
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

  return std::make_pair(error2, error3);
}

template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
std::pair<T,T> QRvalidate<T,U>::validateLocal3D(
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixSol_Q,
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixSol_R,
                        U globalDimensionX,
                        U globalDimensionY,
                        MPI_Comm commWorld
                      )
{
  // Fill in later after 1D is correct
}


template<typename T, typename U>
T QRvalidate<T,U>::getResidual1D_Q(std::vector<T>& myQ, std::vector<T>& solQ, U globalDimensionX, U globalDimensionY, MPI_Comm commWorld)
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
      if (std::abs(myQ[myIndex] + solQ[solIndex]) <= 1e-7)
      {
        T errorSquare = std::abs(myQ[myIndex] + solQ[solIndex]);
        errorSquare *= errorSquare;
        error += errorSquare;
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

template<typename T, typename U>
T QRvalidate<T,U>::getResidual1D_R(std::vector<T>& myQ, std::vector<T>& solQ, U globalDimensionX, U globalDimensionY, MPI_Comm commWorld)
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
      if (std::abs(myQ[myIndex] + solQ[myIndex]) <= 1e-7)
      {
        T errorSquare = std::abs(myQ[myIndex] + solQ[myIndex]);
        errorSquare *= errorSquare;
        error += errorSquare;
        continue;
      }
      T errorSquare = std::abs(myQ[myIndex] - solQ[myIndex]);
      //if (myRank==0) std::cout << errorSquare << " " << myQ[myIndex] << " " << solQ[myIndex] << " i - " << i << ", j - " << j << std::endl;
      errorSquare *= errorSquare;
      error += errorSquare;
    }
  }

  error = std::sqrt(error);
  return error;
}


// We only test the lower triangular for now. The matrices are stored with square structure though.
template<typename T, typename U>
T QRvalidate<T,U>::getResidualTriangle(
		     std::vector<T>& myValues,
		     std::vector<T>& lapackValues,
		     U localDimension,
		     U globalDimension,
		     std::tuple<MPI_Comm, int, int, int, int> commInfo
		   )
{
  T error = 0;
  int pCoordX = std::get<1>(commInfo);
  int pCoordY = std::get<2>(commInfo);
  int pCoordZ = std::get<3>(commInfo);
  bool isRank1 = false;
  if ((pCoordY == 0) && (pCoordX == 0) && (pCoordZ == 0))
  {
    isRank1 = true;
  }

  int pGridDimensionSize = std::get<4>(commInfo);
  U countMyValues = 0;
  U countLapackValues = pCoordX + pCoordY*globalDimension;

  for (U i=0; i<localDimension; i++)
  {
    U saveCountRef = countLapackValues;
    for (U j=0; j<=i; j++)
    {
      T errorSquare = std::abs(myValues[countMyValues] - lapackValues[countLapackValues]);
      //if (isRank1) std::cout << errorSquare << " " << myValues[countMyValues] << " " << lapackValues[countLapackValues] << std::endl;
      errorSquare *= errorSquare;
      error += errorSquare;
      countLapackValues += pGridDimensionSize;
      countMyValues++;
    }
    countLapackValues = saveCountRef + pGridDimensionSize*globalDimension;
    countMyValues += (localDimension - i-1);
  }

  error = std::sqrt(error);
  //if (isRank1) std::cout << "Total error - " << error << "\n\n\n";
  return error;		// return 2-norm
}

