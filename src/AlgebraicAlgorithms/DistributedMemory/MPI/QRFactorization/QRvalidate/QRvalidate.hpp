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
  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, globalDimensionY, globalDimensionX, globalMatrixA.getRawData(), globalDimensionX, &tau[0]);
  LAPACKE_dorgqr(LAPACK_ROW_MAJOR, globalDimensionY, globalDimensionX, globalDimensionX,globalMatrixA.getRawData(),
    globalDimensionX, &tau[0]);

  // Q is in globalMatrixA now

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error = getResidual1D(matrixSol_Q.getVectorData(), globalMatrixA.getVectorData(), globalDimensionX, globalDimensionY, commWorld);

  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, sliceComm);

  // Need to set up error2 for matrix R, but do that later
  T error2 = 0;
  return std::make_pair(error, error2);
}

template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
std::pair<T,T> QRvalidate<T,U>::validateLocal3D(
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
T QRvalidate<T,U>::getResidual1D(std::vector<T>& myQ, std::vector<T>& solQ, U globalDimensionX, U globalDimensionY, MPI_Comm commWorld)
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
      U solIndex = (i+localDimensionY*myRank)*globalDimensionX+j;
      T errorSquare = std::abs(myQ[myIndex] - solQ[solIndex]);
      if (myRank==0) std::cout << errorSquare << " " << myQ[myIndex] << " " << solQ[solIndex] << " i - " << i << ", j - " << j << std::endl;
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
      if (isRank1) std::cout << errorSquare << " " << myValues[countMyValues] << " " << lapackValues[countLapackValues] << std::endl;
      errorSquare *= errorSquare;
      error += errorSquare;
      countLapackValues += pGridDimensionSize;
      countMyValues++;
    }
    countLapackValues = saveCountRef + pGridDimensionSize*globalDimension;
    countMyValues += (localDimension - i-1);
  }

  error = std::sqrt(error);
  if (isRank1) std::cout << "Total error - " << error << "\n\n\n";
  return error;		// return 2-norm
}

