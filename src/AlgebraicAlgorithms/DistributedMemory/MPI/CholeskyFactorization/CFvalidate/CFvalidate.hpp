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

// We enforce that matrixSol must have Square Structure.

template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
std::pair<T,T> CFvalidate<T,U>::validateCF_Local(
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixSol_CF,
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixSol_TI,
                        U localDimension,
                        U globalDimension,
                        MPI_Comm commWorld
                      )
{
  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  auto commInfo = getCommunicatorSlice(commWorld);
  MPI_Comm sliceComm = std::get<0>(commInfo);

  using globalMatrixType = Matrix<T,U,MatrixStructureSquare,Distribution>;
  globalMatrixType globalMatrixA(globalDimension,globalDimension,globalDimension,globalDimension);
  globalMatrixA.DistributeSymmetric(0, 0, 1, 1, true);		// Hardcode so that the Distributer thinks we own the entire matrix.

  // Assume row-major
  LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'L', globalDimension, globalMatrixA.getRawData(), globalDimension);

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error = getResidualTriangle(matrixSol_CF.getVectorData(), globalMatrixA.getVectorData(), localDimension, globalDimension, commInfo);

  LAPACKE_dtrtri(LAPACK_ROW_MAJOR, 'L', 'N', globalDimension, globalMatrixA.getRawData(), globalDimension);

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error2 = getResidualTriangle(matrixSol_TI.getVectorData(), globalMatrixA.getVectorData(), localDimension, globalDimension, commInfo);

  // Now, we need the AllReduce of the error. Very cheap operation in terms of bandwidth cost, since we are only communicating a single double primitive type.
  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, sliceComm);
  return std::make_pair(error, error2);
}


// We only test the lower triangular for now. The matrices are stored with square structure though.
template<typename T, typename U>
T CFvalidate<T,U>::getResidualTriangle(
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
  bool isRank0 = false;
  if ((pCoordX == pCoordY) && (pCoordX == 0) && (pCoordZ == 0))
  {
    isRank0 = true;
  }

  int pGridDimensionSize = std::get<4>(commInfo);
  U countMyValues = 0;
  U countLapackValues = pCoordX + pCoordY*globalDimension;

  for (U i=0; i<localDimension; i++)
  {
    U saveCountRef = countLapackValues;
    for (U j=0; j<=i; j++)
    {
      T errorSquare = abs(myValues[countMyValues] - lapackValues[countLapackValues]);
      if (isRank0) std::cout << errorSquare << " " << myValues[countMyValues] << " " << lapackValues[countLapackValues] << std::endl;
      errorSquare *= errorSquare;
      error += errorSquare;
      countLapackValues += pGridDimensionSize;
      countMyValues++;
    }
    countLapackValues = saveCountRef + pGridDimensionSize*globalDimension;
    countMyValues += (localDimension - i-1);
  }

  error = std::sqrt(error);
  if (isRank0) std::cout << "Total error - " << error << "\n\n\n";
  return error;		// return 2-norm
}

