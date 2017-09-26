/* Author: Edward Hutter */


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

// We enforce that matrixSol must have Square Structure.

template<typename T, typename U, template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
T MMvalidate<T,U,blasEngine>::validateLocal(
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixSol,
                        U localDimensionX,
                        U localDimensionY,
                        U localDimensionZ,
                        U globalDimensionX,
                        U globalDimensionY,
                        U globalDimensionZ,
                        MPI_Comm commWorld,
                        const blasEngineArgumentPackage_gemm<T>& srcPackage
                      )
{
  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  std::tuple<MPI_Comm, int, int, int, int> commInfo = getCommunicatorSlice(commWorld);
  MPI_Comm sliceComm = std::get<0>(commInfo);

  using globalMatrixType = Matrix<T,U,MatrixStructureSquare,Distribution>;
  globalMatrixType globalMatrixA(globalDimensionX,globalDimensionY,globalDimensionX,globalDimensionY);
  globalMatrixType globalMatrixB(globalDimensionZ,globalDimensionX,globalDimensionZ,globalDimensionX);
  globalMatrixA.DistributeRandom(0, 0, 1, 1);		// Hardcode so that the Distributer thinks we own the entire matrix.
  globalMatrixB.DistributeRandom(0, 0, 1, 1);		// Hardcode so that the Distributer thinks we own the entire matrix.

  std::vector<T> matrixAforEngine = globalMatrixA.getVectorData();
  std::vector<T> matrixBforEngine = globalMatrixB.getVectorData();
  std::vector<T> matrixCforEngine(globalDimensionY*globalDimensionZ, 0);	// No matrix needed for this. Only used in BLAS call

  blasEngine<T,U>::_gemm(&matrixAforEngine[0], &matrixBforEngine[0], &matrixCforEngine[0], globalDimensionX, globalDimensionY,
    globalDimensionX, globalDimensionZ, globalDimensionY, globalDimensionZ, globalDimensionY, globalDimensionX, globalDimensionY, srcPackage);

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error = getResidualSquare(matrixSol.getVectorData(), matrixCforEngine, localDimensionX, localDimensionY, localDimensionZ, globalDimensionX,
    globalDimensionY, globalDimensionZ, commInfo);

  // Now, we need the AllReduce of the error. Very cheap operation in terms of bandwidth cost, since we are only communicating a single double primitive type.
  MPI_Allreduce(MPI_IN_PLACE, &error, sizeof(T), MPI_CHAR, MPI_SUM, sliceComm);
  return error;
}

template<typename T, typename U, template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
T MMvalidate<T,U,blasEngine>::validateLocal(
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixSol,
                        U localDimensionX,
                        U localDimensionY,
                        U localDimensionZ,
                        U globalDimensionX,
                        U globalDimensionY,
                        U globalDimensionZ,
                        MPI_Comm commWorld,
                        const blasEngineArgumentPackage_trmm<T>& srcPackage
                      )
{
  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  std::tuple<MPI_Comm, int, int, int> commInfo = getCommunicatorSlice(commWorld);
  MPI_Comm sliceComm = std::get<0>(commInfo);

  using globalMatrixType = Matrix<T,U,MatrixStructureSquare,Distribution>;
  globalMatrixType globalMatrixA(globalDimensionX,globalDimensionY,globalDimensionX,globalDimensionY);
  globalMatrixType globalMatrixB(globalDimensionZ,globalDimensionX,globalDimensionZ,globalDimensionX);
  globalMatrixA.DistributeRandom(0, 0, 1, 1);		// Hardcode so that the Distributer thinks we own the entire matrix.
  globalMatrixB.DistributeRandom(0, 0, 1, 1);		// Hardcode so that the Distributer thinks we own the entire matrix.

  std::vector<T> matrixAforEngine = globalMatrixA.getVectorData();
  std::vector<T> matrixBforEngine = globalMatrixB.getVectorData();

  blasEngine<T,U>::_trmm(&matrixAforEngine[0], &matrixBforEngine[0], (srcPackage.side == blasEngineSide::AblasLeft ? globalDimensionX : globalDimensionY),
    (srcPackage.side == blasEngineSide::AblasLeft ? globalDimensionZ : globalDimensionX), (srcPackage.side == blasEngineSide::AblasLeft ? globalDimensionY : globalDimensionX),
    (srcPackage.side == blasEngineSide::AblasLeft ? globalDimensionX : globalDimensionY), srcPackage);

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error = getResidualSquare(matrixSol, matrixBforEngine, localDimensionX, localDimensionY, localDimensionZ, globalDimensionX,
    globalDimensionY, globalDimensionZ, commInfo);

  // Now, we need the AllReduce of the error. Very cheap operation in terms of bandwidth cost, since we are only communicating a single double primitive type.
  MPI_Allreduce(MPI_IN_PLACE, &error, sizeof(T), MPI_CHAR, MPI_SUM, sliceComm);
  return error;
}

template<typename T, typename U, template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
T MMvalidate<T,U,blasEngine>::validateLocal(
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixSol,
                        U localDimensionX,
                        U localDimensionY,
                        U localDimensionZ,
                        U globalDimensionX,
                        U globalDimensionY,
                        U globalDimensionZ,
                        MPI_Comm commWorld,
                        const blasEngineArgumentPackage_syrk<T>& srcPackage
                      )
{
  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  std::tuple<MPI_Comm, int, int, int> commInfo = getCommunicatorSlice(commWorld);
  MPI_Comm sliceComm = std::get<0>(commInfo);

  using globalMatrixType = Matrix<T,U,MatrixStructureSquare,Distribution>;
  globalMatrixType globalMatrixA(globalDimensionX,globalDimensionY,globalDimensionX,globalDimensionY);
  globalMatrixType globalMatrixB(globalDimensionZ,globalDimensionX,globalDimensionZ,globalDimensionX);
  globalMatrixA.DistributeRandom(0, 0, 1, 1);		// Hardcode so that the Distributer thinks we own the entire matrix.

  std::vector<T> matrixAforEngine = globalMatrixA.getVectorData();
  std::vector<T> matrixBforEngine = globalMatrixB.getVectorData();	// Instead of using C for output matrix, lets use B as we did in SquareMM3D

  blasEngine<T,U>::_syrk(&matrixAforEngine[0], &matrixBforEngine[0], globalDimensionX, globalDimensionY,
    globalDimensionX, globalDimensionY, srcPackage);

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error = getResidualSquare(matrixSol, matrixBforEngine, localDimensionX, localDimensionY, localDimensionZ, globalDimensionX,
    globalDimensionY, globalDimensionZ, commInfo);

  // Now, we need the AllReduce of the error. Very cheap operation in terms of bandwidth cost, since we are only communicating a single double primitive type.
  MPI_Allreduce(MPI_IN_PLACE, &error, sizeof(T), MPI_CHAR, MPI_SUM, sliceComm);
  return error;
}
  
template<typename T, typename U, template<typename,typename> class blasEngine>
T MMvalidate<T,U,blasEngine>::getResidualSquare(
		     std::vector<T>& myValues,
		     std::vector<T>& blasValues,
		     U localDimensionX,
		     U localDimensionY,
		     U localDimensionZ,
		     U globalDimensionX,
		     U globalDimensionY,
	   	     U globalDimensionZ,
		     std::tuple<MPI_Comm, int, int, int, int> commInfo
		   )
{
  T error = 0;
  int pCoordX = std::get<1>(commInfo);
  int pCoordY = std::get<2>(commInfo);
  int pGridDimensionSize = std::get<4>(commInfo);
  U countMyValues = 0;
  U countBlasValues = pCoordX + pCoordY*globalDimensionX;

  for (U i=0; i<localDimensionY; i++)
  {
    U saveCountRef = countBlasValues;
    for (U j=0; j<localDimensionZ; j++)
    {
      T errorSquare = abs(myValues[countMyValues] - blasValues[countBlasValues]);
      //std::cout << errorSquare << " " << myValues[countMyValues] << " " << blasValues[countBlasValues] << std::endl;
      errorSquare *= errorSquare;
      error += errorSquare;
      countBlasValues += pGridDimensionSize;
      countMyValues++;
    }
    countBlasValues = saveCountRef + pGridDimensionSize*globalDimensionX;
  }

  return sqrt(error);		// return 2-norm
}
