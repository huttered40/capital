/* Author: Edward Hutter */

template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
void CFvalidate<T,U>::validateLocal(
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixSol_CF,
                        char dir,
                        MPI_Comm commWorld
                      )
{
  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  int myRank;
  MPI_Comm_rank(commWorld, &myRank);

  auto commInfo = util<T,U>::getCommunicatorSlice(
    commWorld);
  MPI_Comm sliceComm = std::get<0>(commInfo);
  U pGridCoordX = std::get<1>(commInfo);
  U pGridCoordY = std::get<2>(commInfo);
  U pGridCoordZ = std::get<3>(commInfo);
  U pGridDimensionSize = std::get<4>(commInfo);

  U localDimension = matrixSol_CF.getNumRowsLocal();
  U globalDimension = matrixSol_CF.getNumRowsGlobal();
  std::vector<T> globalMatrixA = util<T,U>::getReferenceMatrix(
    matrixA, pGridCoordX*pGridDimensionSize+pGridCoordY, commInfo);

  // for ease in finding Frobenius Norm
  for (U i=0; i<globalDimension; i++)
  {
    for (U j=0; j<globalDimension; j++)
    {
      if ((dir == 'L') && (i>j)) globalMatrixA[i*globalDimension+j] = 0;
      if ((dir == 'U') && (j>i)) globalMatrixA[i*globalDimension+j] = 0;
    }
  }

  #if defined(BGQ) || defined(BLUEWATERS)
  int info;
  #ifdef FLOAT_TYPE
  spotrf_(/*LAPACK_COL_MAJOR, */&dir, &globalDimension, &globalMatrixA[0], &globalDimension, &info);
  #endif
  #ifdef DOUBLE_TYPE
  dpotrf_(/*LAPACK_COL_MAJOR, */&dir, &globalDimension, &globalMatrixA[0], &globalDimension, &info);
  #endif
  #ifdef COMPLEX_FLOAT_TYPE
  cpotrf_(/*LAPACK_COL_MAJOR, */&dir, &globalDimension, &globalMatrixA[0], &globalDimension, &info);
  #endif
  #ifdef COMPLEX_DOUBLE_TYPE
  zpotrf_(/*LAPACK_COL_MAJOR, */&dir, &globalDimension, &globalMatrixA[0], &globalDimension, &info);
  #endif
  #else 
  #ifdef FLOAT_TYPE
  LAPACKE_spotrf(LAPACK_COL_MAJOR, dir, globalDimension, &globalMatrixA[0], globalDimension);
  #endif
  #ifdef DOUBLE_TYPE
  LAPACKE_dpotrf(LAPACK_COL_MAJOR, dir, globalDimension, &globalMatrixA[0], globalDimension);
  #endif
  #ifdef COMPLEX_FLOAT_TYPE
  LAPACKE_cpotrf(LAPACK_COL_MAJOR, dir, globalDimension, &globalMatrixA[0], globalDimension);
  #endif
  #ifdef COMPLEX_DOUBLE_TYPE
  LAPACKE_zpotrf(LAPACK_COL_MAJOR, dir, globalDimension, &globalMatrixA[0], globalDimension);
  #endif
  #endif

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error = (dir == 'L' ? getResidualTriangleLower(matrixSol_CF.getVectorData(), globalMatrixA, localDimension, globalDimension, commInfo)
              : getResidualTriangleUpper(matrixSol_CF.getVectorData(), globalMatrixA, localDimension, globalDimension, commInfo));

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

template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
T CFvalidate<T,U>::validateParallel(
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixTri,
                        char dir,
                        MPI_Comm commWorld,
                        std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D
                      )
{
  int rank,size;
  MPI_Comm_rank(commWorld, &rank);
  MPI_Comm_size(commWorld, &size);

  auto commInfo = util<T,U>::getCommunicatorSlice(
    commWorld);
  MPI_Comm sliceComm = std::get<0>(commInfo);
  U pGridCoordX = std::get<1>(commInfo);
  U pGridCoordY = std::get<2>(commInfo);
  U pGridCoordZ = std::get<3>(commInfo);
  U pGridDimensionSize = std::get<4>(commInfo);
  int helper = pGridDimensionSize;
  helper *= helper;

  #if defined(BLUEWATERS) || defined(STAMPEDE2)
  int transposePartner = pGridCoordX*helper + pGridCoordY*pGridDimensionSize + pGridCoordZ;
  #else
  int transposePartner = pGridCoordZ*helper + pGridCoordX*pGridDimensionSize + pGridCoordY;
  #endif
  util<T,U>::removeTriangle(matrixTri, pGridCoordX, pGridCoordY, pGridDimensionSize, dir);
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixTriTrans = matrixTri;
  util<T,U>::transposeSwap(
    matrixTriTrans, rank, transposePartner, MPI_COMM_WORLD);
  std::string str = "Residual: ";
  return validator<T,U>::validateResidualParallel(
    (dir == 'L' ? matrixTri : matrixTriTrans), (dir == 'L' ? matrixTriTrans : matrixTri), matrixA, dir, commWorld, commInfo3D, MPI_COMM_WORLD, str);
}

// We only test the lower triangular for now. The matrices are stored with square structure though.
template<typename T, typename U>
T CFvalidate<T,U>::getResidualTriangleLower(
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

  U myIndex = 0;
  U solIndex = pCoordY + pCoordX*globalDimension;
  // We want to truncate this to use only the data that we own
  U trueDimensionM = globalDimension/pGridDimensionSize + ((pCoordY < (globalDimension%pGridDimensionSize)) ? 1 : 0);
  U trueDimensionN = globalDimension/pGridDimensionSize + ((pCoordX < (globalDimension%pGridDimensionSize)) ? 1 : 0);

  for (U i=0; i<trueDimensionN; i++)
  {
    U saveCountRef = solIndex;
    U saveCountMy = myIndex;
    for (U j=0; j<trueDimensionM; j++)
    {
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
T CFvalidate<T,U>::getResidualTriangleUpper(
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
  U myIndex = 0;
  U solIndex = pCoordX*globalDimension + pCoordY;

  // We want to truncate this to use only the data that we own
  U trueDimensionM = globalDimension/pGridDimensionSize + ((pCoordY < (globalDimension%pGridDimensionSize)) ? 1 : 0);
  U trueDimensionN = globalDimension/pGridDimensionSize + ((pCoordX < (globalDimension%pGridDimensionSize)) ? 1 : 0);
  for (U i=0; i<trueDimensionN; i++)
  {
    U saveCountRef = solIndex;
    U saveCountMy = myIndex;
    for (U j=0; j<trueDimensionM; j++)
    {
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
