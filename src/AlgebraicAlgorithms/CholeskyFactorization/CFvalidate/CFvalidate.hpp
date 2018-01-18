/* Author: Edward Hutter */

/*  Already defined in MMvalidate
static std::tuple<MPI_Comm, int, int, int, int> getCommunicatorSlice(MPI_Comm commWorld)
{
  int rank,size;
  MPI_Comm_rank(commWorld, &rank);
  MPI_Comm_size(commWorld, &size);

  int pGridDimensionSize = std::nearbyint(std::pow(size,1./3.));
  
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
void CFvalidate<T,U>::validateCF_Local(
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixSol_CF,
                        Matrix<T,U,MatrixStructureSquare,Distribution>& matrixSol_TI,
                        char dir,
                        MPI_Comm commWorld
                      )
{
  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  int myRank;
  MPI_Comm_rank(commWorld, &myRank);

  auto commInfo = getCommunicatorSlice(commWorld);
  MPI_Comm sliceComm = std::get<0>(commInfo);
  U pGridCoordX = std::get<1>(commInfo);
  U pGridCoordY = std::get<2>(commInfo);
  U pGridCoordZ = std::get<3>(commInfo);
  U pGridDimensionSize = std::get<4>(commInfo);

  U localDimension = matrixSol_CF.getNumRowsLocal();
  U globalDimension = matrixSol_CF.getNumRowsGlobal();
  std::vector<T> globalMatrixA = getReferenceMatrix(matrixSol_CF, localDimension, globalDimension, pGridCoordX*pGridDimensionSize+pGridCoordY, commInfo);

  // for ease in finding Frobenius Norm
  for (U i=0; i<globalDimension; i++)
  {
    for (U j=0; j<globalDimension; j++)
    {
      if ((dir == 'L') && (i>j)) globalMatrixA[i*globalDimension+j] = 0;
      if ((dir == 'U') && (j>i)) globalMatrixA[i*globalDimension+j] = 0;
    }
  }

  // Assume row-major
  pTimer myTimer;
  myTimer.setStartTime();
  LAPACKE_dpotrf(LAPACK_COL_MAJOR, dir, globalDimension, &globalMatrixA[0], globalDimension);
  myTimer.setEndTime();
  myTimer.printParallelTime(1e-9, MPI_COMM_WORLD, "LAPACK Cholesky Factorization (dpotrf)");

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error = (dir == 'L' ? getResidualTriangleLower(matrixSol_CF.getVectorData(), globalMatrixA, localDimension, globalDimension, commInfo)
              : getResidualTriangleUpper(matrixSol_CF.getVectorData(), globalMatrixA, localDimension, globalDimension, commInfo));

  MPI_Allreduce(MPI_IN_PLACE, &error, 1, MPI_DOUBLE, MPI_SUM, sliceComm);
  if (myRank == 0) {std::cout << "Total error = " << error << std::endl;}

  myTimer.setStartTime();
  LAPACKE_dtrtri(LAPACK_COL_MAJOR, dir, 'N', globalDimension, &globalMatrixA[0], globalDimension);
  myTimer.setEndTime();
  myTimer.printParallelTime(1e-9, MPI_COMM_WORLD, "LAPACK Triangular Inverse (dtrtri)");

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
  T error2 = (dir == 'L' ? getResidualTriangleLower(matrixSol_TI.getVectorData(), globalMatrixA, localDimension, globalDimension, commInfo)
               : getResidualTriangleUpper(matrixSol_TI.getVectorData(), globalMatrixA, localDimension, globalDimension, commInfo));

  // Now, we need the AllReduce of the error. Very cheap operation in terms of bandwidth cost, since we are only communicating a single double primitive type.
  MPI_Allreduce(MPI_IN_PLACE, &error2, 1, MPI_DOUBLE, MPI_SUM, sliceComm);
  if (myRank == 0) {std::cout << "Total error = " << error2 << std::endl;}

  MPI_Comm_free(&sliceComm);
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
  if ((pCoordY == 0) && (pCoordX == 1) && (pCoordZ == 0))
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

template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
std::vector<T> CFvalidate<T,U>::getReferenceMatrix(
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

  using MatrixType = Matrix<T,U,MatrixStructureSquare,Distribution>;
  MatrixType localMatrix(globalDimension, globalDimension, pGridDimensionSize, pGridDimensionSize);
  localMatrix.DistributeSymmetric(pGridCoordX, pGridCoordY, pGridDimensionSize, pGridDimensionSize, key, true);

  U aggregNumRows = localDimension*pGridDimensionSize;
  U aggregNumColumns = localDimension*pGridDimensionSize;
  U localSize = localDimension*localDimension;
  U globalSize = globalDimension*globalDimension;
  U aggregSize = aggregNumRows*aggregNumColumns;
  std::vector<T> blockedMatrix(aggregSize);
  std::vector<T> cyclicMatrix(aggregSize);
  MPI_Allgather(localMatrix.getRawData(), localSize, MPI_DOUBLE, &blockedMatrix[0], localSize, MPI_DOUBLE, sliceComm);


  U numCyclicBlocksPerRow = localDimension;
  U numCyclicBlocksPerCol = localDimension;
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
  // In case there are hidden zeros, we will recopy
  if (globalDimension%pGridDimensionSize)
  {
    U index = 0;
    for (U i=0; i<globalDimension; i++)
    {
      for (U j=0; j<globalDimension; j++)
      {
        cyclicMatrix[index++] = cyclicMatrix[i*aggregNumRows+j];
      }
    }
    // In this case, globalSize < aggregSize
    cyclicMatrix.resize(globalSize);
  }

  return cyclicMatrix;
}
