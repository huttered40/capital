/* Author: Edward Hutter */


template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void TRSM3D<T,U,blasEngine>::Solve(
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixT,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixTI,
  char dir,
  int tune,
  MPI_Comm commWorld,
  int MM_id
  )
{
  // Need to split up the commWorld communicator into a 3D grid similar to Summa3D
  int rank,size;
  MPI_Comm_rank(commWorld, &rank);
  MPI_Comm_size(commWorld, &size);

  int pGridDimensionSize = std::nearbyint(std::pow(size,1./3.));
  int helper = pGridDimensionSize;
  helper *= helper;
  int pGridCoordX = rank%pGridDimensionSize;
  int pGridCoordY = (rank%helper)/pGridDimensionSize;
  int pGridCoordZ = rank/helper;
  int transposePartner = pGridCoordZ*helper + pGridCoordX*pGridDimensionSize + pGridCoordY;

  // Attain the communicator with only processors on the same 2D slice
  MPI_Comm slice2D;
  MPI_Comm_split(commWorld, pGridCoordZ, rank, &slice2D);

  U localDimension = matrixA.getNumRowsLocal();
  U globalDimension = matrixA.getNumRowsGlobal();
  // the division below may have a remainder, but I think integer division will be ok, as long as we change the base case condition to be <= and not just ==
  U bcDimension = globalDimension/helper;		// Can be tuned later.

  // Basic tuner was added
  for (int i=0; i<tune; i++)
  {
    bcDimension *= 2;
  }
  bcDimension = std::min(bcDimension, globalDimension/pGridDimensionSize);
/*
  if (rank == 0)
  {
    std::cout << "localDimension - " << localDimension << std::endl;
    std::cout << "globalDimension - " << globalDimension << std::endl;
    std::cout << "bcDimension - " << bcDimension << std::endl;
  }
*/

  if (dir == 'L')
  {
    iSolveLower(matrixA, matrixT, matrixTI, localDimension, localDimension, bcDimension, globalDimension, globalDimension,
      0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, transposePartner, MM_id, slice2D, commWorld);
  }
  else if (dir == 'U')
  {
    iSolveUpper(matrixA, matrixT, matrixTI, localDimension, localDimension, bcDimension, globalDimension, globalDimension,
      0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, 0, localDimension, transposePartner, MM_id, slice2D, commWorld);
  }
}

template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void TRSM3D<T,U,blasEngine>::iSolveLower(
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixL,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixLI,
  U localDimension,
  U trueLocalDimension,
  U bcDimension,
  U globalDimension,
  U trueGlobalDimension,
  U matAstartX,
  U matAendX,
  U matAstartY,
  U matAendY,
  U matLstartX,
  U matLendX,
  U matLstartY,
  U matLendY,
  U matLIstartX,
  U matLIendX,
  U matLIstartY,
  U matLIendY,
  U transposePartner,
  int MM_id,
  MPI_Comm commWorld )	// We want to pass in commWorld as MPI_COMM_WORLD because we want to pass that into 3D MM
{
  .. the size of the triangular inverses will be attained via trueLocalDimension and bcDimensio
  U numBlockRows = ...
  U numBlockColumns = ...
  blasEngineArgumentPackage_gemm<T> blasArgs;
  blasArgs.order = blasEngineOrder::AblasColumnMajor;
  blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
  blasArgs.transposeB = blasEngineTranspose::AblasTrans;
  blasArgs.alpha = 1.;
  blasArgs.beta = -1.;
  MM3D<T,U,blasEngine>::Multiply(matrixA, packedMatrix, matrixL, matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY,
      0, localShift, 0, localShift, matLstartX, matLstartX+localShift, matLstartY+localShift, matLendY, commWorld, blasArgs, true, false, true, MM_id);
  // Lets operate on individual columns at a time
  // Potential optimization 1): Don't use MM3D if the columns are too skinny in relation to the block size!
  //   Or this could just be taken care of when we tune block sizes?
  for (U i=0; i<numBlockColumns; i++)
  {
      // Update the current column by accumulating the updates via MM
      if (i>0)
      {
        blasArgs.beta = -1;
      }
      // Solve via MM
 
      blasArgs.beta = 0;
      MM3D<T,U,blasEngine>::Multiply(matrixA, matrixLI, matrixL, );
      // Then we update the next column
  }
}


template<typename T, typename U, template<typename, typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void TRSM3D<T,U,blasEngine>::iSolveUpper(
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixRI,
                       U localDimension,
                       U trueLocalDimension,
                       U bcDimension,
                       U globalDimension,
                       U trueGlobalDimension,
                       U matAstartX,
                       U matAendX,
                       U matAstartY,
                       U matAendY,
                       U matRstartX,
                       U matRendX,
                       U matRstartY,
                       U matRendY,
                       U matRIstartX,
                       U matRIendX,
                       U matRIstartY,
                       U matRIendY,
                       U transposePartner,
                       int MM_id,
                       MPI_Comm commWorld
                     )
{
  .. the size of the triangular inverses will be attained via trueLocalDimension and bcDimensio
  U numBlockRows = ...
  U numBlockColumns = ...
  blasEngineArgumentPackage_gemm<T> blasArgs;
  blasArgs.order = blasEngineOrder::AblasColumnMajor;
  blasArgs.transposeA = blasEngineTranspose::AblasTrans;
  blasArgs.transposeB = blasEngineTranspose::AblasNoTrans;
  blasArgs.alpha = 1.;
  blasArgs.beta = -1.;
  MM3D<T,U,blasEngine>::Multiply(matrixA, packedMatrix, matrixL, matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY,
      0, localShift, 0, localShift, matLstartX, matLstartX+localShift, matLstartY+localShift, matLendY, commWorld, blasArgs, true, false, true, MM_id);
  // Lets operate on individual columns at a time
  // Potential optimization 1): Don't use MM3D if the columns are too skinny in relation to the block size!
  //   Or this could just be taken care of when we tune block sizes?
  for (U i=0; i<numBlockRows; i++)
  {
      // Update the current column by accumulating the updates via MM
      if (i>0)
      {
        blasArgs.beta = -1;
      }
      // Solve via MM
 
      blasArgs.beta = 0;
      MM3D<T,U,blasEngine>::Multiply(matrixRI, matrixA, matrixR, );
      // Then we update the next column
  }
}


template<typename T, typename U, template<typename, typename> class blasEngine>
template<
  template<typename,typename,template<typename,typename,int> class> class StructureArg,
  template<typename,typename,int> class Distribution>
void TRSM3D<T,U,blasEngine>::transposeSwap(
											Matrix<T,U,StructureArg,Distribution>& mat,
											int myRank,
											int transposeRank,
											MPI_Comm commWorld
										     )
{
  if (myRank != transposeRank)
  {
    // Transfer with transpose rank
    MPI_Sendrecv_replace(mat.getRawData(), mat.getNumElems(), MPI_DOUBLE, transposeRank, 0, transposeRank, 0, commWorld, MPI_STATUS_IGNORE);

    // Note: the received data that now resides in mat is NOT transposed, and the Matrix structure is LowerTriangular
    //       This necesitates making the "else" processor serialize its data L11^{-1} from a square to a LowerTriangular,
    //       since we need to make sure that we call a MM::multiply routine with the same Structure, or else segfault.

  }
}


