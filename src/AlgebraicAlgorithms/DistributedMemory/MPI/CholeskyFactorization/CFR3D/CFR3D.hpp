/* Author: Edward Hutter */


template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
void CFR3D<T,U,MatrixStructureSquare,MatrixStructureSquare>::Factor(
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixL,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixLI,
  U dimension,								// Assume this dimension is of each local Matrix that each processor owns. Could change that later.
  MPI_Comm commWorld )
{
  // Need to split up the commWorld communicator into a 3D grid similar to Summa3D
  int rank,size;
  MPI_Comm_rank(commWorld, &rank);
  MPI_Comm_size(commWorld, &size);

  int pGridDimensionSize = ceil(pow(size,1./3.));
  int helper = pGridDimensionSize;
  helper *= helper;
  int pGridCoordX = rank%pGridDimensionSize;
  int pGridCoordY = (rank%helper)/pGridDimensionSize;
  int pGridCoordZ = rank/helper;
  int transposePartner = pGridCoordZ*helper + pGridCoordX*pGridDimensionSize + pGridCoordY;

  U globalDimension = dimension*pGridDimensionSize;
  U bcDimension = globalDimension/(helper*helper);		// Can be tuned later.

  rFactor(matrixA, matrixL, matrixLI, dimension, bcDimension, globalDimension,
    0, dimension, 0, dimension, 0, dimension, 0, dimension, 0, dimension, 0, dimension, transposePartner, commWorld);
}

template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
void CFR3D<T,U,MatrixStructureSquare,MatrixStructureSquare>::rFactor(
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixL,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixLI,
  U localDimension,
  U bcDimension,
  U globalDimension,
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
  MPI_Comm commWorld )	// We want to pass in commWorld as MPI_COMM_WORLD because we want to pass that into 3D MM
{
  if (globalDimension == bcDimension)
  {
    std::cout << "Base case has been reached with " << globalDimension << " " << localDimension << " " << bcDimension << "\n";

    // First: AllGather matrix A so that every processor has the same replicated diagonal square partition of matrix A of dimension bcDimension
    //          Note that processors only want to communicate with those on their same 2D slice, since the matrices are replicated on every slice
    //          Note that before the AllGather, we need to serialize the matrix A into the small square matrix
    // Second: Data will be received in a blocked order due to AllGather semantics, which is not what we want. We need to get back to cyclic again
    //           This is an ugly process, as it was in the last code.
    // Third: Once data is in cyclic format, we call call sequential Cholesky Factorization and Triangular Inverse.
    // Fourth: Save the data that each processor owns according to the cyclic rule.

    int rank,size;
    MPI_Comm slice2D;
    MPI_Comm_rank(commWorld, &rank);
    MPI_Comm_size(commWorld, &size);

    int pGridDimensionSize = ceil(pow(size,1./3.));
    int helper = pGridDimensionSize;
    helper *= helper;
    int pGridCoordZ = rank/helper;

    // Attain the communicator with only processors on the same 2D slice
    MPI_Comm_split(commWorld, pGridCoordZ, rank, &slice2D);

    Matrix<T,U,MatrixStructureSquare,Distribution> baseCaseMatrixA(std::vector<T>(), localDimension, localDimension,
      globalDimension, globalDimension);
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(matrixA, baseCaseMatrixA, matAstartX,
      matAendX, matAstartY, matAendY);

    std::vector<T> blockedBaseCaseData(bcDimension*bcDimension);
    std::vector<T>cyclicBaseCasedata(bcDimension*bcDimension);
    MPI_Allgather(baseCaseMatrixA.getRawData(), sizeof(T)*baseCaseMatrixA.getNumElems(), MPI_CHAR,
      &blockedBaseCaseData[0], sizeof(T)*baseCaseMatrixA.getNumElems(), MPI_CHAR, slice2D);

    // Right now, we assume matrixA has Square Structure, if we want to let the user pass in just the unique part via a Triangular Structure,
    //   then we will need to change this.
    //   Note: this operation is just not cache efficient due to hopping around blockedBaseCaseData. Locality is not what we would like,
    //     but not sure it can really be improved here. Something to look into later.
    

    // Now, I want to use something similar to a template class for libraries conforming to the standards of LAPACK, such as FLAME.
    //   I want to be able to mix and match.
    
    return;
  }

  U localShift = (localDimension>>1);
  U globalShift = (globalDimension>>1);
  rFactor(matrixA, matrixL, matrixLI, localShift, bcDimension, globalShift,
    matAstartX, matAstartX+localShift, matAstartY, matAstartY+localShift,
    matLstartX, matLstartX+localShift, matLstartY, matLstartY+localShift,
    matLIstartX, matLIstartX+localShift, matLIstartY, matLIstartY+localShift, transposePartner, commWorld);

  int rank;
  // use MPI_COMM_WORLD for this p2p communication for transpose, but could use a smaller communicator
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Regardless of whether or not we don't need to communicate, we still need to serialize into a square buffer
  if (rank != transposePartner)
  {
    // Serialize the square matrix into the nonzero lower triangular (so avoid sending the zeros in the upper-triangular part of matrix)
    Matrix<T,U,MatrixStructureLowerTriangular,Distribution> packedMatrix(std::vector<T>(), localShift, localShift, globalShift, globalShift);
    // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
    Serializer<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular>::Serialize(matrixLI, packedMatrix,
      matLIstartX, matLIstartX+localShift, matLIstartY, matLIstartY+localShift);
 
    // Transfer with transpose rank
    MPI_Sendrecv_replace(packedMatrix.getRawData(), sizeof(T)*packedMatrix.getNumElems(), MPI_CHAR, transposePartner, 0, transposePartner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Note: the received data that now resides in packedMatrix is NOT transposed, and the Matrix structure is LowerTriangular
    //       This necesitates making the "else" processor serialize its data L11^{-1} from a square to a LowerTriangular,
    //       since we need to make sure that we call a MM::multiply routine with the same Structure, or else segfault.

    // Now, we want to be able to move this "raw, temporary, and anonymous" buffer into a Matrix structure, to be passed into
      // MatrixMultiplication at no cost. We definitely do not want to copy, we want to move!
    // But wait! That buffer is exactly what we need anyway
    // Wait again!! Why do we need to do this? The structure is already LowerTriangular. Just keep it and have the non-transpose processor
    //   serialize his Square matrix structure into a lower triangular

    // Call matrix multiplication that does not cut up matrix B in C <- AB
    //    Need to set up the struct that has useful BLAS info

    // I am using gemm right now, but I might want to use dtrtri or something due to B being triangular at heart
    blasEngineArgumentPackage_gemm<double> blasArgs;
    blasArgs.order = blasEngineOrder::AblasRowMajor;
    blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
    blasArgs.transposeB = blasEngineTranspose::AblasTrans;
    blasArgs.alpha = 1.;
    blasArgs.beta = 1.;
//    SquareMM3D<double,int,MatrixStructureSquare,MatrixStructureLowerTriangular,MatrixStructureSquare, cblasEngine>::
//      Multiply(matrixA, packedMatrix, matrixL, matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY,
//        0, localShift, 0, localShift, matLstartX, matLstartX+localShift, matLstartY+localShift, matLendY, MPI_COMM_WORLD, blasArgs, true, false, true);
  }
  else
  {
    // For processors that are their own transpose within the slice they are on in a 3D processor grid.
    // We want to serialize LI from Square into LowerTriangular so it can match the "transposed" processors that did it to send half the words for one reason
    Matrix<T,U,MatrixStructureLowerTriangular,Distribution> tempLI(std::vector<T>(), localShift, localShift, globalShift, globalShift);
    Serializer<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular>::Serialize(matrixLI, tempLI, matLIstartX,
      matLIstartX+localShift, matLIstartY, matLIstartY+localShift);

    // I am using gemm right now, but I might want to use dtrtri or something due to B being triangular at heart
    blasEngineArgumentPackage_gemm<double> blasArgs;
    blasArgs.order = blasEngineOrder::AblasRowMajor;
    blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
    blasArgs.transposeB = blasEngineTranspose::AblasTrans;
    blasArgs.alpha = 1.;
    blasArgs.beta = 1.;
//    SquareMM3D<double,int,MatrixStructureSquare,MatrixStructureLowerTriangular,MatrixStructureSquare, cblasEngine>::
//      Multiply(matrixA, tempLI, matrixL, matAstartX, matAstartX+localShift, matAstartY+localShift, matAendY,
//        0, localShift, 0, localShift, matLstartX, matLstartX+localShift, matLstartY+localShift, matLendY, MPI_COMM_WORLD, blasArgs, true, false, true);
  }

  // Note that we aim to "fill up" the bottom-left part of L when this returns.

}
