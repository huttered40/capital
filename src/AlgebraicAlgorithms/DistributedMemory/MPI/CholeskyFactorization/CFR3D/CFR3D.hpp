/* Author: Edward Hutter */


template<typename T, typename U>
template<template<typename,typename,int> class Distribution>
void CFR3D<T,U,MatrixStructureSquare,MatrixStructureSquare>::Factor(
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixA,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixL,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixLI,
  U dimension,
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

  U bcDimension = dimension/helper;		// Can be tuned later.

  U globalDimension = dimension*pGridDimensionSize;

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
  MPI_Comm commWorld )
{
  if (globalDimension == bcDimension)
  {
    std::cout << "Base case has been reached with " << globalDimension << " " << localDimension << " " << bcDimension << "\n";
    return;
  }

  U localShift = (localDimension>>1);
  U globalShift = (globalDimension>>1);
  rFactor(matrixA, matrixL, matrixLI, shift, bcDimension, (globalDimension>>1),
    matAstartX, matAstartX+localShift, matAstartY, matAstartY+localShift,
    matLstartX, matLstartX+localShift, matLstartY, matLstartY+localShift,
    matLIstartX, matLIstartX+localShift, matLIstartY, matLIstartY+localShift, transposePartner, commWorld);

  // use a pointer so that we can assign it to the matrix received in the if/else statement, need that extra scope
    // and dont want to create an extra instance
  Matrix<T,U,MatrixStructureSquare,Distribution>* transposeData;
  int rank;

  // use MPI_COMM_WORLD for this p2p communication for transpose, but could use a smaller communicator
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Regardless of whether or not we don't need to communicate, we still need to serialize into a square buffer
  if (rank != transposePartner)
  {
    // Serialize the square matrix into the nonzero lower triangular (so avoid sending the zeros in the upper-triangular part of matrix)
    Matrix<T,U,MatrixStructureLowerTriangular,Distribution> packedMatrix(std::vector<T>(), localShift, localShift, globalShift, globalShift);
    Serializer<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular>::Serialize(matrixLI, packedMatrix,
      0, shift, 0, shift);
 
    // Transfer with transpose rank
    MPI_Sendrecv_replace(packedMatrix.getRawData(), tempMatrix.getNumElems(), sizeof(T)*MPI_CHAR, transposePartner, 0, transposePartner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

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
    blasEngineArgumentPackage_gemm blasArgs;
    blasArgs.order = blasEngineOrder::AblasRowMajor;
    blasArgs.transposeA = blasEngineTranspose::AblasNoTrans;
    blasArgs.transposeB = blasEngineTranspose::AblasTrans;
    SquareMM3D<double,int,MatrixStructureSquare,MatrixStructureLowerTriangular,MatrixStructureSquare, cblasEngine>::
      Multiply(matrixA, packedMatrix, matrixL, matrixAstartX, matrixAstartX+shift, matrixAstartY+shift, matrixAendY,
        0, shift, 0, shift, matrixLstartX, matrixLstartX+shift, matrixLstartY+shift, matrixLendY, MPI_COMM_WORLD, blasArgs, true, false, true);
    
  }
  else
  {
    // For processors that are their own transpose within the slice they are on in a 3D processor grid.
    // We want to serialize LI from Square into LowerTriangular so it can match the "transposed" processors that did it to send half the words for one reason


  }

  // Note that we aim to "fill up" the bottom-right part of L when this returns.
  // Fil up a BLAS struct
  SquareMM3D<T,U,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare>::Multiply(...overloaded...);

  // Get set for another SquareMM3D

  // Perform a subtraction

  // perform recursive call

  // Two more instances of SquareMM3D.
  // Also, we have a use case for adding the constant factor arguments to the BLAS enum struct.

}
