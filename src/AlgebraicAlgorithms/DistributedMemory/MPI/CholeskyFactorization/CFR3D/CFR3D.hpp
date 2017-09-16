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

  T* transposeData;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);	// use MPI_COMM_WORLD for this p2p communication, but could use a smaller communicator

  // Regardless of whether or not we don't need to communicate, we still need to serialize into a square buffer
  if (rank != transposePartner)
  {
    Matric<T,U,MatrixStructureLowerTriangular,Distribution> tempMatrix(std::vector<T>(), localShift, localShift, globalShift, globalShift);
    Serializer<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular>::Serialize(matrixLI, tempMatrix,
      0, shift, 0, shift);
 
    MPI_Sendrecv_replace(tempMatrix.getRawData(), tempMatrix.getNumElems(), sizeof(T)*MPI_CHAR, transposePartner, 0, transposePartner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // Serialize into square matrix

    Matric<T,U,MatrixStructureLowerTriangular,Distribution> CutMatrixL(std::vector<T>(), localShift, localShift, globalShift, globalShift);
    Serializer<T,U,MatrixStructureLowerTriangular,MatrixStructureSquare>::Serialize(tempMatrix, CutMatrixL, 0, shift, 0, shift);
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
