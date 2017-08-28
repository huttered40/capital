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
  U bcDimension = dimension/helper;
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

  U shift = (localDimension>>1);
  rFactor(matrixA, matrixL, matrixLI, shift, bcDimension, (globalDimension>>1),
    matAstartX, matAstartX+shift, matAstartY, matAstartY+shift,
    matLstartX, matLstartX+shift, matLstartY, matLstartY+shift,
    matLIstartX, matLIstartX+shift, matLIstartY, matLIstartY+shift, transposePartner, commWorld);

  T* transposeData;
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);	// use MPI_COMM_WORLD for this p2p communication, but could use a smaller communicator

  // Regardless of whether or not we don't need to communicate, we still need to serialize into a square buffer
  if (rank != transposePartner)
  {
    U triangleSize = ((shift*(shift+1))>>1);
    T* dest = new T[triangleSize];
    T* source = matrixLI.getData(); 
    int info1 = 0;
    Serializer<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular>::Serialize(source, dest, localDimension, localDimension,
      0, shift, 0, shift, info1);
 
    MPI_Sendrecv_replace(dest, triangleSize, MPI_DOUBLE, transposePartner, 0, transposePartner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
 
    // Serialize into square matrix
    transposeData = new T[shift*shift];
    int info2 = 0;
    Serializer<T,U,MatrixStructureLowerTriangular,MatrixStructureSquare>::Serialize(dest, transposeData, localDimension, localDimension, 0, shift, 0, shift, info2);
    delete[] dest;
  }
  else
  {
    // No communication necessary. Serialize into square matrix
    T* source = matrixLI.getData();
    transposeData = new T[shift*shift];
    int info3 = 0;
    Serializer<T,U,MatrixStructureSquare,MatrixStructureSquare>::Serialize(source, transposeData, localDimension, localDimension, 0, shift, 0, shift, info3);
  }

  // Note that we aim to "fill up" the top-left part of L and L^{-1} when this returns. 
  //Summa3D<...>::Multiply(...);

}
