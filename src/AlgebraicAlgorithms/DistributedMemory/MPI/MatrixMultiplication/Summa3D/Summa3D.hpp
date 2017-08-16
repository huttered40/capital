/* Author: Edward Hutter */


template<typename T, typename U,
  template<typename,typename, template<typename,typename,int> class> class StructureA,
  template<typename,typename, template<typename,typename,int> class> class StructureB,
  template<typename,typename, template<typename,typename,int> class> class StructureC>
template<template<typename,typename,int> class Distribution>
void Summa3D<T,U,StructureA,StructureB,StructureC>::Multiply(
                                                              Matrix<T,U,StructureA,Distribution>& matrixA,
                                                              Matrix<T,U,StructureB,Distribution>& matrixB,
                                                              Matrix<T,U,StructureC,Distribution>& matrixC,
                                                              U dimensionX,
                                                              U dimensionY,
                                                              U dimensionZ,
                                                              int pGridCoordX,
                                                              int pGridCoordY,
                                                              int pGridCoordZ,
                                                              MPI_Comm commWorld
                                                            )
{
  T* dataA = matrixA.getData()[0]; 
  T* dataB = matrixB.getData()[0];
  U sizeA = matrixA.getNumElems();
  U sizeB = matrixB.getNumElems();
  T* foreignA;
  T* foreignB;

  int rank;
  MPI_Comm_rank(commWorld, &rank);
  MPI_Comm rowComm;
  MPI_Comm columnComm;
  MPI_Comm sliceComm;
  
  // First, split the 3D Cube processor grid communicator into groups based on what 2D slice they are located on.
  // Then, subdivide further into row groups and column groups
  MPI_Comm_split(commWorld, pGridCoordZ, rank, &sliceComm);
  MPI_Comm_split(sliceComm, pGridCoordY, pGridCoordX, &rowComm);
  MPI_Comm_split(sliceComm, pGridCoordX, pGridCoordY, &columnComm);

  // Broadcast
  if (pGridCoordX == pGridCoordZ)
  {
    MPI_Bcast(dataA, sizeA, MPI_DOUBLE, pGridCoordZ, rowComm);
  }
  else
  {
    foreignA = new T[sizeA];
    MPI_Bcast(foreignA, sizeA, MPI_DOUBLE, pGridCoordZ, rowComm);
  }

  // Broadcast data along columns
  if (pGridCoordX == pGridCoordZ)
  {
    MPI_Bcast(dataB, sizeB, MPI_DOUBLE, pGridCoordZ, columnComm);
  }
  else
  {
    foreignB = new T[sizeB];
    MPI_Bcast(foreignB, sizeB, MPI_DOUBLE, pGridCoordZ, columnComm);
  }
}

template<typename T, typename U,
  template<typename,typename, template<typename,typename,int> class> class StructureA,
  template<typename,typename, template<typename,typename,int> class> class StructureB,
  template<typename,typename, template<typename,typename,int> class> class StructureC>
template<template<typename,typename,int> class Distribution>
void Summa3D<T,U,StructureA,StructureB,StructureC>::Multiply(
                                                              Matrix<T,U,StructureA,Distribution>& matrixA,
                                                              Matrix<T,U,StructureB,Distribution>& matrixB,
                                                              Matrix<T,U,StructureC,Distribution>& matrixC,
                                                              U matrixAcutXstart,
                                                              U matrixAcutXend,
                                                              U matrixAcutYstart,
                                                              U matrixAcutYend,
                                                              U matrixBcutYstart,
                                                              U matrixBcutYend,
                                                              U matrixBcutZstart,
                                                              U matrixBcutZend,
                                                              U matrixCcutXstart,
                                                              U matrixCcutXend,
                                                              U matrixCcutZstart,
                                                              U matrixCcutZend,
                                                              int pGridCoordX,
                                                              int pGridCoordY,
                                                              int pGridCoordZ,
                                                              MPI_Comm commWorld
                                                            )
{
}
