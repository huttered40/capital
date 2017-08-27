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
                                                              MPI_Comm commWorld
                                                            )
{
  int rank,size;
  MPI_Comm_rank(commWorld, &rank);
  MPI_Comm_size(commWorld, &size);

  int pGridDimensionSize = ceil(pow(size,1./3.));
  int helper = pGridDimensionSize;
  helper *= helper;
  int pGridCoordX = rank%pGridDimensionSize;
  int pGridCoordY = (rank%helper)/pGridDimensionSize;
  int pGridCoordZ = rank/helper;

  MPI_Comm rowComm;
  MPI_Comm columnComm;
  MPI_Comm sliceComm;
  MPI_Comm depthComm;

  // First, split the 3D Cube processor grid communicator into groups based on what 2D slice they are located on.
  // Then, subdivide further into row groups and column groups
  MPI_Comm_split(commWorld, pGridCoordY*pGridDimensionSize+pGridCoordX, rank, &depthComm);
  MPI_Comm_split(commWorld, pGridCoordZ, rank, &sliceComm);
  MPI_Comm_split(sliceComm, pGridCoordY, pGridCoordX, &rowComm);
  MPI_Comm_split(sliceComm, pGridCoordX, pGridCoordY, &columnComm);

  T* dataA = matrixA.getData(); 
  T* dataB = matrixB.getData();
  U sizeA = matrixA.getNumElems();
  U sizeB = matrixB.getNumElems();
  T* foreignA = nullptr;
  T* foreignB = nullptr;

  bool isRootRow = ((pGridCoordX == pGridCoordZ) ? true : false);
  bool isRootColumn = ((pGridCoordY == pGridCoordZ) ? true : false);

  // Broadcast
  if (isRootRow)
  {
    MPI_Bcast(dataA, sizeA, MPI_DOUBLE, pGridCoordZ, rowComm);
  }
  else
  {
    foreignA = new T[sizeA];
    MPI_Bcast(foreignA, sizeA, MPI_DOUBLE, pGridCoordZ, rowComm);
  }

  // Broadcast data along columns
  if (isRootColumn)
  {
    MPI_Bcast(dataB, sizeB, MPI_DOUBLE, pGridCoordZ, columnComm);
  }
  else
  {
    foreignB = new T[sizeB];
    MPI_Bcast(foreignB, sizeB, MPI_DOUBLE, pGridCoordZ, columnComm);
  }

  // Now need to perform the cblas call via Summa3DEngine (to use the right cblas call based on the structure combo)
  // Need to call serialize blindly, even if we are going from square to square
  //   This is annoyingly required for cblas calls. For now, just abide by the rules.
  // We also must create an interface to serialize from vectors to vectors to avoid instantiating temporary matrices.
  // These can be made static methods in the Matrix class to the MatrixSerialize class.
  // Its just another option for the user.

  T* matrixAtoSerialize = isRootRow ? dataA : foreignA;
  T* matrixBtoSerialize = isRootColumn ? dataB : foreignB;
  T* matrixAforEngine = nullptr;
  T* matrixBforEngine = nullptr;
  Serializer<T,U,StructureA,MatrixStructureSquare>::Serialize(matrixAtoSerialize, matrixAforEngine, dimensionX, dimensionY);
  Serializer<T,U,StructureB,MatrixStructureSquare>::Serialize(matrixBtoSerialize, matrixBforEngine, dimensionY, dimensionZ);

  T* matrixCforEngine = matrixC.getData();
  U numElems = matrixC.getNumElems();

  blasEngine<T,U,StructureA, StructureB, StructureC>::multiply(matrixAforEngine, matrixBforEngine, matrixCforEngine, dimensionX, dimensionY,
    dimensionX, dimensionZ, dimensionY, dimensionZ);

  MPI_Allreduce(MPI_IN_PLACE, matrixCforEngine, numElems, MPI_DOUBLE, MPI_SUM, depthComm);

  if (!foreignA)
  {
    delete[] foreignA;
  }
  if (!foreignB)
  {
    delete[] foreignB;
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
                                                              MPI_Comm commWorld
                                                            )
{
  int rank,size;
  MPI_Comm_rank(commWorld, &rank);
  MPI_Comm_size(commWorld, &size);

  int pGridDimensionSize = ceil(pow(size,1./3.));
  int helper = pGridDimensionSize;
  helper *= helper;
  int pGridCoordX = rank%pGridDimensionSize;
  int pGridCoordY = (rank%helper)/pGridDimensionSize;
  int pGridCoordZ = rank/helper;

  MPI_Comm rowComm;
  MPI_Comm columnComm;
  MPI_Comm sliceComm;
  MPI_Comm depthComm;

  // First, split the 3D Cube processor grid communicator into groups based on what 2D slice they are located on.
  // Then, subdivide further into row groups and column groups
  MPI_Comm_split(commWorld, pGridCoordY*pGridDimensionSize+pGridCoordX, rank, &depthComm);
  MPI_Comm_split(commWorld, pGridCoordZ, rank, &sliceComm);
  MPI_Comm_split(sliceComm, pGridCoordY, pGridCoordX, &rowComm);
  MPI_Comm_split(sliceComm, pGridCoordX, pGridCoordY, &columnComm);

  U rangeA_x = matrixAcutXend-matrixAcutXstart
  U rangeA_y = matrixAcutYend-matrixAcutYstart
  U rangeB_x = matrixBcutXend-matrixBcutXstart
  U rangeB_y = matrixBcutYend-matrixBcutYstart

  T* dataA;
  T* dataB;
  U sizeA = matrixA.getNumElems();
  U sizeB = matrixB.getNumElems();
  bool wholeA = false;
  bool wholeB = false;  

  // To avoid needless serialization if matrix is whole, we perform simple check
  U temp = rangeA_x * rangeA_y;
  if (temp == sizeA)
  {
    dataA = matrixA.getData();
  }
  else
  {
    wholeA = true;
    dataA = new T[temp];
    T* matAsource = matrixA.getData();
    Serializer<T,U,StructureA,StructureA>::Serialize(matAsource, dataA, matrixAcutXstart,
      matrixAcutXend, matrixAcutYstart, matrixAcutYend);
    // Now, dataA is set and ready to be communicated
  }
  temp = (matrixBcutXend-matrixBcutXstart);
  temp *= (matrixBcutYend-matrixBcutYstart);
  if (temp == sizeB)
  {
    dataB = matrixB.getData();
  }
  else
  {
    wholeB = true;
    dataB = new T[temp];
    T* matBsource = matrixB.getData();
    Serializer<T,U,StructureB,StructureB>::Serialize(matBsource, dataB, matrixBcutXstart,
      matrixBcutXend, matrixBcutYstart, matrixBcutYend);
    // Now, dataB is set and ready to be communicated
  }

  T* foreignA = nullptr;
  T* foreignB = nullptr;

  bool isRootRow = ((pGridCoordX == pGridCoordZ) ? true : false);
  bool isRootColumn = ((pGridCoordY == pGridCoordZ) ? true : false);

  // Broadcast
  if (isRootRow)
  {
    MPI_Bcast(dataA, sizeA, MPI_DOUBLE, pGridCoordZ, rowComm);
  }
  else
  {
    foreignA = new T[sizeA];
    MPI_Bcast(foreignA, sizeA, MPI_DOUBLE, pGridCoordZ, rowComm);
  }

  // Broadcast data along columns
  if (isRootColumn)
  {
    MPI_Bcast(dataB, sizeB, MPI_DOUBLE, pGridCoordZ, columnComm);
  }
  else
  {
    foreignB = new T[sizeB];
    MPI_Bcast(foreignB, sizeB, MPI_DOUBLE, pGridCoordZ, columnComm);
  }

  // Now need to perform the cblas call via Summa3DEngine (to use the right cblas call based on the structure combo)
  // Need to call serialize blindly, even if we are going from square to square
  //   This is annoyingly required for cblas calls. For now, just abide by the rules.
  // We also must create an interface to serialize from vectors to vectors to avoid instantiating temporary matrices.
  // These can be made static methods in the Matrix class to the MatrixSerialize class.
  // Its just another option for the user.

  T* matrixAtoSerialize = isRootRow ? dataA : foreignA;
  T* matrixBtoSerialize = isRootColumn ? dataB : foreignB;
  T* matrixAforEngine = nullptr;
  T* matrixBforEngine = nullptr;
  Serializer<T,U,StructureA,MatrixStructureSquare>::Serialize(matrixAtoSerialize, matrixAforEngine, dimensionX, dimensionY);
  Serializer<T,U,StructureB,MatrixStructureSquare>::Serialize(matrixBtoSerialize, matrixBforEngine, dimensionY, dimensionZ);

  T* matrixCforEngine = matrixC.getData();
  U numElems = matrixC.getNumElems();

  blasEngine<T,U,StructureA, StructureB, StructureC>::multiply(matrixAforEngine, matrixBforEngine, matrixCforEngine, dimensionX, dimensionY,
    dimensionX, dimensionZ, dimensionY, dimensionZ);

  MPI_Allreduce(MPI_IN_PLACE, matrixCforEngine, numElems, MPI_DOUBLE, MPI_SUM, depthComm);

  if (!foreignA)
  {
    delete[] foreignA;
  }
  if (!foreignB)
  {
    delete[] foreignB;
  }

  if (wholeA)
  {
    delete[] dataA;
  }
  if (wholeB)
  {
    delete[] dataB;
  }
}
