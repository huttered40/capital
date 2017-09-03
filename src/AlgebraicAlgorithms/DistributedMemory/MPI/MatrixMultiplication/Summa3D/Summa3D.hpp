/* Author: Edward Hutter */


template<typename T, typename U,
  template<typename,typename, template<typename,typename,int> class> class StructureA,
  template<typename,typename, template<typename,typename,int> class> class StructureB,
  template<typename,typename, template<typename,typename,int> class> class StructureC,
  template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void Summa3D<T,U,StructureA,StructureB,StructureC,blasEngine>::Multiply(
                                                              Matrix<T,U,StructureA,Distribution>& matrixA,
                                                              Matrix<T,U,StructureB,Distribution>& matrixB,
                                                              Matrix<T,U,StructureC,Distribution>& matrixC,
                                                              U dimensionX,
                                                              U dimensionY,
                                                              U dimensionZ,
                                                              MPI_Comm commWorld,
                                                              const blasEngineArgumentPackage& srcPackage
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
  int infoA = 0;
  int infoB = 0;
  Serializer<T,U,StructureA,MatrixStructureSquare>::Serialize(matrixAtoSerialize, matrixAforEngine, dimensionX, dimensionY, infoA);
  Serializer<T,U,StructureB,MatrixStructureSquare>::Serialize(matrixBtoSerialize, matrixBforEngine, dimensionY, dimensionZ, infoB);

  // infoA and infoB have information as to what kind of memory allocations occured within Serializer

  T* matrixCforEngine = matrixC.getData();
  U numElems = matrixC.getNumElems();

  switch (srcPackage.method)
  {
    case blasEngineMethod::AblasGemm:
    {
      blasEngine<T,U>::_gemm(matrixAforEngine, matrixBforEngine, matrixCforEngine, dimensionX, dimensionY,
        dimensionX, dimensionZ, dimensionY, dimensionZ, 1., 1., dimensionY, dimensionX, dimensionY, srcPackage);
      break;
    }
    case blasEngineMethod::AblasTrmm:
    {
      const blasEngineArgumentPackage_trmm& blasArgs = static_cast<const blasEngineArgumentPackage_trmm&>(srcPackage);
      blasEngine<T,U>::_trmm(matrixAforEngine, matrixBforEngine, (blasArgs.side == blasEngineSide::AblasLeft ? dimensionX : dimensionY),
        (blasArgs.side == blasEngineSide::AblasLeft ? dimensionZ : dimensionX), 1., (blasArgs.side == blasEngineSide::AblasLeft ? dimensionY : dimensionX),
        (blasArgs.side == blasEngineSide::AblasLeft ? dimensionX : dimensionY), srcPackage);
      break;
    }
    default:
    {
      std::cout << "Bad BLAS method used in blasEngineArgumentPackage\n";
      abort();
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, matrixCforEngine, numElems, MPI_DOUBLE, MPI_SUM, depthComm);

  if (!foreignA)
  {
    delete[] foreignA;
  }
  if (!foreignB)
  {
    delete[] foreignB;
  }

  if (infoA == 2)
  {
    delete[] matrixAforEngine;
  }
  if (infoB == 2)
  {
    delete[] matrixBforEngine;
  }
}

template<typename T, typename U,
  template<typename,typename, template<typename,typename,int> class> class StructureA,
  template<typename,typename, template<typename,typename,int> class> class StructureB,
  template<typename,typename, template<typename,typename,int> class> class StructureC,
  template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void Summa3D<T,U,StructureA,StructureB,StructureC,blasEngine>::Multiply(
                                                              Matrix<T,U,StructureA,Distribution>& matrixA,
                                                              Matrix<T,U,StructureB,Distribution>& matrixB,
                                                              Matrix<T,U,StructureC,Distribution>& matrixC,
                                                              U matrixAcutXstart,
                                                              U matrixAcutXend,
                                                              U matrixAcutYstart,
                                                              U matrixAcutYend,
                                                              U matrixBcutXstart,
                                                              U matrixBcutXend,
                                                              U matrixBcutZstart,
                                                              U matrixBcutZend,
                                                              U matrixCcutYstart,
                                                              U matrixCcutYend,
                                                              U matrixCcutZstart,
                                                              U matrixCcutZend,
                                                              MPI_Comm commWorld,
                                                              const blasEngineArgumentPackage& srcPackage
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

  U rangeA_x = matrixAcutXend-matrixAcutXstart;
  U rangeA_y = matrixAcutYend-matrixAcutYstart;
  U rangeB_x = matrixBcutXend-matrixBcutXstart;
  U rangeB_z = matrixBcutZend-matrixBcutZstart;

  T* dataA;
  T* dataB;
  U sizeA = matrixA.getNumElems(rangeA_x, rangeA_y);
  U sizeB = matrixB.getNumElems(rangeB_x, rangeB_z);

  // No clear way to prevent a needless copy if the cut dimensions of a matrix are full.
  dataA = new T[sizeA];
  T* matAsource = matrixA.getData();
  int infoA1 = 0;
  Serializer<T,U,StructureA,StructureA>::Serialize(matAsource, dataA, matrixAcutXstart,
    matrixAcutXend, matrixAcutYstart, matrixAcutYend, infoA1);
  // Now, dataA is set and ready to be communicated

  dataB = new T[sizeB];
  T* matBsource = matrixB.getData();
  int infoB1 = 0;
  Serializer<T,U,StructureB,StructureB>::Serialize(matBsource, dataB, matrixBcutXstart,
    matrixBcutXend, matrixBcutZstart, matrixBcutZend, infoB1);
  // Now, dataB is set and ready to be communicated

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
  int infoA2 = 0;
  int infoB2 = 0;
  Serializer<T,U,StructureA,MatrixStructureSquare>::Serialize(matrixAtoSerialize, matrixAforEngine, 0, rangeA_x, 0, rangeA_y, infoA2);
  Serializer<T,U,StructureB,MatrixStructureSquare>::Serialize(matrixBtoSerialize, matrixBforEngine, 0, rangeB_x, 0, rangeB_z, infoB2);

  T* matrixCforEngine = matrixC.getData();
  U numElems = matrixC.getNumElems();

  U rangeC_y = matrixCcutYend - matrixCcutYstart; 
  U rangeC_z = matrixCcutZend - matrixCcutZstart;

  switch (srcPackage.method)
  {
    case blasEngineMethod::AblasGemm:
    {
      blasEngine<T,U>::_gemm(matrixAforEngine, matrixBforEngine, matrixCforEngine, rangeA_x, rangeA_y,
        rangeB_x, rangeB_z, rangeC_y, rangeC_z, 1., 1., rangeA_y, rangeB_x, rangeC_y, srcPackage);
      break;
    }
    case blasEngineMethod::AblasTrmm:
    {
      const blasEngineArgumentPackage_trmm& blasArgs = static_cast<const blasEngineArgumentPackage_trmm&>(srcPackage);
      blasEngine<T,U>::_trmm(matrixAforEngine, matrixBforEngine, (blasArgs.side == blasEngineSide::AblasLeft ? rangeB_x : rangeA_y),
        (blasArgs.side == blasEngineSide::AblasLeft ? rangeB_z : rangeA_x), 1., (blasArgs.side == blasEngineSide::AblasLeft ? rangeA_y : rangeB_x),
        (blasArgs.side == blasEngineSide::AblasLeft ? rangeB_x : rangeA_y), srcPackage);
      break;
    }
    default:
    {
      std::cout << "Bad BLAS method used in blasEngineArgumentPackage\n";
      abort();
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, matrixCforEngine, numElems, MPI_DOUBLE, MPI_SUM, depthComm);

  if (!foreignA)
  {
    delete[] foreignA;
  }
  if (!foreignB)
  {
    delete[] foreignB;
  }
  if (infoA1 == 2)
  {
    delete[] dataA;
  }
  if (infoB1 == 2)
  {
    delete[] dataB;
  }
  if (infoA2 == 2)
  {
    delete[] matrixAforEngine;
  }
  if (infoB2 == 2)
  {
    delete[] matrixBforEngine;
  }
}
