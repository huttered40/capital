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

  std::vector<T>& dataA = matrixA.getVectorData(); 
  std::vector<T>& dataB = matrixB.getVectorData();
  U sizeA = matrixA.getNumElems();
  U sizeB = matrixB.getNumElems();
  std::vector<T> foreignA;
  std::vector<T> foreignB;

  bool isRootRow = ((pGridCoordX == pGridCoordZ) ? true : false);
  bool isRootColumn = ((pGridCoordY == pGridCoordZ) ? true : false);

  // Broadcast
  if (isRootRow)
  {
    MPI_Bcast(&dataA[0], sizeof(T)*sizeA, MPI_CHAR, pGridCoordZ, rowComm);
  }
  else
  {
    foreignA.resize(sizeA);
    MPI_Bcast(&foreignA[0], sizeof(T)*sizeA, MPI_CHAR, pGridCoordZ, rowComm);
  }

  // Broadcast data along columns
  if (isRootColumn)
  {
    MPI_Bcast(&dataB[0], sizeof(T)*sizeB, MPI_CHAR, pGridCoordZ, columnComm);
  }
  else
  {
    foreignB.resize(sizeB);
    MPI_Bcast(&foreignB[0], sizeof(T)*sizeB, MPI_CHAR, pGridCoordZ, columnComm);
  }

  // Now need to perform the cblas call via Summa3DEngine (to use the right cblas call based on the structure combo)
  // Need to call serialize blindly, even if we are going from square to square
  //   This is annoyingly required for cblas calls. For now, just abide by the rules.
  // We also must create an interface to serialize from vectors to vectors to avoid instantiating temporary matrices.
  // These can be made static methods in the Matrix class to the MatrixSerialize class.
  // Its just another option for the user.

  std::vector<T>& matrixAtoSerialize = isRootRow ? dataA : foreignA;
  std::vector<T>& matrixBtoSerialize = isRootColumn ? dataB : foreignB;
  std::vector<T> matrixAforEngine;
  std::vector<T> matrixBforEngine;

  // Now, the trouble here is that we want to make this generic, but we don't want to perform an extra copy/serialization if we don't have to
  // For example, if StructureA == MatrixStructureSquare, we don't want to perform any work, but we want this decision to be made
  // by the library in Serializer, not here. Well, I guess I just make the distinction here. Good enough

  if (!std::is_same<StructureA<T,U,Distribution>,MatrixStructureSquare<T,U,Distribution>>::value)		// compile time if statement. Branch prediction should be correct.
  {
    Serializer<T,U,StructureA,MatrixStructureSquare>::Serialize(matrixAtoSerialize, matrixAforEngine, dimensionX, dimensionY);
  }
  else
  {
    matrixAforEngine = std::move(matrixAtoSerialize);
  }
  if (!std::is_same<StructureA<T,U,Distribution>,MatrixStructureSquare<T,U,Distribution>>::value)
  {
    Serializer<T,U,StructureB,MatrixStructureSquare>::Serialize(matrixBtoSerialize, matrixBforEngine, dimensionY, dimensionZ);
  }
  else
  {
    matrixBforEngine = std::move(matrixBtoSerialize);
  }


  std::vector<T>& matrixCforEngine = matrixC.getVectorData();
  U numElems = matrixC.getNumElems();				// We assume that the user initialized matrixC correctly, even for TRMM

  switch (srcPackage.method)
  {
    case blasEngineMethod::AblasGemm:
    {
      blasEngine<T,U>::_gemm(&matrixAforEngine[0], &matrixBforEngine[0], &matrixCforEngine[0], dimensionX, dimensionY,
        dimensionX, dimensionZ, dimensionY, dimensionZ, 1., 1., dimensionY, dimensionX, dimensionY, srcPackage);
      break;
    }
    case blasEngineMethod::AblasTrmm:
    {
      const blasEngineArgumentPackage_trmm& blasArgs = static_cast<const blasEngineArgumentPackage_trmm&>(srcPackage);
      blasEngine<T,U>::_trmm(&matrixAforEngine[0], &matrixBforEngine[0], (blasArgs.side == blasEngineSide::AblasLeft ? dimensionX : dimensionY),
        (blasArgs.side == blasEngineSide::AblasLeft ? dimensionZ : dimensionX), 1., (blasArgs.side == blasEngineSide::AblasLeft ? dimensionY : dimensionX),
        (blasArgs.side == blasEngineSide::AblasLeft ? dimensionX : dimensionY), srcPackage);
      matrixCforEngine = std::move(matrixBforEngine);	// TRMM doesn't touch matrixC, so the user actually doesn't even have to allocate it, but we
                                                        //   move data into it before the AllReduce so the user gets back the solution in matrixC
      break;
    }
    default:
    {
      std::cout << "Invalid BLAS method used in blasEngineArgumentPackage\n";
      abort();
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &matrixCforEngine[0], sizeof(T)*numElems, MPI_CHAR, MPI_SUM, depthComm);

  // Unlike before when I had explicit new calls, the memory will get deleted automatically since the vectors will go out of scope
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
                                                              U matrixBcutZstart,
                                                              U matrixBcutZend,
                                                              U matrixBcutXstart,
                                                              U matrixBcutXend,
                                                              U matrixCcutZstart,
                                                              U matrixCcutZend,
                                                              U matrixCcutYstart,
                                                              U matrixCcutYend,
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

  U sizeA = matrixA.getNumElems(rangeA_x, rangeA_y);
  U sizeB = matrixB.getNumElems(rangeB_x, rangeB_z);
  std::vector<T> dataA(sizeA);
  std::vector<T> dataB(sizeB);

  // No clear way to prevent a needless copy if the cut dimensions of a matrix are full.
  std::vector<T>& matAsource = matrixA.getVectorData();
  Serializer<T,U,StructureA,StructureA>::Serialize(matAsource, dataA, matrixAcutXstart,
    matrixAcutXend, matrixAcutYstart, matrixAcutYend);
  // Now, dataA is set and ready to be communicated

  std::vector<T>& matBsource = matrixB.getVectorData();
  Serializer<T,U,StructureB,StructureB>::Serialize(matBsource, dataB, matrixBcutZstart,
    matrixBcutZend, matrixBcutXstart, matrixBcutXend);
  // Now, dataB is set and ready to be communicated

  std::vector<T> foreignA;
  std::vector<T> foreignB;

  bool isRootRow = ((pGridCoordX == pGridCoordZ) ? true : false);
  bool isRootColumn = ((pGridCoordY == pGridCoordZ) ? true : false);

  // Broadcast
  if (isRootRow)
  {
    MPI_Bcast(&dataA[0], sizeof(T)*sizeA, MPI_CHAR, pGridCoordZ, rowComm);
  }
  else
  {
    foreignA.resize(sizeA);
    MPI_Bcast(&foreignA[0], sizeof(T)*sizeA, MPI_CHAR, pGridCoordZ, rowComm);
  }

  // Broadcast data along columns
  if (isRootColumn)
  {
    MPI_Bcast(&dataB[0], sizeof(T)*sizeB, MPI_CHAR, pGridCoordZ, columnComm);
  }
  else
  {
    foreignB.resize(sizeB);
    MPI_Bcast(foreignB, sizeof(T)*sizeB, MPI_CHAR, pGridCoordZ, columnComm);
  }

  // Now need to perform the cblas call via Summa3DEngine (to use the right cblas call based on the structure combo)
  // Need to call serialize blindly, even if we are going from square to square
  //   This is annoyingly required for cblas calls. For now, just abide by the rules.
  // We also must create an interface to serialize from vectors to vectors to avoid instantiating temporary matrices.
  // These can be made static methods in the Matrix class to the MatrixSerialize class.
  // Its just another option for the user.

  std::vector<T>& matrixAtoSerialize = isRootRow ? dataA : foreignA;
  std::vector<T>& matrixBtoSerialize = isRootColumn ? dataB : foreignB;
  std::vector<T> matrixAforEngine;
  std::vector<T> matrixBforEngine;
  Serializer<T,U,StructureA,MatrixStructureSquare>::Serialize(matrixAtoSerialize, matrixAforEngine, 0, rangeA_x, 0, rangeA_y);
  Serializer<T,U,StructureB,MatrixStructureSquare>::Serialize(matrixBtoSerialize, matrixBforEngine, 0, rangeB_x, 0, rangeB_z);

  std::vector<T>& matrixCtoSerialize = matrixC.getVectorData();
  U numElems = matrixC.getNumElems();
  std::vector<T> matrixCforEngine;

  U rangeC_y = matrixCcutYend - matrixCcutYstart; 
  U rangeC_z = matrixCcutZend - matrixCcutZstart;
  Serializer<T,U,StructureC,MatrixStructureSquare>::Serialize(matrixCtoSerialize, matrixCforEngine,
    matrixCcutZstart, matrixCcutZend, matrixCcutYstart, matrixCcutYend);

  switch (srcPackage.method)
  {
    case blasEngineMethod::AblasGemm:
    {
      blasEngine<T,U>::_gemm(&matrixAforEngine[0], &matrixBforEngine[0], &matrixCforEngine[0], rangeA_x, rangeA_y,
        rangeB_x, rangeB_z, rangeC_y, rangeC_z, 1., 1., rangeA_y, rangeB_x, rangeC_y, srcPackage);
      break;
    }
    case blasEngineMethod::AblasTrmm:
    {
      const blasEngineArgumentPackage_trmm& blasArgs = static_cast<const blasEngineArgumentPackage_trmm&>(srcPackage);
      blasEngine<T,U>::_trmm(&matrixAforEngine[0], &matrixBforEngine[0], (blasArgs.side == blasEngineSide::AblasLeft ? rangeB_x : rangeA_y),
        (blasArgs.side == blasEngineSide::AblasLeft ? rangeB_z : rangeA_x), 1., (blasArgs.side == blasEngineSide::AblasLeft ? rangeA_y : rangeB_x),
        (blasArgs.side == blasEngineSide::AblasLeft ? rangeB_x : rangeA_y), srcPackage);
      matrixCforEngine = std::move(matrixBforEngine);			// could be wrong. Might need to be matrixA??
      break;
    }
    default:
    {
      std::cout << "Bad BLAS method used in blasEngineArgumentPackage\n";
      abort();
    }
  }

  MPI_Allreduce(MPI_IN_PLACE, &matrixCforEngine[0], sizeof(T)*numElems, MPI_CHAR, MPI_SUM, depthComm);

  std::vector<T>& matrixCsrc = matrixC.getVectorData();
  // reverse serialize, to put the solved piece of matrixC into where it should go.
  Serializer<T,U,MatrixStructureSquare,StructureC>::Serialize(matrixCforEngine, matrixCsrc,
    matrixCcutZstart, matrixCcutZend, matrixCcutYstart, matrixCcutYend);
  
}
