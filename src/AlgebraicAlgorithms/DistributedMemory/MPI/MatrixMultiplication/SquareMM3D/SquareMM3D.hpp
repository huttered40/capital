/* Author: Edward Hutter */


template<typename T, typename U,
  template<typename,typename, template<typename,typename,int> class> class StructureA,
  template<typename,typename, template<typename,typename,int> class> class StructureB,
  template<typename,typename, template<typename,typename,int> class> class StructureC,
  template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void SquareMM3D<T,U,StructureA,StructureB,StructureC,blasEngine>::Multiply(
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

  // Need to call serialize blindly, even if we are going from square to square
  //   This is annoyingly required for cblas calls. For now, just abide by the rules.
  // We also must create an interface to serialize from vectors to vectors to avoid instantiating temporary matrices.
  // These can be made static methods in the Matrix class to the MatrixSerialize class.
  // Its just another option for the user.

  // Based on whether matrices matrixA and matrixB have square structure or not, we will use a pointer to a T or a vector of T
  //   In order to utilize the Serializer interface, we need to work with vectors. This gives the method the ability to
  //   resize the vectors to the appropriate size, which is not obvious to this algorithm due to the template structure. If I used a T*
  //   for the Serializer interface, then the Serializer might have needed to dynamically allocate memory for the reasons just described,
  //   but then the user of Serialize would have to explicitely delete that memory that it wasn't even sure was allocated or how much memory
  //   was allocated. Since the Serializer work with static class methods, saing state to call some sort of Serializer delete is
  //   not possible. Therefore, using a binary option for vectors of pointers to avod the copy if the Structue is square is the cheapest way I can
  //   think of. The only downside is two unused vectors/pointers on the function stack. Seems like a price worth paying.
  // Also, to be clear, using a vector if we don't need to serialize only has two bad choices: copy that whole thing into the vector, or move the whole
  //   thing into the vector at the risk of destroying the data in the original matrix parameter, causing terrible conseqences for the caller of this algorithm.

  T* matrixAforEnginePtr = nullptr;
  T* matrixBforEnginePtr = nullptr;

  // Now, the trouble here is that we want to make this generic, but we don't want to perform an extra copy/serialization if we don't have to
  // For example, if StructureA == MatrixStructureSquare, we don't want to perform any work, but we want this decision to be made
  // by the library in Serializer, not here. Well, I guess I just make the distinction here. Good enough

  if (!std::is_same<StructureA<T,U,Distribution>,MatrixStructureSquare<T,U,Distribution>>::value)		// compile time if statement. Branch prediction should be correct.
  {
    // Using the getters from local matrixA, even if it isn't the data that is being stored for this matrix, should still be ok.
    Matrix<T,U,StructureA,Distribution> matrixAforEngine(std::vector<T>(), matrixA.getNumColumnsLocal(), matrixA.getNumRowsLocal(),
      matrixA.getNumColumnsGlobal(), matrixA.getNumRowsGlobal());
    if (!isRootRow)
    {
      Matrix<T,U,StructureA,Distribution> matrixAtoSerialize(std::move(foreignA), matrixA.getNumColumnsLocal(), matrixA.getNumRowsLocal(),
        matrixA.getNumColumnsGlobal(), matrixA.getNumRowsGlobal());
      Serializer<T,U,StructureA,MatrixStructureSquare>::Serialize(matrixAtoSerialize, matrixAforEngine);
    }
    else
    {
      Serializer<T,U,StructureA,MatrixStructureSquare>::Serialize(matrixA, matrixAforEngine);
    }

    matrixAforEnginePtr = matrixAforEngine.getRawData();
  }
  else
  {
    matrixAforEnginePtr = &(isRootRow ? dataA : foreignA)[0];
  }
  if (!std::is_same<StructureB<T,U,Distribution>,MatrixStructureSquare<T,U,Distribution>>::value)
  {
    // Using the getters from local matrixA, even if it isn't the data that is being stored for this matrix, should still be ok.
    Matrix<T,U,StructureB,Distribution> matrixBforEngine(std::vector<T>(), matrixB.getNumColumnsLocal(), matrixB.getNumRowsLocal(),
      matrixB.getNumColumnsGlobal(), matrixB.getNumRowsGlobal());
    if (isRootColumn)
    {
      Matrix<T,U,StructureB,Distribution> matrixBtoSerialize(std::move(foreignB), matrixB.getNumColumnsLocal(), matrixB.getNumRowsLocal(),
        matrixB.getNumColumnsGlobal(), matrixB.getNumRowsGlobal());
      Serializer<T,U,StructureB,MatrixStructureSquare>::Serialize(matrixBtoSerialize, matrixBforEngine);
    }
    else
    {
      Serializer<T,U,StructureB,MatrixStructureSquare>::Serialize(matrixB, matrixBforEngine);
    }
    matrixBforEnginePtr = matrixBforEngine.getRawData();
  }
  else
  {
    matrixBforEnginePtr = &(isRootColumn ? dataB : foreignB)[0];
  }


  // I guess we are assuming that matrixC has Square Structure and not Triangular? For now, fine.
  std::vector<T>& matrixCforEngine = matrixC.getVectorData();
  U numElems = matrixC.getNumElems();				// We assume that the user initialized matrixC correctly, even for TRMM

  // Does C need to be Square? Big gaping hole in this algorithm right now that will come back to bite us later.

  switch (srcPackage.method)
  {
    case blasEngineMethod::AblasGemm:
    {
      blasEngine<T,U>::_gemm(matrixAforEnginePtr, matrixBforEnginePtr, &matrixCforEngine[0], dimensionX, dimensionY,
        dimensionX, dimensionZ, dimensionY, dimensionZ, 1., 1., dimensionY, dimensionX, dimensionY, srcPackage);
      break;
    }
    case blasEngineMethod::AblasTrmm:
    {
      const blasEngineArgumentPackage_trmm& blasArgs = static_cast<const blasEngineArgumentPackage_trmm&>(srcPackage);
      blasEngine<T,U>::_trmm(matrixAforEnginePtr, matrixBforEnginePtr, (blasArgs.side == blasEngineSide::AblasLeft ? dimensionX : dimensionY),
        (blasArgs.side == blasEngineSide::AblasLeft ? dimensionZ : dimensionX), 1., (blasArgs.side == blasEngineSide::AblasLeft ? dimensionY : dimensionX),
        (blasArgs.side == blasEngineSide::AblasLeft ? dimensionX : dimensionY), srcPackage);

      // Note: the below statement is awkward and should be changed later.
      // TRMM doesn't touch matrixC, so the user actually doesn't even have to allocate it, but we
      //   move data into it before the AllReduce so the user gets back the solution in matrixC
      
      // For now, just bite the bullet and incur the copy.
      memcpy(&matrixCforEngine[0], matrixBforEnginePtr, sizeof(T)*numElems);
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
void SquareMM3D<T,U,StructureA,StructureB,StructureC,blasEngine>::Multiply(
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
                                                              const blasEngineArgumentPackage& srcPackage,
                                                              bool cutA,
                                                              bool cutB,
                                                              bool cutC
                                                            )
{
  // We will set up 3 matrices and call the method above.

  U rangeA_x = matrixAcutXend-matrixAcutXstart;
  U rangeA_y = matrixAcutYend-matrixAcutYstart;
  U rangeB_x = matrixBcutXend-matrixBcutXstart;
  U rangeB_z = matrixBcutZend-matrixBcutZstart;
  U rangeC_y = matrixCcutYend - matrixCcutYstart; 
  U rangeC_z = matrixCcutZend - matrixCcutZstart;
  U globalDiffA = matrixA.getNumRowsGlobal() / matrixA.getNumRowsLocal();		// picked rows arbitrarily
  U globalDiffB = matrixB.getNumRowsGlobal() / matrixB.getNumRowsLocal();		// picked rows arbitrarily
  U globalDiffC = matrixC.getNumRowsGlobal() / matrixC.getNumRowsLocal();		// picked rows arbitrarily

  U sizeA = matrixA.getNumElems(rangeA_x, rangeA_y);
  U sizeB = matrixB.getNumElems(rangeB_x, rangeB_z);
  U sizeC = matrixC.getNumElems(rangeC_y, rangeC_z);

  // No clear way to prevent a needless copy if the cut dimensions of a matrix are full.

  // For now, matrixA, matrixB, and matrixC MUST be square. Or else compiler error. -- wait, why? That means they could
    // communicate more data than necessary, say if we have a triangular matrix, why would we want to serialize that
    // into square before MM? Because we immediately broadcast the data, THEN we worry about using square buffers for
    // BLAS routines. DONT WORRY ABOUT THAT HERE!

  // To fix scope problems with if/else, use cheap pointers with outside scope
    //  that can point to local structures, so no expensive dereference here (But I should check on this locality)

  Matrix<T,U,StructureA,Distribution>* ptrA;
  Matrix<T,U,StructureA,Distribution>* ptrB;
  Matrix<T,U,StructureA,Distribution>* ptrC;

  if (cutA)
  {
    Matrix<T,U,StructureA,Distribution> matrixAtoSerialize(std::vector<T>(), rangeA_x, rangeA_y, rangeA_x*globalDiffA, rangeA_y*globalDiffA);
    Serializer<T,U,StructureA,StructureA>::Serialize(matrixA, matrixAtoSerialize, matrixAcutXstart,
      matrixAcutXend, matrixAcutYstart, matrixAcutYend);
    ptrA = &matrixAtoSerialize;
  }
  else
  {
    ptrA = &matrixA;			// Should be cheap, but verify!
  }

  if (cutB)
  {
    Matrix<T,U,StructureB,Distribution> matrixBtoSerialize(std::vector<T>(), rangeB_z, rangeB_x, rangeB_z*globalDiffB, rangeB_x*globalDiffB);
    Serializer<T,U,StructureB,StructureB>::Serialize(matrixB, matrixBtoSerialize, matrixBcutZstart,
      matrixBcutZend, matrixBcutXstart, matrixBcutXend);
    ptrB = &matrixBtoSerialize;			// Should be cheap, but verify!
  }
  else
  {
    ptrB = &matrixB;
  }

  if (cutC)
  {
    Matrix<T,U,StructureC,Distribution> matrixCtoSerialize(std::vector<T>(), rangeC_z, rangeC_y, rangeC_z*globalDiffC, rangeC_y*globalDiffC);
    Serializer<T,U,StructureC,StructureC>::Serialize(matrixC, matrixCtoSerialize,
      matrixCcutZstart, matrixCcutZend, matrixCcutYstart, matrixCcutYend);
    ptrC = &matrixCtoSerialize;			// Should be cheap, but verify!
  }
  else
  {
    ptrC = &matrixC;
  }

  // Call the SquareMM3D method
  Multiply(*ptrA, *ptrB, *ptrC, rangeA_y, rangeA_x, rangeB_x, commWorld, srcPackage);

  //Serialize the correct small matrixC into the big matrix C that is the parameter to this method

  // reverse serialize, to put the solved piece of matrixC into where it should go.
  // QUESTION: Why do we want matrixC to have Square Structure? Why? Might come back to bite us later. Look into this
  Serializer<T,U,MatrixStructureSquare,StructureC>::Serialize(*ptrC, matrixC,
    matrixCcutZstart, matrixCcutZend, matrixCcutYstart, matrixCcutYend, true);
  
}
