/* Author: Edward Hutter */

template<typename T, typename U, template<typename,typename> class blasEngine>
template<
          template<typename,typename, template<typename,typename,int> class> class Structure,
          template<typename,typename,int> class Distribution
        >
T MMvalidate<T,U,blasEngine>::validateLocal(
                        Matrix<T,U,Structure,Distribution>& matrixSol,
                        U localDimensionX,
                        U localDimensionY,
                        U localDimensionZ,
                        U globalDimensionX,
                        U globalDimensionY,
                        U globalDimensionZ,
                        MPI_Comm comm,
                        const blasEngineArgumentPackage& srcPackage
                      )
{
  // What I want to do here is generate a full matrix with the correct values
  //   and then compare with the local part of matrixSol.
  //   Finally, we can AllReduce the residuals.

  // Best way: reuse existing correct code : use MatrixDistributer, so we should create a Matrix instance.

  using globalMatrixType = Matrix<T,U,Structure,Distribution>;
  globalMatrixType globalMatrixA(globalDimensionX,globalDimensionY,globalDimensionY,globalDimensionZ);
  globalMatrixType globalMatrixB(globalDimensionZ,globalDimensionX,globalDimensionY,globalDimensionZ);
  globalMatrixA.DistributeRandom(1, 1, 1, 1);		// Hardcode so that the Distributer thinks we own the entire matrix.
  globalMatrixB.DistributeRandom(1, 1, 1, 1);		// Hardcode so that the Distributer thinks we own the entire matrix.

  // No need to serialize this data. It is already in proper format for a call to BLAS
  std::vector<T> matrixAforEngine = globalMatrixA.getVectorData();
  std::vector<T> matrixBforEngine = globalMatrixB.getVectorData();
  std::vector<T> matrixCforEngine(globalDimensionY*globalDimensionZ, 0);	// No matrix needed for this. Only used in BLAS call

  switch (srcPackage.method)
  {
    case blasEngineMethod::AblasGemm:
    {
      blasEngine<T,U>::_gemm(&matrixAforEngine[0], &matrixBforEngine[0], &matrixCforEngine[0], globalDimensionX, globalDimensionY,
        globalDimensionX, globalDimensionZ, globalDimensionY, globalDimensionZ, 1., 1., globalDimensionY, globalDimensionX, globalDimensionY, srcPackage);
      break;
    }
    case blasEngineMethod::AblasTrmm:
    {
      const blasEngineArgumentPackage_trmm& blasArgs = static_cast<const blasEngineArgumentPackage_trmm&>(srcPackage);
      blasEngine<T,U>::_trmm(&matrixAforEngine[0], &matrixBforEngine[0], (blasArgs.side == blasEngineSide::AblasLeft ? globalDimensionX : globalDimensionY),
        (blasArgs.side == blasEngineSide::AblasLeft ? globalDimensionZ : globalDimensionX), 1., (blasArgs.side == blasEngineSide::AblasLeft ? globalDimensionY : globalDimensionX)
,
        (blasArgs.side == blasEngineSide::AblasLeft ? globalDimensionX : globalDimensionY), srcPackage);
      break;
    }
    default:
    {
      std::cout << "Bad BLAS method used in blasEngineArgumentPackage\n";
      abort();
    }
  }

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.
/*
  T error = 0;
  for (U i=0; i<...; i++)
  {
    for (U j=0; j<...; j++)
    {
      
    }
  }
*/

  // Now, we need the AllReduce of the error. Very cheap operation in terms of bandwidth cost, since we are only communicating
  //   a single double primitive type.
/*
  int rank,size,provided;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // size -- total number of processors in the 3D grid

  int pGridDimensionSize = ceil(pow(size,1./3.));
  
  int helper = pGridDimensionSize;
  helper *= helper;
  int pCoordX = rank%pGridDimensionSize;
  int pCoordY = (rank%helper)/pGridDimensionSize;
  int pCoordZ = rank/helper;
*/

  return 0.;
}

template<typename T, typename U, template<typename,typename> class blasEngine>
template<
          template<typename,typename, template<typename,typename,int> class> class Structure,
          template<typename,typename,int> class Distribution
        >
T MMvalidate<T,U,blasEngine>::validateLocal(
                        Matrix<T,U,Structure,Distribution>& matrixSol,
                        U matrixSolcutYstart,
                        U matrixSolcutYend,
                        U matrixSolcutZstart,
                        U matrixSolcutZend,
                        U globalDimensionX,
                        U globalDimensionY,
                        U globalDimensionZ,
                        MPI_Comm comm,
                        const blasEngineArgumentPackage& srcPackage
                      )
{
  // Note: we will generate the entire global matrix on each layer for each processor for now, but this involves extra work that is not necessary.

/*
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

  // Now need to perform the cblas call via Summa3DEngine (to use the right cblas call based on the structure combo)
  // Need to call serialize blindly, even if we are going from square to square
  //   This is annoyingly required for cblas calls. For now, just abide by the rules.
  // We also must create an interface to serialize from vectors to vectors to avoid instantiating temporary matrices.
  // These can be made static methods in the Matrix class to the MatrixSerialize class.
  // Its just another option for the user.

  T* matrixAtoSerialize = dataA;
  T* matrixBtoSerialize = dataB;
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

  // The BLAS call below needs modifed because we need to allow for transpose or triangular structure AND allow for dtrmm instead of dgemm
  blasEngine<T,U>::_gemm(matrixAforEngine, matrixBforEngine, matrixCforEngine, rangeA_x, rangeA_y,
    rangeB_x, rangeB_z, rangeC_y, rangeC_z, blasEngineInfo);
  // Assume for now that first 2 bits give 4 possibilies
  //   0 -> _gemm
  //   1 -> _trmm
  //   2 -> ?
  //   3 -> ?
  bool helper1 = blasEngineInfo&0x1;
  blasEngineInfo >>= 1;
  bool helper2 = blasEngineInfo&0x1;
  blasEngineInfo >>= 1;
  int whichRoutine = static_cast<int>(helper2)*2 + static_cast<int>(helper1);
  switch (whichRoutine)
  {
    case 0:
    {
      blasEngine<T,U>::_gemm(matrixAforEngine, matrixBforEngine, matrixCforEngine, rangeA_x, rangeA_y,
        rangeB_x, rangeB_z, rangeC_y, rangeC_z, 1., 1., rangeA_y, rangeB_x, rangeC_y, blasEngineInfo);
      break;
    }
    case 1:
    {
      int checkOrder = (0x2 & (blasEngineInfo>>1));		// check the 2nd bit to see if square matrix is on left or right
      blasEngine<T,U>::_trmm(matrixAforEngine, matrixBforEngine, (checkOrder ? rangeB_x : rangeA_y),
        (checkOrder ? rangeB_z : rangeA_x), 1., (checkOrder ? rangeA_y : rangeB_x), (checkOrder ? rangeB_x : rangeA_y), blasEngineInfo);
      break;
    }
    case 2:
    {
      break;
    }
    case 3:
    {
      break;
    }
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
*/
}
