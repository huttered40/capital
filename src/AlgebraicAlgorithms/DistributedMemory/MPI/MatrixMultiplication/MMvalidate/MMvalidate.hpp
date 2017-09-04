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

  int rank,size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // size -- total number of processors in the 3D grid

  int pGridDimensionSize = ceil(pow(size,1./3.));
  
  int helper = pGridDimensionSize;
  helper *= helper;
  int pCoordX = rank%pGridDimensionSize;
  int pCoordY = (rank%helper)/pGridDimensionSize;
  int pCoordZ = rank/helper;

  MPI_Comm sliceComm;
  MPI_Comm_split(comm, pCoordZ, rank, &sliceComm);

  using globalMatrixType = Matrix<T,U,Structure,Distribution>;
  globalMatrixType globalMatrixA(globalDimensionX,globalDimensionY,globalDimensionX,globalDimensionY);
  globalMatrixType globalMatrixB(globalDimensionZ,globalDimensionX,globalDimensionZ,globalDimensionX);
  globalMatrixA.DistributeRandom(0, 0, 1, 1);		// Hardcode so that the Distributer thinks we own the entire matrix.
  globalMatrixB.DistributeRandom(0, 0, 1, 1);		// Hardcode so that the Distributer thinks we own the entire matrix.

  // No need to serialize this data. It is already in proper format for a call to BLAS
  std::vector<T> matrixAforEngine = globalMatrixA.getVectorData();
  std::vector<T> matrixBforEngine = globalMatrixB.getVectorData();
  std::vector<T> matrixCforEngine(globalDimensionY*globalDimensionZ, 0);	// No matrix needed for this. Only used in BLAS call

  switch (srcPackage.method)
  {
    case blasEngineMethod::AblasGemm:
    {
      // Later on, may want to resize matrixCforEngine in here to avoid needless memory allocation in TRMM cases
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
      matrixCforEngine = matrixBforEngine;// Dont move right now. Im worried about corrupting data in matrixB. Look into this!! std::move(matrixBforEngine;...
      break;
    }
    default:
    {
      std::cout << "Bad BLAS method used in blasEngineArgumentPackage\n";
      abort();
    }
  }

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.

  // Temporary code for quick n' dirty simulation of squae matrices

  T error = 0;
  std::vector<T>& tester = matrixSol.getVectorData();
  U countTester = 0;
  U countRef = pCoordX + pCoordY*globalDimensionX;

  for (U i=0; i<localDimensionY; i++)
  {
    U saveCountRef = countRef;
    for (U j=0; j<localDimensionZ; j++)
    {
      error += abs(tester[countTester] - matrixCforEngine[countRef]);
      //if (rank == 1) { std::cout << tester[countTester] << " " << matrixCforEngine[countRef] << std::endl;}
      countRef += pGridDimensionSize;
      countTester++;
    }
    countRef = saveCountRef + pGridDimensionSize*globalDimensionX;
  }

  // Now, we need the AllReduce of the error. Very cheap operation in terms of bandwidth cost, since we are only communicating
  //   a single double primitive type.

  // sizeof trick is risky
  MPI_Allreduce(MPI_IN_PLACE, &error, sizeof(T), MPI_CHAR, MPI_SUM, sliceComm);
  return error;
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

  int rank,size;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &size);

  // size -- total number of processors in the 3D grid

  int pGridDimensionSize = ceil(pow(size,1./3.));
  
  int helper = pGridDimensionSize;
  helper *= helper;
  int pCoordX = rank%pGridDimensionSize;
  int pCoordY = (rank%helper)/pGridDimensionSize;
  int pCoordZ = rank/helper;

  MPI_Comm sliceComm;
  MPI_Comm_split(comm, pCoordZ, rank, &sliceComm);

  using globalMatrixType = Matrix<T,U,Structure,Distribution>;
  globalMatrixType globalMatrixA(globalDimensionX,globalDimensionY,globalDimensionX,globalDimensionY);
  globalMatrixType globalMatrixB(globalDimensionZ,globalDimensionX,globalDimensionZ,globalDimensionX);
  globalMatrixA.DistributeRandom(0, 0, 1, 1);		// Hardcode so that the Distributer thinks we own the entire matrix.
  globalMatrixB.DistributeRandom(0, 0, 1, 1);		// Hardcode so that the Distributer thinks we own the entire matrix.

  // No need to serialize this data. It is already in proper format for a call to BLAS
  std::vector<T> matrixAforEngine = globalMatrixA.getVectorData();
  std::vector<T> matrixBforEngine = globalMatrixB.getVectorData();
  std::vector<T> matrixCforEngine(globalDimensionY*globalDimensionZ, 0);	// No matrix needed for this. Only used in BLAS call

  switch (srcPackage.method)
  {
    case blasEngineMethod::AblasGemm:
    {
      // Later on, may want to resize matrixCforEngine in here to avoid needless memory allocation in TRMM cases
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
      matrixCforEngine = matrixBforEngine;// Dont move right now. Im worried about corrupting data in matrixB. Look into this!! std::move(matrixBforEngine;...
      break;
    }
    default:
    {
      std::cout << "Bad BLAS method used in blasEngineArgumentPackage\n";
      abort();
    }
  }

  // Now we need to iterate over both matrixCforEngine and matrixSol to find the local error.

  // Temporary code for quick n' dirty simulation of squae matrices

  T error = 0;
  std::vector<T>& tester = matrixSol.getVectorData();
  U countTester;;;;;;
  U countRef = pCoordX + pCoordY*globalDimensionX;

  for (U i=matrixSolcutYstart; i<matrixSolcutYend; i++)
  {
    U saveCountRef = countRef;
    for (U j=matrixSolcutZstart; j<matrixSolcutZend; j++)
    {
      error += abs(tester[countTester] - matrixCforEngine[countRef]);
      //if (rank == 1) { std::cout << tester[countTester] << " " << matrixCforEngine[countRef] << std::endl;}
      countRef += pGridDimensionSize;
      countTester++;
    }
    countRef = saveCountRef + pGridDimensionSize*globalDimensionX;
  }

  // Now, we need the AllReduce of the error. Very cheap operation in terms of bandwidth cost, since we are only communicating
  //   a single double primitive type.

  // sizeof trick is risky
  MPI_Allreduce(MPI_IN_PLACE, &error, sizeof(T), MPI_CHAR, MPI_SUM, sliceComm);
  return error;

}
