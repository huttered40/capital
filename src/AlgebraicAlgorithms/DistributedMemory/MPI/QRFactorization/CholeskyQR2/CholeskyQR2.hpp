/* Author: Edward Hutter */

static std::tuple<
			MPI_Comm,
			MPI_Comm,
			MPI_Comm,
			MPI_Comm,
			MPI_Comm
		 >
		getTunableCommunicators(MPI_Comm commWorld, int pGridDimensionD, int pGridDimensionC)
{
  int worldRank, worldSize, sliceRank, columnRank;
  MPI_Comm_rank(commWorld, &worldRank);
  MPI_Comm_size(commWorld, &worldSize);

  int sliceSize = pGridDimensionD*pGridDimensionC;
  MPI_Comm sliceComm, rowComm, columnComm, columnContigComm, columnAltComm, depthComm, miniCubeComm;
  MPI_Comm_split(commWorld, worldRank/sliceSize, worldRank, &sliceComm);
  MPI_Comm_rank(sliceComm, &sliceRank);
  MPI_Comm_split(sliceComm, sliceRank/pGridDimensionC, sliceRank, &rowComm);
  int cubeInt = worldRank%sliceSize;
  MPI_Comm_split(commWorld, cubeInt/(pGridDimensionC*pGridDimensionC), worldRank, &miniCubeComm);
  MPI_Comm_split(commWorld, cubeInt, worldRank, &depthComm);
  MPI_Comm_split(sliceComm, worldRank%pGridDimensionC, worldRank, &columnComm);
  MPI_Comm_rank(columnComm, &columnRank);
  MPI_Comm_split(columnComm, columnRank/pGridDimensionC, columnRank, &columnContigComm);
  MPI_Comm_split(columnComm, columnRank%pGridDimensionC, columnRank, &columnAltComm); 

  MPI_Comm_free(&sliceComm);
  MPI_Comm_free(&columnComm);
  return std::make_tuple(rowComm, columnContigComm, columnAltComm, depthComm, miniCubeComm);
}


template<typename T,typename U,
  template<typename,typename,template<typename,typename,int> class> class StructureA,		// Note: this vould be either rectangular or square.
  template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,StructureA,blasEngine>::Factor1D(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, U globalDimensionM, U globalDimensionN, MPI_Comm commWorld)
{
  // We assume data is owned relative to a 1D processor grid, so every processor owns a chunk of data consisting of
  //   all columns and a block of rows.

  int numPEs;
  MPI_Comm_size(commWorld, &numPEs);
  U localDimensionM = globalDimensionM/numPEs;		// no error check here, but hopefully 

  Matrix<T,U,StructureA,Distribution> matrixQ2(std::vector<T>(localDimensionM*globalDimensionN), globalDimensionN, localDimensionM, globalDimensionN,
    globalDimensionM, true);
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixR1(std::vector<T>(globalDimensionN*globalDimensionN), globalDimensionN, globalDimensionN, globalDimensionN,
    globalDimensionN, true);
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixR2(std::vector<T>(globalDimensionN*globalDimensionN), globalDimensionN, globalDimensionN, globalDimensionN,
    globalDimensionN, true);

  Factor1D_cqr(matrixA, matrixQ2, matrixR1, globalDimensionN, localDimensionM, commWorld);
  Factor1D_cqr(matrixQ2, matrixQ, matrixR2, globalDimensionN, localDimensionM, commWorld);

  // Try gemm first, then try trmm later.
  blasEngineArgumentPackage_gemm<T> gemmPack1;
  gemmPack1.order = blasEngineOrder::AblasColumnMajor;
  gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
  gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
  gemmPack1.alpha = 1.;
  gemmPack1.beta = 0.;
  blasEngine<T,U>::_gemm(matrixR2.getRawData(), matrixR1.getRawData(), matrixR.getRawData(), globalDimensionN, globalDimensionN,
    globalDimensionN, globalDimensionN, globalDimensionN, globalDimensionN, gemmPack1);
}

template<typename T,typename U,
  template<typename,typename,template<typename,typename,int> class> class StructureA,		// Note: this vould be either rectangular or square.
  template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,StructureA,blasEngine>::Factor3D(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, U dimensionX, U dimensionY, MPI_Comm commWorld)
{
  // We assume data is owned relative to a 3D processor grid

  int numPEs, myRank, pGridDimensionSize;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_size(commWorld, &myRank);
  auto commInfo3D = getCommunicatorSlice(commWorld);

  // Simple asignments like these don't need pass-by-reference. Remember the new pass-by-value semantics are efficient anyways
  pGridDimensionSize = std::get<4>(commInfo3D);

  U localDimensionX = dimensionX/pGridDimensionSize;		// no error check here, but hopefully 
  U localDimensionY = dimensionY/pGridDimensionSize;		// no error check here, but hopefully 

  Matrix<T,U,StructureA,Distribution> matrixQ2(std::vector<T>(localDimensionX*localDimensionY,0), localDimensionX, localDimensionY, dimensionX,
    dimensionY, true);
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixR1(std::vector<T>(localDimensionX*localDimensionX,0), localDimensionX, localDimensionX, dimensionX,
    dimensionX, true);
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixR2(std::vector<T>(localDimensionX*localDimensionX,0), localDimensionX, localDimensionX, dimensionX,
    dimensionX, true);
  Factor3D_cqr(matrixA, matrixQ2, matrixR1, localDimensionX, localDimensionY, commWorld);
  Factor3D_cqr(matrixQ2, matrixQ, matrixR2, localDimensionX, localDimensionY, commWorld);

  // Try gemm first, then try trmm later.
  blasEngineArgumentPackage_gemm<T> gemmPack1;
  gemmPack1.order = blasEngineOrder::AblasColumnMajor;
  gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
  gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
  gemmPack1.alpha = 1.;
  gemmPack1.beta = 0.;

  // Later optimization - Serialize all 3 matrices into UpperTriangular first, then call this with those matrices, so we don't have to
  //   send half of the data!
  MM3D<T,U,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare,blasEngine>::Multiply(matrixR2, matrixR1,
    matrixR, localDimensionX, localDimensionX, localDimensionX, commWorld, gemmPack1);

  MPI_Comm_free(&std::get<0>(commInfo3D));
}

template<typename T,typename U,
  template<typename,typename,template<typename,typename,int> class> class StructureA,		// Note: this vould be either rectangular or square.
  template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,StructureA,blasEngine>::FactorTunable(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, U globalDimensionM, U globalDimensionN, int gridDimensionD, int gridDimensionC, MPI_Comm commWorld)
{
  // We assume data is owned relative to a 3D processor grid

  int numPEs, myRank, pGridDimensionSize;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_size(commWorld, &myRank);

  auto tunableCommunicators = getTunableCommunicators(commWorld, gridDimensionD, gridDimensionC);
  MPI_Comm miniCubeComm = std::get<4>(tunableCommunicators);
  int cubeSize;
  MPI_Comm_size(miniCubeComm, &cubeSize);
  std::cout << "size of cube comm - " << cubeSize << std::endl;

  U localDimensionM = globalDimensionM/gridDimensionD;
  U localDimensionN = globalDimensionN/gridDimensionC;

  // Need to get the right global dimensions here, use a tunable package struct or something
  Matrix<T,U,StructureA,Distribution> matrixQ2(std::vector<T>(localDimensionN*localDimensionM,0), localDimensionN, localDimensionM, globalDimensionN,
    globalDimensionM, true);
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixR1(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN, globalDimensionN,
    globalDimensionN, true);
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixR2(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN, globalDimensionN,
    globalDimensionN, true);
  FactorTunable_cqr(matrixA, matrixQ2, matrixR1, localDimensionM, localDimensionN, gridDimensionD, gridDimensionC, commWorld, tunableCommunicators);
  FactorTunable_cqr(matrixQ2, matrixQ, matrixR2, localDimensionM, localDimensionN, gridDimensionD, gridDimensionC, commWorld, tunableCommunicators);

  // Try gemm first, then try trmm later.
  blasEngineArgumentPackage_gemm<T> gemmPack1;
  gemmPack1.order = blasEngineOrder::AblasColumnMajor;
  gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
  gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
  gemmPack1.alpha = 1.;
  gemmPack1.beta = 0.;

  // Later optimization - Serialize all 3 matrices into UpperTriangular first, then call this with those matrices, so we don't have to
  //   send half of the data!
  MM3D<T,U,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare,blasEngine>::Multiply(matrixR2, matrixR1,
    matrixR, localDimensionN, localDimensionN, localDimensionN, miniCubeComm, gemmPack1);

  MPI_Comm_free(&std::get<0>(tunableCommunicators));
  MPI_Comm_free(&std::get<1>(tunableCommunicators));
  MPI_Comm_free(&std::get<2>(tunableCommunicators));
  MPI_Comm_free(&std::get<3>(tunableCommunicators));
  MPI_Comm_free(&std::get<4>(tunableCommunicators));
}


template<typename T,typename U,
  template<typename,typename,template<typename,typename,int> class> class StructureA,		// Note: this vould be either rectangular or square.
  template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,StructureA,blasEngine>::Factor1D_cqr(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, U localDimensionX, U localDimensionY, MPI_Comm commWorld)
{
  int numPEs;
  MPI_Comm_size(commWorld, &numPEs);

  // Changed from syrk to gemm
  std::vector<T> localMMvec(localDimensionX*localDimensionX);
  blasEngineArgumentPackage_syrk<T> syrkPack;
  syrkPack.order = blasEngineOrder::AblasColumnMajor;
  syrkPack.uplo = blasEngineUpLo::AblasUpper;
  syrkPack.transposeA = blasEngineTranspose::AblasTrans;
  syrkPack.alpha = 1.;
  syrkPack.beta = 0.;

  blasEngine<T,U>::_syrk(matrixA.getRawData(), matrixR.getRawData(), localDimensionX, localDimensionY,
    localDimensionY, localDimensionX, syrkPack);

  // MPI_Allreduce first to replicate the dimensionY x dimensionY matrix on each processor
  // Optimization potential: only allReduce half of this matrix because its symmetric
  //   but only try this later to see if it actually helps, because to do this, I will have to serialize and re-serialize. Would only make sense if dimensionX is huge.
  MPI_Allreduce(MPI_IN_PLACE, matrixR.getRawData(), localDimensionX*localDimensionX, MPI_DOUBLE, MPI_SUM, commWorld);

  // For correctness, because we are storing R and RI as square matrices AND using GEMM later on for Q=A*RI, lets manually set the lower-triangular portion of R to zeros
  //   Note that we could also do this before the AllReduce, but it wouldnt affect the cost
  for (U i=0; i<localDimensionX; i++)
  {
    for (U j=i+1; j<localDimensionX; j++)
    {
      matrixR.getRawData()[i*localDimensionX+j] = 0;
    }
  }

  // Now, localMMvec is replicated on every processor in commWorld
  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', localDimensionX, matrixR.getRawData(), localDimensionX);

  // Need a true copy to avoid corrupting R-inverse.
  std::vector<T> RI = matrixR.getVectorData();

  // Next: sequential triangular inverse. Question: does DTRTRI require packed storage or square storage? I think square, so that it can use BLAS-3.
  LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', localDimensionX, &RI[0], localDimensionX);

  // Finish by performing local matrix multiplication Q = A*R^{-1}
  blasEngineArgumentPackage_gemm<T> gemmPack;
  gemmPack.order = blasEngineOrder::AblasColumnMajor;
  gemmPack.transposeA = blasEngineTranspose::AblasNoTrans;
  gemmPack.transposeB = blasEngineTranspose::AblasNoTrans;
  gemmPack.alpha = 1.;
  gemmPack.beta = 0.;
  blasEngine<T,U>::_gemm(matrixA.getRawData(), &RI[0], matrixQ.getRawData(), localDimensionY, localDimensionX,
    localDimensionX, localDimensionY, localDimensionX, localDimensionY, gemmPack);
}

template<typename T,typename U,
  template<typename,typename,template<typename,typename,int> class> class StructureA,		// Note: this vould be either rectangular or square.
  template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,StructureA,blasEngine>::Factor3D_cqr(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, U localDimensionX, U localDimensionY, MPI_Comm commWorld)
{
  int numPEs, myRank, pGridDimensionSize;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_rank(commWorld, &myRank);
  auto commInfo3D = setUpCommunicators(commWorld);

  MPI_Comm rowComm = std::get<0>(commInfo3D);
  MPI_Comm columnComm = std::get<1>(commInfo3D);
  MPI_Comm sliceComm = std::get<2>(commInfo3D);
  MPI_Comm depthComm = std::get<3>(commInfo3D);
  int pGridCoordX = std::get<4>(commInfo3D);
  int pGridCoordY = std::get<5>(commInfo3D);
  int pGridCoordZ = std::get<6>(commInfo3D);
  MPI_Comm_size(rowComm, &pGridDimensionSize);

  // Need to perform the multiple steps to get our partition of matrixA
  std::vector<T>& dataA = matrixA.getVectorData();
  U sizeA = matrixA.getNumElems();
  std::vector<T> foreignA;	// dont fill with data first, because if root its a waste,
                                //   but need it to outside to get outside scope
  bool isRootRow = ((pGridCoordX == pGridCoordZ) ? true : false);
  bool isRootColumn = ((pGridCoordY == pGridCoordZ) ? true : false);

  // No optimization here I am pretty sure due to final result being symmetric, as it is cyclic and transpose isnt true as I have painfully found out before.
  BroadcastPanels((isRootRow ? dataA : foreignA), sizeA, isRootRow, pGridCoordZ, rowComm);

  std::vector<T> localB(localDimensionX*localDimensionX);
  blasEngineArgumentPackage_gemm<T> gemmPack1;
  gemmPack1.order = blasEngineOrder::AblasColumnMajor;
  gemmPack1.transposeA = blasEngineTranspose::AblasTrans;
  gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
  gemmPack1.alpha = 1.;
  gemmPack1.beta = 0.;

  // I don't think I can run syrk here, so I will use gemm. Maybe in the future after I have it correct I can experiment.
  blasEngine<T,U>::_gemm((isRootRow ? &dataA[0] : &foreignA[0]), &dataA[0], &localB[0], localDimensionX, localDimensionX,
    localDimensionY, localDimensionY, localDimensionY, localDimensionX, gemmPack1);

  MPI_Reduce((isRootColumn ? MPI_IN_PLACE : &localB[0]), &localB[0], localDimensionX*localDimensionX, MPI_DOUBLE,
    MPI_SUM, pGridCoordZ, columnComm);

  MPI_Bcast(&localB[0], localDimensionX*localDimensionX, MPI_DOUBLE, pGridCoordY, depthComm);

  // Stuff localB vector into its own matrix so that we can pass it into CFR3D
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixB(std::move(localB), localDimensionX, localDimensionX,
    localDimensionX*pGridDimensionSize, localDimensionX*pGridDimensionSize, true);

  // Create an extra matrix for R-inverse
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixRI(std::vector<T>(localDimensionX*localDimensionX,0), localDimensionX, localDimensionX,
    localDimensionX*pGridDimensionSize, localDimensionX*pGridDimensionSize, true);

  CFR3D<T,U,MatrixStructureSquare,MatrixStructureSquare,blasEngine>::Factor(matrixB, matrixR, matrixRI, localDimensionX, 'U', commWorld);

  // Need to be careful here. matrixRI must be truly upper-triangular for this to be correct as I found out in 1D case.
  gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
  MM3D<T,U,StructureA,MatrixStructureSquare,StructureA,blasEngine>::Multiply(matrixA, matrixRI,
    matrixQ, localDimensionX, localDimensionY, localDimensionX, commWorld, gemmPack1);

  MPI_Comm_free(&rowComm);
  MPI_Comm_free(&columnComm);
  MPI_Comm_free(&sliceComm);
  MPI_Comm_free(&depthComm);
}

template<typename T,typename U,
  template<typename,typename,template<typename,typename,int> class> class StructureA,		// Note: this vould be either rectangular or square.
  template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,StructureA,blasEngine>::FactorTunable_cqr(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, U localDimensionM, U localDimensionN, int gridDimensionD, int gridDimensionC, MPI_Comm commWorld,
      std::tuple<MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm> tunableCommunicators)
{
  MPI_Comm rowComm = std::get<0>(tunableCommunicators);
  MPI_Comm columnContigComm = std::get<1>(tunableCommunicators);
  MPI_Comm columnAltComm = std::get<2>(tunableCommunicators);
  MPI_Comm depthComm = std::get<3>(tunableCommunicators);
  MPI_Comm miniCubeComm = std::get<4>(tunableCommunicators);

  int worldRank,worldSize;
  MPI_Comm_rank(commWorld, &worldRank);
  MPI_Comm_size(commWorld, &worldSize);
  int sliceSize = gridDimensionD*gridDimensionC;
  int pCoordX = worldRank%gridDimensionC;
  int pCoordY = (worldRank%sliceSize)/gridDimensionC;
  int pCoordZ = worldRank/sliceSize;

  int columnContigRank;
  MPI_Comm_rank(columnContigComm, &columnContigRank);

  // Need to perform the multiple steps to get our partition of matrixA
  std::vector<T>& dataA = matrixA.getVectorData();
  U sizeA = matrixA.getNumElems();
  std::vector<T> foreignA;	// dont fill with data first, because if root its a waste,
                                //   but need it to outside to get outside scope
  bool isRootRow = ((pCoordX == pCoordZ) ? true : false);
  bool isRootColumn = ((columnContigRank == pCoordZ) ? true : false);

  // No optimization here I am pretty sure due to final result being symmetric, as it is cyclic and transpose isnt true as I have painfully found out before.
  BroadcastPanels((isRootRow ? dataA : foreignA), sizeA, isRootRow, pCoordZ, rowComm);

  std::vector<T> localB(localDimensionN*localDimensionN);
  blasEngineArgumentPackage_gemm<T> gemmPack1;
  gemmPack1.order = blasEngineOrder::AblasColumnMajor;
  gemmPack1.transposeA = blasEngineTranspose::AblasTrans;
  gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
  gemmPack1.alpha = 1.;
  gemmPack1.beta = 0.;

  // I don't think I can run syrk here, so I will use gemm. Maybe in the future after I have it correct I can experiment.
  blasEngine<T,U>::_gemm((isRootRow ? &dataA[0] : &foreignA[0]), &dataA[0], &localB[0], localDimensionN, localDimensionN,
    localDimensionM, localDimensionM, localDimensionM, localDimensionN, gemmPack1);

  MPI_Reduce((isRootColumn ? MPI_IN_PLACE : &localB[0]), &localB[0], localDimensionN*localDimensionN, MPI_DOUBLE,
    MPI_SUM, pCoordZ, columnContigComm);
  MPI_Allreduce(MPI_IN_PLACE, &localB[0], localDimensionN*localDimensionN, MPI_DOUBLE,
    MPI_SUM, columnAltComm);

  MPI_Bcast(&localB[0], localDimensionN*localDimensionN, MPI_DOUBLE, columnContigRank, depthComm);

  // Stuff localB vector into its own matrix so that we can pass it into CFR3D
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixB(std::move(localB), localDimensionN, localDimensionN,
    localDimensionN*gridDimensionC, localDimensionN*gridDimensionC, true);

  // Create an extra matrix for R-inverse
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixRI(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN,
    localDimensionN*gridDimensionC, localDimensionN*gridDimensionC, true);

  CFR3D<T,U,MatrixStructureSquare,MatrixStructureSquare,blasEngine>::Factor(matrixB, matrixR, matrixRI, localDimensionN, 'U', miniCubeComm);

  // Need to be careful here. matrixRI must be truly upper-triangular for this to be correct as I found out in 1D case.
  gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
  MM3D<T,U,StructureA,MatrixStructureSquare,StructureA,blasEngine>::Multiply(matrixA, matrixRI,
    matrixQ, localDimensionM, localDimensionN, localDimensionN, miniCubeComm, gemmPack1);

}


template<typename T,typename U,
  template<typename,typename,template<typename,typename,int> class> class StructureA,		// Note: this vould be either rectangular or square.
  template<typename,typename> class blasEngine>
void CholeskyQR2<T,U,StructureA, blasEngine>::BroadcastPanels(
											std::vector<T>& data,
											U size,
											bool isRoot,
											int pGridCoordZ,
											MPI_Comm panelComm
										    )
{
  if (isRoot)
  {
    MPI_Bcast(&data[0], size, MPI_DOUBLE, pGridCoordZ, panelComm);
  }
  else
  {
    data.resize(size);
    MPI_Bcast(&data[0], size, MPI_DOUBLE, pGridCoordZ, panelComm);
  }
}

