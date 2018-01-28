/* Author: Edward Hutter */

static std::tuple<
			MPI_Comm,
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
  MPI_Comm_split(sliceComm, sliceRank%pGridDimensionC, sliceRank, &columnComm);
  MPI_Comm_rank(columnComm, &columnRank);
  MPI_Comm_split(columnComm, columnRank/pGridDimensionC, columnRank, &columnContigComm);
  MPI_Comm_split(columnComm, columnRank%pGridDimensionC, columnRank, &columnAltComm); 
  
  MPI_Comm_free(&columnComm);
  return std::make_tuple(rowComm, columnContigComm, columnAltComm, depthComm, sliceComm, miniCubeComm);
}


template<typename T,typename U,template<typename,typename> class blasEngine>
template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,blasEngine>::Factor1D(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, MPI_Comm commWorld)
{
  // We assume data is owned relative to a 1D processor grid, so every processor owns a chunk of data consisting of
  //   all columns and a block of rows.

  int numPEs;
  MPI_Comm_size(commWorld, &numPEs);
  U globalDimensionM = matrixA.getNumRowsGlobal();
  U globalDimensionN = matrixA.getNumColumnsGlobal();
  U localDimensionM = matrixA.getNumRowsLocal();//globalDimensionM/numPEs;		// no error check here, but hopefully 

  Matrix<T,U,StructureA,Distribution> matrixQ2(std::vector<T>(localDimensionM*globalDimensionN), globalDimensionN, localDimensionM, globalDimensionN,
    globalDimensionM, true);
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixR1(std::vector<T>(globalDimensionN*globalDimensionN), globalDimensionN, globalDimensionN, globalDimensionN,
    globalDimensionN, true);
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixR2(std::vector<T>(globalDimensionN*globalDimensionN), globalDimensionN, globalDimensionN, globalDimensionN,
    globalDimensionN, true);

  Factor1D_cqr(matrixA, matrixQ2, matrixR1, commWorld);
  Factor1D_cqr(matrixQ2, matrixQ, matrixR2, commWorld);

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

template<typename T,typename U,template<typename,typename> class blasEngine>
template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,blasEngine>::Factor3D(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR,MPI_Comm commWorld)
{
  // We assume data is owned relative to a 3D processor grid

  int numPEs, myRank, pGridDimensionSize;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_size(commWorld, &myRank);
  auto commInfo3D = getCommunicatorSlice(commWorld);

  // Simple asignments like these don't need pass-by-reference. Remember the new pass-by-value semantics are efficient anyways
  pGridDimensionSize = std::get<4>(commInfo3D);

  U globalDimensionM = matrixA.getNumRowsGlobal();
  U globalDimensionN = matrixA.getNumColumnsGlobal();
  U localDimensionN = matrixA.getNumColumnsLocal();		// no error check here, but hopefully 
  U localDimensionM = matrixA.getNumRowsLocal();		// no error check here, but hopefully 

  Matrix<T,U,StructureA,Distribution> matrixQ2(std::vector<T>(localDimensionN*localDimensionM,0), localDimensionN, localDimensionM, globalDimensionN,
    globalDimensionM, true);
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixR1(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN, globalDimensionN,
    globalDimensionN, true);
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixR2(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN, globalDimensionN,
    globalDimensionN, true);
  Factor3D_cqr(matrixA, matrixQ2, matrixR1, commWorld);
  Factor3D_cqr(matrixQ2, matrixQ, matrixR2, commWorld);

  // Try gemm first, then try trmm later.
  blasEngineArgumentPackage_gemm<T> gemmPack1;
  gemmPack1.order = blasEngineOrder::AblasColumnMajor;
  gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
  gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
  gemmPack1.alpha = 1.;
  gemmPack1.beta = 0.;

  // Later optimization - Serialize all 3 matrices into UpperTriangular first, then call this with those matrices, so we don't have to
  //   send half of the data!
  MM3D<T,U,blasEngine>::Multiply(matrixR2, matrixR1, matrixR, commWorld, gemmPack1, 0);

  MPI_Comm_free(&std::get<0>(commInfo3D));
}

template<typename T,typename U,template<typename,typename> class blasEngine>
template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,blasEngine>::FactorTunable(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, int gridDimensionD, int gridDimensionC, MPI_Comm commWorld)
{
  // We assume data is owned relative to a 3D processor grid
  int numPEs, myRank, pGridDimensionSize;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_rank(commWorld, &myRank);

  auto tunableCommunicators = getTunableCommunicators(commWorld, gridDimensionD, gridDimensionC);
  MPI_Comm miniCubeComm = std::get<5>(tunableCommunicators);


  U globalDimensionM = matrixA.getNumRowsGlobal();
  U globalDimensionN = matrixA.getNumColumnsGlobal();
  U localDimensionM = matrixA.getNumRowsLocal();//globalDimensionM/gridDimensionD;
  U localDimensionN = matrixA.getNumColumnsLocal();//globalDimensionN/gridDimensionC;

  // Need to get the right global dimensions here, use a tunable package struct or something
  Matrix<T,U,StructureA,Distribution> matrixQ2(std::vector<T>(localDimensionN*localDimensionM,0), localDimensionN, localDimensionM, globalDimensionN,
    globalDimensionM, true);
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixR1(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN, globalDimensionN,
    globalDimensionN, true);
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixR2(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN, globalDimensionN,
    globalDimensionN, true);
  FactorTunable_cqr(matrixA, matrixQ2, matrixR1, gridDimensionD, gridDimensionC, commWorld, tunableCommunicators);
  FactorTunable_cqr(matrixQ2, matrixQ, matrixR2, gridDimensionD, gridDimensionC, commWorld, tunableCommunicators);

  // Try gemm first, then try trmm later.
  blasEngineArgumentPackage_gemm<T> gemmPack1;
  gemmPack1.order = blasEngineOrder::AblasColumnMajor;
  gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
  gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
  gemmPack1.alpha = 1.;
  gemmPack1.beta = 0.;

  // Later optimization - Serialize all 3 matrices into UpperTriangular first, then call this with those matrices, so we don't have to
  //   send half of the data!
  MM3D<T,U,blasEngine>::Multiply(matrixR2, matrixR1, matrixR, miniCubeComm, gemmPack1, 0);

  MPI_Comm_free(&std::get<0>(tunableCommunicators));
  MPI_Comm_free(&std::get<1>(tunableCommunicators));
  MPI_Comm_free(&std::get<2>(tunableCommunicators));
  MPI_Comm_free(&std::get<3>(tunableCommunicators));
  MPI_Comm_free(&std::get<4>(tunableCommunicators));
  MPI_Comm_free(&std::get<5>(tunableCommunicators));
}


template<typename T,typename U,template<typename,typename> class blasEngine>
template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,blasEngine>::Factor1D_cqr(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, MPI_Comm commWorld)
{
  int numPEs;
  MPI_Comm_size(commWorld, &numPEs);

  // Changed from syrk to gemm
  U localDimensionM = matrixA.getNumRowsLocal();
  U localDimensionN = matrixA.getNumColumnsGlobal();//globalDimensionM/numPEs;		// no error check here, but hopefully 
  std::vector<T> localMMvec(localDimensionN*localDimensionN);
  blasEngineArgumentPackage_syrk<T> syrkPack;
  syrkPack.order = blasEngineOrder::AblasColumnMajor;
  syrkPack.uplo = blasEngineUpLo::AblasUpper;
  syrkPack.transposeA = blasEngineTranspose::AblasTrans;
  syrkPack.alpha = 1.;
  syrkPack.beta = 0.;

  blasEngine<T,U>::_syrk(matrixA.getRawData(), matrixR.getRawData(), localDimensionN, localDimensionM,
    localDimensionM, localDimensionN, syrkPack);

  // MPI_Allreduce first to replicate the dimensionY x dimensionY matrix on each processor
  // Optimization potential: only allReduce half of this matrix because its symmetric
  //   but only try this later to see if it actually helps, because to do this, I will have to serialize and re-serialize. Would only make sense if dimensionX is huge.
  MPI_Allreduce(MPI_IN_PLACE, matrixR.getRawData(), localDimensionN*localDimensionN, MPI_DOUBLE, MPI_SUM, commWorld);

  // For correctness, because we are storing R and RI as square matrices AND using GEMM later on for Q=A*RI, lets manually set the lower-triangular portion of R to zeros
  //   Note that we could also do this before the AllReduce, but it wouldnt affect the cost
  for (U i=0; i<localDimensionN; i++)
  {
    for (U j=i+1; j<localDimensionN; j++)
    {
      matrixR.getRawData()[i*localDimensionN+j] = 0;
    }
  }

  // Now, localMMvec is replicated on every processor in commWorld
  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', localDimensionN, matrixR.getRawData(), localDimensionN);

  // Need a true copy to avoid corrupting R-inverse.
  std::vector<T> RI = matrixR.getVectorData();

  // Next: sequential triangular inverse. Question: does DTRTRI require packed storage or square storage? I think square, so that it can use BLAS-3.
  LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', localDimensionN, &RI[0], localDimensionN);

  // Finish by performing local matrix multiplication Q = A*R^{-1}
  blasEngineArgumentPackage_gemm<T> gemmPack;
  gemmPack.order = blasEngineOrder::AblasColumnMajor;
  gemmPack.transposeA = blasEngineTranspose::AblasNoTrans;
  gemmPack.transposeB = blasEngineTranspose::AblasNoTrans;
  gemmPack.alpha = 1.;
  gemmPack.beta = 0.;
  blasEngine<T,U>::_gemm(matrixA.getRawData(), &RI[0], matrixQ.getRawData(), localDimensionM, localDimensionN,
    localDimensionN, localDimensionM, localDimensionN, localDimensionM, gemmPack);
}

template<typename T,typename U,template<typename,typename> class blasEngine>
template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,blasEngine>::Factor3D_cqr(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, MPI_Comm commWorld)
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
  U localDimensionN = matrixA.getNumColumnsLocal();		// no error check here, but hopefully 
  U localDimensionM = matrixA.getNumRowsLocal();		// no error check here, but hopefully 
  std::vector<T>& dataA = matrixA.getVectorData();
  U sizeA = matrixA.getNumElems();
  std::vector<T> foreignA;	// dont fill with data first, because if root its a waste,
                                //   but need it to outside to get outside scope
  bool isRootRow = ((pGridCoordX == pGridCoordZ) ? true : false);
  bool isRootColumn = ((pGridCoordY == pGridCoordZ) ? true : false);

  // No optimization here I am pretty sure due to final result being symmetric, as it is cyclic and transpose isnt true as I have painfully found out before.
  BroadcastPanels((isRootRow ? dataA : foreignA), sizeA, isRootRow, pGridCoordZ, rowComm);

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
    MPI_SUM, pGridCoordZ, columnComm);

  MPI_Bcast(&localB[0], localDimensionN*localDimensionN, MPI_DOUBLE, pGridCoordY, depthComm);

  // Stuff localB vector into its own matrix so that we can pass it into CFR3D
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixB(std::move(localB), localDimensionN, localDimensionN,
    matrixA.getNumColumnsGlobal(), matrixA.getNumColumnsGlobal(), true);

  // Create an extra matrix for R-inverse
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixRI(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN,
    matrixA.getNumColumnsGlobal(), matrixA.getNumColumnsGlobal(), true);

  CFR3D<T,U,blasEngine>::Factor(matrixB, matrixR, matrixRI, 'U', 0, commWorld);


// For now, comment this out, because I am experimenting with using TriangularSolve TRSM instead of MM3D
//   But later on once it works, use an integer or something to have both available, important when benchmarking
  // Need to be careful here. matrixRI must be truly upper-triangular for this to be correct as I found out in 1D case.
//  gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
//  MM3D<T,U,blasEngine>::Multiply(matrixA, matrixRI, matrixQ, commWorld, gemmPack1, 0);

  int MMid = 0;  // Broadcast + Allreduce
  // For debugging purposes, I am using a copy of A instead of A itself
  Matrix<T,U,StructureA,Distribution> matrixAcopy = matrixA;
  TRSM3D<T,U,blasEngine>::iSolveUpperLeft(matrixQ, matrixR, matrixRI, matrixAcopy, 0, localDimensionN, 0, localDimensionM, 0, localDimensionN,
    0, localDimensionN, 0, localDimensionN, 0, localDimensionM, MMid, commWorld);

  MPI_Comm_free(&rowComm);
  MPI_Comm_free(&columnComm);
  MPI_Comm_free(&sliceComm);
  MPI_Comm_free(&depthComm);
}

template<typename T,typename U,template<typename,typename> class blasEngine>
template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,blasEngine>::FactorTunable_cqr(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, int gridDimensionD, int gridDimensionC, MPI_Comm commWorld,
      std::tuple<MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm> tunableCommunicators)
{
  MPI_Comm rowComm = std::get<0>(tunableCommunicators);
  MPI_Comm columnContigComm = std::get<1>(tunableCommunicators);
  MPI_Comm columnAltComm = std::get<2>(tunableCommunicators);
  MPI_Comm depthComm = std::get<3>(tunableCommunicators);
  MPI_Comm miniCubeComm = std::get<5>(tunableCommunicators);

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
  U globalDimensionM = matrixA.getNumRowsGlobal();
  U globalDimensionN = matrixA.getNumColumnsGlobal();
  U localDimensionM = matrixA.getNumRowsLocal();//globalDimensionM/gridDimensionD;
  U localDimensionN = matrixA.getNumColumnsLocal();//globalDimensionN/gridDimensionC;
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
    globalDimensionN, globalDimensionN, true);

  // Create an extra matrix for R-inverse
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixRI(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN,
    globalDimensionN, globalDimensionN, true);

//  CFR3D<T,U,blasEngine>::Factor(matrixB, matrixR, matrixRI, 'U', 0, miniCubeComm);

  // Need to be careful here. matrixRI must be truly upper-triangular for this to be correct as I found out in 1D case.
  gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
  MM3D<T,U,blasEngine>::Multiply(matrixA, matrixRI, matrixQ, miniCubeComm, gemmPack1, 0);

}


template<typename T,typename U, template<typename,typename> class blasEngine>
void CholeskyQR2<T,U,blasEngine>::BroadcastPanels(
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

