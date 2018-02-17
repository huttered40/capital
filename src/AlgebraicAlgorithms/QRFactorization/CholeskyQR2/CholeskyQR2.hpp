/* Author: Edward Hutter */

static std::tuple<
			MPI_Comm,
			MPI_Comm,
			MPI_Comm,
			MPI_Comm,
			MPI_Comm,
			MPI_Comm
		 >
		getTunableCommunicators(
#ifdef TIMER
      pTimer& timer,
#endif
      MPI_Comm commWorld, int pGridDimensionD, int pGridDimensionC)
{
  int worldRank, worldSize, sliceRank, columnRank;
#ifdef TIMER
  size_t index1 = timer.setStartTime("MPI_Comm_rank");
#endif
  MPI_Comm_rank(commWorld, &worldRank);
#ifdef TIMER
  timer.setEndTime("MPI_Comm_rank", index1);
  size_t index2= timer.setStartTime("MPI_Comm_size");
#endif
  MPI_Comm_size(commWorld, &worldSize);
#ifdef TIMER
  timer.setEndTime("MPI_Comm_size", index2);
#endif

  int sliceSize = pGridDimensionD*pGridDimensionC;
  MPI_Comm sliceComm, rowComm, columnComm, columnContigComm, columnAltComm, depthComm, miniCubeComm;
#ifdef TIMER
  size_t index3 = timer.setStartTime("MPI_Comm_split");
#endif
  MPI_Comm_split(commWorld, worldRank/sliceSize, worldRank, &sliceComm);
#ifdef TIMER
  timer.setEndTime("MPI_Comm_split", index3);
#endif
  MPI_Comm_rank(sliceComm, &sliceRank);
#ifdef TIMER
  size_t index4 = timer.setStartTime("MPI_Comm_split");
#endif
  MPI_Comm_split(sliceComm, sliceRank/pGridDimensionC, sliceRank, &rowComm);
#ifdef TIMER
  timer.setEndTime("MPI_Comm_split", index4);
#endif
  int cubeInt = worldRank%sliceSize;
#ifdef TIMER
  size_t index5 = timer.setStartTime("MPI_Comm_split");
#endif
  MPI_Comm_split(commWorld, cubeInt/(pGridDimensionC*pGridDimensionC), worldRank, &miniCubeComm);
#ifdef TIMER
  timer.setEndTime("MPI_Comm_split", index5);
  size_t index6 = timer.setStartTime("MPI_Comm_split");
#endif
  MPI_Comm_split(commWorld, cubeInt, worldRank, &depthComm);
#ifdef TIMER
  timer.setEndTime("MPI_Comm_split", index6);
  size_t index7 = timer.setStartTime("MPI_Comm_split");
#endif
  MPI_Comm_split(sliceComm, sliceRank%pGridDimensionC, sliceRank, &columnComm);
#ifdef TIMER
  timer.setEndTime("MPI_Comm_split", index7);
#endif
  MPI_Comm_rank(columnComm, &columnRank);
#ifdef TIMER
  size_t index8 = timer.setStartTime("MPI_Comm_split");
#endif
  MPI_Comm_split(columnComm, columnRank/pGridDimensionC, columnRank, &columnContigComm);
#ifdef TIMER
  timer.setEndTime("MPI_Comm_split", index8);
  size_t index9 = timer.setStartTime("MPI_Comm_split");
#endif
  MPI_Comm_split(columnComm, columnRank%pGridDimensionC, columnRank, &columnAltComm); 
#ifdef TIMER
  timer.setEndTime("MPI_Comm_split", index9);
#endif
  
  MPI_Comm_free(&columnComm);
  return std::make_tuple(rowComm, columnContigComm, columnAltComm, depthComm, sliceComm, miniCubeComm);
}


template<typename T,typename U,template<typename,typename> class blasEngine>
template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,blasEngine>::Factor1D(
#ifdef TIMER
    pTimer& timer,
#endif
    Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, MPI_Comm commWorld)
{
  // We assume data is owned relative to a 1D processor grid, so every processor owns a chunk of data consisting of
  //   all columns and a block of rows.

  U globalDimensionM = matrixA.getNumRowsGlobal();
  U globalDimensionN = matrixA.getNumColumnsGlobal();
  U localDimensionM = matrixA.getNumRowsLocal();

#ifdef TIMER
  size_t index2 = timer.setStartTime("3 Matrix generation in CQR2");
#endif
  Matrix<T,U,StructureA,Distribution> matrixQ2(std::vector<T>(localDimensionM*globalDimensionN), globalDimensionN, localDimensionM, globalDimensionN,
    globalDimensionM, true);
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixR1(std::vector<T>(globalDimensionN*globalDimensionN), globalDimensionN, globalDimensionN, globalDimensionN,
    globalDimensionN, true);
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixR2(std::vector<T>(globalDimensionN*globalDimensionN), globalDimensionN, globalDimensionN, globalDimensionN,
    globalDimensionN, true);
#ifdef TIMER
  timer.setEndTime("3 Matrix generation in CQR2", index2);
#endif

#ifdef TIMER
  size_t index3 = timer.setStartTime("CholeskyQR2::Factor1D_cqr");
#endif
  Factor1D_cqr(
#ifdef TIMER
    timer,
#endif
    matrixA, matrixQ2, matrixR1, commWorld);
#ifdef TIMER
  timer.setEndTime("CholeskyQR2::Factor1D_cqr", index3);
  size_t index4 = timer.setStartTime("CholeskyQR2::Factor1D_cqr");
#endif
  Factor1D_cqr(
#ifdef TIMER
    timer,
#endif
    matrixQ2, matrixQ, matrixR2, commWorld);
#ifdef TIMER
  timer.setEndTime("CholeskyQR2::Factor1D_cqr", index4);
#endif

  // Try gemm first, then try trmm later.
  blasEngineArgumentPackage_gemm<T> gemmPack1;
  gemmPack1.order = blasEngineOrder::AblasColumnMajor;
  gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
  gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
  gemmPack1.alpha = 1.;
  gemmPack1.beta = 0.;
#ifdef TIMER
  size_t index5 = timer.setStartTime("gemm");
#endif
  blasEngine<T,U>::_gemm(matrixR2.getRawData(), matrixR1.getRawData(), matrixR.getRawData(), globalDimensionN, globalDimensionN,
    globalDimensionN, globalDimensionN, globalDimensionN, globalDimensionN, gemmPack1);
#ifdef TIMER
  timer.setEndTime("gemm", index5);
#endif
}

template<typename T,typename U,template<typename,typename> class blasEngine>
template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,blasEngine>::Factor3D(
#ifdef TIMER
    pTimer& timer,
#endif
    Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR,MPI_Comm commWorld, std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D, 
      int MMid, int TSid, int INVid, int inverseCutOffMultiplier, int baseCaseMultiplier)
{
  // We assume data is owned relative to a 3D processor grid

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
#ifdef TIMER
  size_t index2 = timer.setStartTime("CholeskyQR2::Factor3D_cqr");
#endif
  Factor3D_cqr(
#ifdef TIMER
    timer,
#endif
    matrixA, matrixQ2, matrixR1, commWorld, commInfo3D, MMid, TSid, INVid, inverseCutOffMultiplier, baseCaseMultiplier);
#ifdef TIMER
  timer.setEndTime("CholeskyQR2::Factor3D_cqr", index2);
  size_t index3 = timer.setStartTime("CholeskyQR2::Factor3D_cqr");
#endif
  Factor3D_cqr(
#ifdef TIMER
    timer,
#endif
    matrixQ2, matrixQ, matrixR2, commWorld, commInfo3D, MMid, TSid, INVid, inverseCutOffMultiplier, baseCaseMultiplier);
#ifdef TIMER
  timer.setEndTime("CholeskyQR2::Factor3D_cqr", index3);
#endif

  // Try gemm first, then try trmm later.
  blasEngineArgumentPackage_gemm<T> gemmPack1;
  gemmPack1.order = blasEngineOrder::AblasColumnMajor;
  gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
  gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
  gemmPack1.alpha = 1.;
  gemmPack1.beta = 0.;

  // Later optimization - Serialize all 3 matrices into UpperTriangular first, then call this with those matrices, so we don't have to
  //   send half of the data!
#ifdef TIMER
  size_t index4 = timer.setStartTime("MM3D::Multiply");
#endif
  MM3D<T,U,blasEngine>::Multiply(
#ifdef TIMER
    timer,
#endif
    matrixR2, matrixR1, matrixR, commWorld, commInfo3D, gemmPack1, MMid);
#ifdef TIMER
  timer.setEndTime("MM3D::Multiply", index4);
#endif
}

template<typename T,typename U,template<typename,typename> class blasEngine>
template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,blasEngine>::FactorTunable(
#ifdef TIMER
    pTimer& timer,
#endif
    Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, int gridDimensionD, int gridDimensionC, MPI_Comm commWorld,
      std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm>& commInfoTunable, int MMid, int TSid, int INVid, int inverseCutOffMultiplier,
      int baseCaseMultiplier)
{
  MPI_Comm miniCubeComm = std::get<5>(commInfoTunable);
  std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int> commInfo3D = setUpCommunicators(
#ifdef TIMER
    timer,
#endif
    miniCubeComm);

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
#ifdef TIMER
  size_t index2 = timer.setStartTime("FactorTunable_cqr");
#endif
  FactorTunable_cqr(
#ifdef TIMER
    timer,
#endif
    matrixA, matrixQ2, matrixR1, gridDimensionD, gridDimensionC, commWorld, commInfoTunable, commInfo3D, MMid, TSid, INVid, inverseCutOffMultiplier, baseCaseMultiplier);
#ifdef TIMER
  timer.setEndTime("FactorTunable_cqr", index2);
  size_t index3 = timer.setStartTime("FactorTunable_cqr");
#endif
  FactorTunable_cqr(
#ifdef TIMER
    timer,
#endif
    matrixQ2, matrixQ, matrixR2, gridDimensionD, gridDimensionC, commWorld, commInfoTunable, commInfo3D, MMid, TSid, INVid, inverseCutOffMultiplier, baseCaseMultiplier);
#ifdef TIMER
  timer.setEndTime("FactorTunable_cqr", index3);
#endif

  // Try gemm first, then try trmm later.
  blasEngineArgumentPackage_gemm<T> gemmPack1;
  gemmPack1.order = blasEngineOrder::AblasColumnMajor;
  gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
  gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
  gemmPack1.alpha = 1.;
  gemmPack1.beta = 0.;

  // Later optimization - Serialize all 3 matrices into UpperTriangular first, then call this with those matrices, so we don't have to
  //   send half of the data!
#ifdef TIMER
  size_t index4 = timer.setStartTime("MM3D::Multiply");
#endif
  MM3D<T,U,blasEngine>::Multiply(
#ifdef TIMER
    timer,
#endif
    matrixR2, matrixR1, matrixR, miniCubeComm, commInfo3D, gemmPack1, MMid);
#ifdef TIMER
  timer.setEndTime("MM3D::Multiply", index4);
#endif
}


template<typename T,typename U,template<typename,typename> class blasEngine>
template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,blasEngine>::Factor1D_cqr(
#ifdef TIMER
    pTimer& timer,
#endif
    Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, MPI_Comm commWorld)
{
  // Changed from syrk to gemm
  U localDimensionM = matrixA.getNumRowsLocal();
  U localDimensionN = matrixA.getNumColumnsGlobal();
  std::vector<T> localMMvec(localDimensionN*localDimensionN);
  blasEngineArgumentPackage_syrk<T> syrkPack;
  syrkPack.order = blasEngineOrder::AblasColumnMajor;
  syrkPack.uplo = blasEngineUpLo::AblasUpper;
  syrkPack.transposeA = blasEngineTranspose::AblasTrans;
  syrkPack.alpha = 1.;
  syrkPack.beta = 0.;

#ifdef TIMER
  size_t index1 = timer.setStartTime("syrk");
#endif
  blasEngine<T,U>::_syrk(matrixA.getRawData(), matrixR.getRawData(), localDimensionN, localDimensionM,
    localDimensionM, localDimensionN, syrkPack);
#ifdef TIMER
  timer.setEndTime("syrk", index1);
#endif

  // MPI_Allreduce first to replicate the dimensionY x dimensionY matrix on each processor
  // Optimization potential: only allReduce half of this matrix because its symmetric
  //   but only try this later to see if it actually helps, because to do this, I will have to serialize and re-serialize. Would only make sense if dimensionX is huge.
#ifdef TIMER
  size_t index2 = timer.setStartTime("MPI_Allreduce");
#endif
  MPI_Allreduce(MPI_IN_PLACE, matrixR.getRawData(), localDimensionN*localDimensionN, MPI_DOUBLE, MPI_SUM, commWorld);
#ifdef TIMER
  timer.setEndTime("MPI_Allreduce", index2);
#endif

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
#ifdef TIMER
  size_t index3 = timer.setStartTime("LAPACK_DPOTRF");
#endif
  LAPACKE_dpotrf(LAPACK_COL_MAJOR, 'U', localDimensionN, matrixR.getRawData(), localDimensionN);
#ifdef TIMER
  timer.setEndTime("LAPACK_DPOTRF", index3);
#endif

  // Need a true copy to avoid corrupting R-inverse.
  std::vector<T> RI = matrixR.getVectorData();

  // Next: sequential triangular inverse. Question: does DTRTRI require packed storage or square storage? I think square, so that it can use BLAS-3.
#ifdef TIMER
  size_t index4 = timer.setStartTime("LAPACK_DTRTRI");
#endif
  LAPACKE_dtrtri(LAPACK_COL_MAJOR, 'U', 'N', localDimensionN, &RI[0], localDimensionN);
#ifdef TIMER
  timer.setEndTime("LAPACK_DTRTRI", index4);
#endif

  // Finish by performing local matrix multiplication Q = A*R^{-1}
  blasEngineArgumentPackage_gemm<T> gemmPack;
  gemmPack.order = blasEngineOrder::AblasColumnMajor;
  gemmPack.transposeA = blasEngineTranspose::AblasNoTrans;
  gemmPack.transposeB = blasEngineTranspose::AblasNoTrans;
  gemmPack.alpha = 1.;
  gemmPack.beta = 0.;
#ifdef TIMER
  size_t index5 = timer.setStartTime("gemm");
#endif
  blasEngine<T,U>::_gemm(matrixA.getRawData(), &RI[0], matrixQ.getRawData(), localDimensionM, localDimensionN,
    localDimensionN, localDimensionM, localDimensionN, localDimensionM, gemmPack);
#ifdef TIMER
  timer.setEndTime("gemm", index5);
#endif
}

template<typename T,typename U,template<typename,typename> class blasEngine>
template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,blasEngine>::Factor3D_cqr(
#ifdef TIMER
    pTimer& timer,
#endif
    Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, MPI_Comm commWorld, std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D,
      int MMid, int TSid, int INVid, int inverseCutOffMultiplier, int baseCaseMultiplier)
{

  MPI_Comm rowComm = std::get<0>(commInfo3D);
  MPI_Comm columnComm = std::get<1>(commInfo3D);
  MPI_Comm sliceComm = std::get<2>(commInfo3D);
  MPI_Comm depthComm = std::get<3>(commInfo3D);
  int pGridCoordX = std::get<4>(commInfo3D);
  int pGridCoordY = std::get<5>(commInfo3D);
  int pGridCoordZ = std::get<6>(commInfo3D);

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
#ifdef TIMER
  size_t index2 = timer.setStartTime("BroadcastPanels");
#endif
  BroadcastPanels(
#ifdef TIMER
    timer,
#endif
    (isRootRow ? dataA : foreignA), sizeA, isRootRow, pGridCoordZ, rowComm);
#ifdef TIMER
  timer.setEndTime("BroadcastPanels", index2);
#endif

  std::vector<T> localB(localDimensionN*localDimensionN);
  blasEngineArgumentPackage_gemm<T> gemmPack1;
  gemmPack1.order = blasEngineOrder::AblasColumnMajor;
  gemmPack1.transposeA = blasEngineTranspose::AblasTrans;
  gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
  gemmPack1.alpha = 1.;
  gemmPack1.beta = 0.;

  // I don't think I can run syrk here, so I will use gemm. Maybe in the future after I have it correct I can experiment.
#ifdef TIMER
  size_t index3 = timer.setStartTime("gemm");
#endif
  blasEngine<T,U>::_gemm((isRootRow ? &dataA[0] : &foreignA[0]), &dataA[0], &localB[0], localDimensionN, localDimensionN,
    localDimensionM, localDimensionM, localDimensionM, localDimensionN, gemmPack1);
#ifdef TIMER
  timer.setEndTime("gemm", index3);
#endif

#ifdef TIMER
  size_t index4 = timer.setStartTime("MPI_Reduce");
#endif
  MPI_Reduce((isRootColumn ? MPI_IN_PLACE : &localB[0]), &localB[0], localDimensionN*localDimensionN, MPI_DOUBLE,
    MPI_SUM, pGridCoordZ, columnComm);
#ifdef TIMER
  timer.setEndTime("MPI_Reduce", index4);
#endif

#ifdef TIMER
  size_t index5 = timer.setStartTime("MPI_Bcast");
#endif
  MPI_Bcast(&localB[0], localDimensionN*localDimensionN, MPI_DOUBLE, pGridCoordY, depthComm);
#ifdef TIMER
  timer.setEndTime("MPI_Bcast", index5);
#endif

  // Stuff localB vector into its own matrix so that we can pass it into CFR3D
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixB(std::move(localB), localDimensionN, localDimensionN,
    matrixA.getNumColumnsGlobal(), matrixA.getNumColumnsGlobal(), true);

  // Create an extra matrix for R-inverse
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixRI(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN,
    matrixA.getNumColumnsGlobal(), matrixA.getNumColumnsGlobal(), true);

#ifdef TIMER
  size_t index6 = timer.setStartTime("CFR3D::Factor");
#endif
  std::vector<U> baseCaseDimList = CFR3D<T,U,blasEngine>::Factor(
#ifdef TIMER
    timer,
#endif
    matrixB, matrixR, matrixRI, inverseCutOffMultiplier, 'U', baseCaseMultiplier, commWorld, commInfo3D, MMid, TSid);
#ifdef TIMER
  timer.setEndTime("CFR3D::Factor", index6);
#endif


// For now, comment this out, because I am experimenting with using TriangularSolve TRSM instead of MM3D
//   But later on once it works, use an integer or something to have both available, important when benchmarking
  // Need to be careful here. matrixRI must be truly upper-triangular for this to be correct as I found out in 1D case.

  if (!INVid)
  {
    gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
#ifdef TIMER
    size_t index7 = timer.setStartTime("MM3D::Multiply");
#endif
    MM3D<T,U,blasEngine>::Multiply(
#ifdef TIMER
      timer,
#endif
      matrixA, matrixRI, matrixQ, commWorld, commInfo3D, gemmPack1, 0);
#ifdef TIMER
    timer.setEndTime("MM3D::Multiply", index7);
#endif
  }
  else
  {
    // For debugging purposes, I am using a copy of A instead of A itself
    gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
    gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
    // alpha and beta fields don't matter. All I need from this struct are whether or not transposes are used.
#ifdef TIMER
    size_t index8 = timer.setStartTime("TRSM::iSolveUpperLeft");
#endif
    TRSM3D<T,U,blasEngine>::iSolveUpperLeft(
#ifdef TIMER
      timer,
#endif
      matrixQ, matrixR, matrixRI, matrixA,
      baseCaseDimList, gemmPack1, commWorld, commInfo3D, MMid, TSid);
#ifdef TIMER
    timer.setEndTime("TRSM::iSolveUpperLeft", index8);
#endif
  }
}

template<typename T,typename U,template<typename,typename> class blasEngine>
template<template<typename,typename,template<typename,typename,int> class> class StructureA, template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,blasEngine>::FactorTunable_cqr(
#ifdef TIMER
    pTimer& timer,
#endif
    Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureA,Distribution>& matrixQ,
    Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR, int gridDimensionD, int gridDimensionC, MPI_Comm commWorld,
      std::tuple<MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm>& tunableCommunicators,
      std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D, int MMid, int TSid, int INVid, int inverseCutOffMultiplier,
        int baseCaseMultiplier)
{
  MPI_Comm rowComm = std::get<0>(tunableCommunicators);
  MPI_Comm columnContigComm = std::get<1>(tunableCommunicators);
  MPI_Comm columnAltComm = std::get<2>(tunableCommunicators);
  MPI_Comm depthComm = std::get<3>(tunableCommunicators);
  MPI_Comm miniCubeComm = std::get<5>(tunableCommunicators);

  int worldRank;
  MPI_Comm_rank(commWorld, &worldRank);
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
#ifdef TIMER
  size_t index1 = timer.setStartTime("BroadcastPanels");
#endif
  BroadcastPanels(
#ifdef TIMER
    timer,
#endif
    (isRootRow ? dataA : foreignA), sizeA, isRootRow, pCoordZ, rowComm);
#ifdef TIMER
  timer.setEndTime("BroadcastPanels", index1);
#endif

  std::vector<T> localB(localDimensionN*localDimensionN);
  blasEngineArgumentPackage_gemm<T> gemmPack1;
  gemmPack1.order = blasEngineOrder::AblasColumnMajor;
  gemmPack1.transposeA = blasEngineTranspose::AblasTrans;
  gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
  gemmPack1.alpha = 1.;
  gemmPack1.beta = 0.;

  // I don't think I can run syrk here, so I will use gemm. Maybe in the future after I have it correct I can experiment.
#ifdef TIMER
  size_t index2 = timer.setStartTime("gemm");
#endif
  blasEngine<T,U>::_gemm((isRootRow ? &dataA[0] : &foreignA[0]), &dataA[0], &localB[0], localDimensionN, localDimensionN,
    localDimensionM, localDimensionM, localDimensionM, localDimensionN, gemmPack1);
#ifdef TIMER
  timer.setEndTime("gemm", index2);
#endif

#ifdef TIMER
  size_t index3 = timer.setStartTime("MPI_Reduce");
#endif
  MPI_Reduce((isRootColumn ? MPI_IN_PLACE : &localB[0]), &localB[0], localDimensionN*localDimensionN, MPI_DOUBLE,
    MPI_SUM, pCoordZ, columnContigComm);
#ifdef TIMER
  timer.setEndTime("MPI_Reduce", index3);
  size_t index4 = timer.setStartTime("MPI_Allreduce");
#endif
  MPI_Allreduce(MPI_IN_PLACE, &localB[0], localDimensionN*localDimensionN, MPI_DOUBLE,
    MPI_SUM, columnAltComm);
#ifdef TIMER
  timer.setEndTime("MPI_Allreduce", index4);
#endif

#ifdef TIMER
  size_t index5 = timer.setStartTime("MPI_Bcast");
#endif
  MPI_Bcast(&localB[0], localDimensionN*localDimensionN, MPI_DOUBLE, columnContigRank, depthComm);
#ifdef TIMER
  timer.setEndTime("MPI_Bcast", index5);
#endif

  // Stuff localB vector into its own matrix so that we can pass it into CFR3D
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixB(std::move(localB), localDimensionN, localDimensionN,
    globalDimensionN, globalDimensionN, true);

  // Create an extra matrix for R-inverse
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixRI(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN,
    globalDimensionN, globalDimensionN, true);

#ifdef TIMER
  size_t index6 = timer.setStartTime("CFR3D::Factor");
#endif
  std::vector<U> baseCaseDimList = CFR3D<T,U,blasEngine>::Factor(
#ifdef TIMER
    timer,
#endif
    matrixB, matrixR, matrixRI, inverseCutOffMultiplier, 'U', baseCaseMultiplier, miniCubeComm, commInfo3D, MMid, TSid);
#ifdef TIMER
  timer.setEndTime("CFR3D::Factor", index6);
#endif

  if (!INVid)
  {
    gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
#ifdef TIMER
    size_t index7 = timer.setStartTime("MM3D::Multiply");
#endif
    MM3D<T,U,blasEngine>::Multiply(
#ifdef TIMER
      timer,
#endif
      matrixA, matrixRI, matrixQ, miniCubeComm, commInfo3D, gemmPack1, MMid);
#ifdef TIMER
    timer.setEndTime("MM3D::Multiply", index7);
#endif
  }
  else
  {
    // For debugging purposes, I am using a copy of A instead of A itself
    gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
    gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
    // alpha and beta fields don't matter. All I need from this struct are whether or not transposes are used.
#ifdef TIMER
    size_t index7 = timer.setStartTime("TRSM3D::iSolveUpperLeft");
#endif
    TRSM3D<T,U,blasEngine>::iSolveUpperLeft(
#ifdef TIMER
      timer,
#endif
      matrixQ, matrixR, matrixRI, matrixA,
      baseCaseDimList, gemmPack1, miniCubeComm, commInfo3D, MMid, TSid);
#ifdef TIMER
    timer.setEndTime("TRSM3D::iSolveUpperLeft", index7);
#endif
  }



}

template<typename T,typename U, template<typename,typename> class blasEngine>
void CholeskyQR2<T,U,blasEngine>::BroadcastPanels(
#ifdef TIMER
                      pTimer& timer,
#endif
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
