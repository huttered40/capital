/* Author: Edward Hutter */

template<typename MatrixAType, typename MatrixRType>
void CholeskyQR2::Factor1D(MatrixAType& matrixA, MatrixRType& matrixR, MPI_Comm commWorld){
  TAU_FSTART(Factor1D);
  // We assume data is owned relative to a 1D processor grid, so every processor owns a chunk of data consisting of
  //   all columns and a block of rows.

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  U globalDimensionN = matrixA.getNumColumnsGlobal();
  U localDimensionN = matrixA.getNumColumnsLocal();

  MatrixRType matrixR2(std::vector<T>(localDimensionN*localDimensionN), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN, true);

  Factor1D_cqr(matrixA, matrixR, commWorld);
  Factor1D_cqr(matrixA, matrixR2, commWorld);

  // Remove all zeros from LT part of only matrixR, since we are using TRMM
  // TODO: motivation for policy class design decision. Is this cost worth it just to use TRMM instead of GEMM?
  for (U i=0; i<localDimensionN; i++){
    for (U j=i+1; j<localDimensionN; j++){
      matrixR.getRawData()[i*localDimensionN+j] = 0;
    }
  }
  blasEngineArgumentPackage_trmm<T> trmmPack1(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasLeft, blasEngineUpLo::AblasUpper,
    blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
  blasEngine::_trmm(matrixR2.getRawData(), matrixR.getRawData(), localDimensionN, localDimensionN,
    localDimensionN, localDimensionN, trmmPack1);
  TAU_FSTOP(Factor1D);
}

template<typename MatrixAType, typename MatrixRType>
void CholeskyQR2::Factor3D(MatrixAType& matrixA, MatrixRType& matrixR, MPI_Comm commWorld, std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D, 
                           size_t inverseCutOffMultiplier, size_t baseCaseMultiplier, size_t panelDimensionMultiplier){
  TAU_FSTART(Factor3D);
  // We assume data is owned relative to a 3D processor grid

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  U globalDimensionN = matrixA.getNumColumnsGlobal();
  U localDimensionN = matrixA.getNumColumnsLocal();		// no error check here, but hopefully 

  MatrixRType matrixR2(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN, true);
  Factor3D_cqr(matrixA, matrixR, commWorld, commInfo3D, inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier);
  Factor3D_cqr(matrixA, matrixR2, commWorld, commInfo3D, inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier);

  blasEngineArgumentPackage_trmm<T> trmmPack1(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasLeft, blasEngineUpLo::AblasUpper,
    blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
  // Later optimization - Serialize all 3 matrices into UpperTriangular first, then call this with those matrices, so we don't have to
  //   send half of the data!
  MM3D::Multiply(matrixR2, matrixR, commWorld, commInfo3D, trmmPack1);
  TAU_FSTOP(Factor3D);
}

template<typename MatrixAType, typename MatrixRType>
void CholeskyQR2::FactorTunable(MatrixAType& matrixA, MatrixRType& matrixR, size_t gridDimensionD, size_t gridDimensionC, MPI_Comm commWorld,
      std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm>& commInfoTunable, size_t inverseCutOffMultiplier, size_t baseCaseMultiplier, size_t panelDimensionMultiplier){
  TAU_FSTART(FactorTunable);
  if (gridDimensionC == 1){
    Factor1D(matrixA, matrixR, commWorld);  
    return;
  }
  MPI_Comm miniCubeComm = std::get<5>(commInfoTunable);
  std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t> commInfo3D = util::build3DTopology(miniCubeComm);
  if (gridDimensionC == gridDimensionD){
    Factor3D(matrixA, matrixR, commWorld, commInfo3D, inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier);
    util::destroy3DTopology(commInfo3D);
    return;
  }

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  U globalDimensionN = matrixA.getNumColumnsGlobal();
  U localDimensionN = matrixA.getNumColumnsLocal();//globalDimensionN/gridDimensionC;

  // Need to get the right global dimensions here, use a tunable package struct or something
  MatrixRType matrixR2(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN, true);
  FactorTunable_cqr(matrixA, matrixR, gridDimensionD, gridDimensionC, commWorld, commInfoTunable, commInfo3D, inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier);
  FactorTunable_cqr(matrixA, matrixR2, gridDimensionD, gridDimensionC, commWorld, commInfoTunable, commInfo3D, inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier);

  blasEngineArgumentPackage_trmm<T> trmmPack1(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasLeft, blasEngineUpLo::AblasUpper,
    blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
  // Later optimization - Serialize all 3 matrices into UpperTriangular first, then call this with those matrices, so we don't have to
  //   send half of the data!
  MM3D::Multiply(matrixR2, matrixR, miniCubeComm, commInfo3D, trmmPack1);

  util::destroy3DTopology(commInfo3D);
  TAU_FSTOP(FactorTunable);
}


template<typename MatrixAType, typename MatrixRType>
void CholeskyQR2::Factor1D_cqr(MatrixAType& matrixA, MatrixRType& matrixR, MPI_Comm commWorld){
  TAU_FSTART(Factor1D_cqr);

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  // Changed from syrk to gemm
  U localDimensionM = matrixA.getNumRowsLocal();
  U localDimensionN = matrixA.getNumColumnsLocal();
  std::vector<T> localMMvec(localDimensionN*localDimensionN);
  blasEngineArgumentPackage_syrk<T> syrkPack(blasEngineOrder::AblasColumnMajor, blasEngineUpLo::AblasUpper, blasEngineTranspose::AblasTrans, 1., 0.);
  blasEngine::_syrk(matrixA.getRawData(), matrixR.getRawData(), localDimensionN, localDimensionM, localDimensionM, localDimensionN, syrkPack);

  // MPI_Allreduce to replicate the dimensionY x dimensionY matrix on each processor
  // Optimization potential: only Allreduce half of this matrix because its symmetric
  //   but only try this later to see if it actually helps, because to do this, I will have to serialize and re-serialize. Would only make sense if dimensionX is huge.
  MPI_Allreduce(MPI_IN_PLACE, matrixR.getRawData(), localDimensionN*localDimensionN, MPI_DATATYPE, MPI_SUM, commWorld);

  lapackEngineArgumentPackage_potrf potrfArgs(lapackEngineOrder::AlapackColumnMajor, lapackEngineUpLo::AlapackUpper);
  lapackEngineArgumentPackage_trtri trtriArgs(lapackEngineOrder::AlapackColumnMajor, lapackEngineUpLo::AlapackUpper, lapackEngineDiag::AlapackNonUnit);
  lapackEngine::_potrf(matrixR.getRawData(), localDimensionN, localDimensionN, potrfArgs);
  std::vector<T> RI = matrixR.getVectorData();
  lapackEngine::_trtri(&RI[0], localDimensionN, localDimensionN, trtriArgs);

  // Finish by performing local matrix multiplication Q = A*R^{-1}
  blasEngineArgumentPackage_trmm<T> trmmPack1(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasRight, blasEngineUpLo::AblasUpper, blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
  blasEngine::_trmm(&RI[0], matrixA.getRawData(), localDimensionM, localDimensionN, localDimensionN, localDimensionM, trmmPack1);
  TAU_FSTOP(Factor1D_cqr);
}

template<typename MatrixAType, typename MatrixRType>
void CholeskyQR2::Factor3D_cqr(MatrixAType& matrixA, MatrixRType& matrixR, MPI_Comm commWorld,
      std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D, size_t inverseCutOffMultiplier, size_t baseCaseMultiplier, size_t panelDimensionMultiplier){

  TAU_FSTART(Factor3D_cqr);

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using Distribution = typename MatrixAType::DistributionType;
  using Offload = typename MatrixAType::OffloadType;

  MPI_Comm rowComm = std::get<0>(commInfo3D);
  MPI_Comm columnComm = std::get<1>(commInfo3D);
  MPI_Comm sliceComm = std::get<2>(commInfo3D);
  MPI_Comm depthComm = std::get<3>(commInfo3D);
  size_t pGridCoordX = std::get<4>(commInfo3D);
  size_t pGridCoordY = std::get<5>(commInfo3D);
  size_t pGridCoordZ = std::get<6>(commInfo3D);

  // Need to perform the multiple steps to get our partition of matrixA
  U localDimensionN = matrixA.getNumColumnsLocal();		// no error check here, but hopefully 
  U localDimensionM = matrixA.getNumRowsLocal();		// no error check here, but hopefully 
  U globalDimensionN = matrixA.getNumColumnsGlobal();		// no error check here, but hopefully 
  std::vector<T>& dataA = matrixA.getVectorData();
  U sizeA = matrixA.getNumElems();
  std::vector<T> foreignA;	// dont fill with data first, because if root its a waste,
                                //   but need it to outside to get outside scope
  bool isRootRow = ((pGridCoordX == pGridCoordZ) ? true : false);
  bool isRootColumn = ((pGridCoordY == pGridCoordZ) ? true : false);

  // No optimization here I am pretty sure due to final result being symmetric, as it is cyclic and transpose isnt true as I have painfully found out before.
  BroadcastPanels((isRootRow ? dataA : foreignA), sizeA, isRootRow, pGridCoordZ, rowComm);
  blasEngineArgumentPackage_gemm<T> gemmPack1(blasEngineOrder::AblasColumnMajor, blasEngineTranspose::AblasTrans, blasEngineTranspose::AblasNoTrans, 1., 0.);

  // I don't think I can run syrk here, so I will use gemm. Maybe in the future after I have it correct I can experiment.
  blasEngine::_gemm((isRootRow ? &dataA[0] : &foreignA[0]), &dataA[0], matrixR.getRawData(), localDimensionN, localDimensionN,
    localDimensionM, localDimensionM, localDimensionM, localDimensionN, gemmPack1);

  MPI_Reduce((isRootColumn ? MPI_IN_PLACE : matrixR.getRawData()), matrixR.getRawData(), localDimensionN*localDimensionN, MPI_DATATYPE, MPI_SUM, pGridCoordZ, columnComm);
  MPI_Bcast(matrixR.getRawData(), localDimensionN*localDimensionN, MPI_DATATYPE, pGridCoordY, depthComm);

  // Create an extra matrix for R-inverse
  MatrixRType matrixRI(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN, matrixA.getNumColumnsGlobal(), matrixA.getNumColumnsGlobal(), true);

  std::pair<bool,std::vector<U>> baseCaseDimList = CFR3D::Factor(matrixR, matrixRI, inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier, 'U', commWorld, commInfo3D);

// For now, comment this out, because I am experimenting with using TriangularSolve TRSM instead of MM3D
//   But later on once it works, use an integer or something to have both available, important when benchmarking
  // Need to be careful here. matrixRI must be truly upper-triangular for this to be correct as I found out in 1D case.

  if (baseCaseDimList.first){
    blasEngineArgumentPackage_trmm<T> trmmPack1(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasRight, blasEngineUpLo::AblasUpper,
      blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
    MM3D::Multiply(matrixRI, matrixA, commWorld, commInfo3D, trmmPack1);
  }
  else{
    // Note: there are issues with serializing a square matrix into a rectangular. To bypass that,
    //        and also to communicate only the nonzeros, I will serialize into packed triangular buffers before calling TRSM
    Matrix<T,U,UpperTriangular,Distribution,Offload> rectR(std::vector<T>(), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN);
    Matrix<T,U,UpperTriangular,Distribution,Offload> rectRI(std::vector<T>(), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN);
    // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
    Serializer<Square,UpperTriangular>::Serialize(matrixR, rectR);
    Serializer<Square,UpperTriangular>::Serialize(matrixRI, rectRI);
    // alpha and beta fields don't matter. All I need from this struct are whether or not transposes are used.
    gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
    gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
    blasEngineArgumentPackage_trmm<T> trmmPackage(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasRight, blasEngineUpLo::AblasUpper,
      blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
    TRSM3D::iSolveUpperLeft(matrixA, rectR, rectRI, baseCaseDimList.second, gemmPack1, trmmPackage, commWorld, commInfo3D);
  }
  TAU_FSTOP(Factor3D_cqr);
}

template<typename MatrixAType, typename MatrixRType>
void CholeskyQR2::FactorTunable_cqr(MatrixAType& matrixA, MatrixRType& matrixR,
      size_t gridDimensionD, size_t gridDimensionC, MPI_Comm commWorld, std::tuple<MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm, MPI_Comm>& tunableCommunicators,
      std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D, size_t inverseCutOffMultiplier, size_t baseCaseMultiplier, size_t panelDimensionMultiplier){
  TAU_FSTART(FactorTunable_cqr);

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using Distribution = typename MatrixAType::DistributionType;
  using Offload = typename MatrixAType::OffloadType;

  MPI_Comm rowComm = std::get<0>(tunableCommunicators);
  MPI_Comm columnContigComm = std::get<1>(tunableCommunicators);
  MPI_Comm columnAltComm = std::get<2>(tunableCommunicators);
  MPI_Comm depthComm = std::get<3>(tunableCommunicators);
  MPI_Comm miniCubeComm = std::get<5>(tunableCommunicators);

  int worldRank;
  MPI_Comm_rank(commWorld, &worldRank);
  size_t sliceSize = gridDimensionD*gridDimensionC;
  #if defined(BLUEWATERS) || defined(STAMPEDE2)
  size_t helper = gridDimensionC*gridDimensionC;
  size_t pCoordZ = worldRank%gridDimensionC;
  size_t pCoordX = (worldRank%helper)/gridDimensionC;
  #else
  size_t pCoordX = worldRank%gridDimensionC;
  size_t pCoordZ = worldRank/sliceSize;
  #endif

  int columnContigRank;
  MPI_Comm_rank(columnContigComm, &columnContigRank);

  // Need to perform the multiple steps to get our partition of matrixA
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
  blasEngineArgumentPackage_gemm<T> gemmPack1(blasEngineOrder::AblasColumnMajor, blasEngineTranspose::AblasTrans, blasEngineTranspose::AblasNoTrans, 1., 0.);

  // I don't think I can run syrk here, so I will use gemm. Maybe in the future after I have it correct I can experiment.
  blasEngine::_gemm((isRootRow ? &dataA[0] : &foreignA[0]), &dataA[0], matrixR.getRawData(), localDimensionN, localDimensionN,
    localDimensionM, localDimensionM, localDimensionM, localDimensionN, gemmPack1);

  MPI_Reduce((isRootColumn ? MPI_IN_PLACE : matrixR.getRawData()), matrixR.getRawData(), localDimensionN*localDimensionN, MPI_DATATYPE, MPI_SUM, pCoordZ, columnContigComm);
  MPI_Allreduce(MPI_IN_PLACE, matrixR.getRawData(), localDimensionN*localDimensionN, MPI_DATATYPE,MPI_SUM, columnAltComm);
  MPI_Bcast(matrixR.getRawData(), localDimensionN*localDimensionN, MPI_DATATYPE, columnContigRank, depthComm);

  // Create an extra matrix for R-inverse
  MatrixRType matrixRI(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN, true);

  std::pair<bool,std::vector<U>> baseCaseDimList = CFR3D::Factor(matrixR, matrixRI, inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier, 'U', miniCubeComm, commInfo3D);

  if (baseCaseDimList.first){
    blasEngineArgumentPackage_trmm<T> trmmPack1(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasRight, blasEngineUpLo::AblasUpper,
      blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
    MM3D::Multiply(matrixRI, matrixA, miniCubeComm, commInfo3D, trmmPack1);
  }
  else{
    // Note: there are issues with serializing a square matrix into a rectangular. To bypass that,
    //        and also to communicate only the nonzeros, I will serialize into packed triangular buffers before calling TRSM
    Matrix<T,U,UpperTriangular,Distribution,Offload> rectR(std::vector<T>(), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN);
    Matrix<T,U,UpperTriangular,Distribution,Offload> rectRI(std::vector<T>(), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN);
    // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
    Serializer<Square,UpperTriangular>::Serialize(matrixR, rectR);
    Serializer<Square,UpperTriangular>::Serialize(matrixRI, rectRI);
    // alpha and beta fields don't matter. All I need from this struct are whether or not transposes are used.
    gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
    gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
    blasEngineArgumentPackage_trmm<T> trmmPackage(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasRight, blasEngineUpLo::AblasUpper,
      blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
    TRSM3D::iSolveUpperLeft(matrixA, rectR, rectRI, baseCaseDimList.second, gemmPack1, trmmPackage, miniCubeComm, commInfo3D);
  }
  TAU_FSTOP(FactorTunable_cqr);
}

template<typename T, typename U>
void CholeskyQR2::BroadcastPanels(std::vector<T>& data, U size, bool isRoot, size_t pGridCoordZ, MPI_Comm panelComm){
  TAU_FSTART(BroadcastPanels);
  if (isRoot){
    MPI_Bcast(&data[0], size, MPI_DATATYPE, pGridCoordZ, panelComm);
  }
  else{
    data.resize(size);
    MPI_Bcast(&data[0], size, MPI_DATATYPE, pGridCoordZ, panelComm);
  }
  TAU_FSTOP(BroadcastPanels);
}
