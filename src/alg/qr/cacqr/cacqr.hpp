/* Author: Edward Hutter */

namespace qr{

template<typename T, typename U>
void cacqr::broadcast_panels(std::vector<T>& data, U size, bool isRoot, size_t pGridCoordZ, MPI_Comm panelComm){
  TAU_FSTART(cacqr::broadcast_panels);
  if (isRoot){
    MPI_Bcast(&data[0], size, MPI_DATATYPE, pGridCoordZ, panelComm);
  }
  else{
    data.resize(size);
    MPI_Bcast(&data[0], size, MPI_DATATYPE, pGridCoordZ, panelComm);
  }
  TAU_FSTOP(cacqr::broadcast_panels);
}

template<typename MatrixAType, typename MatrixRType, typename CommType>
void cacqr::invoke_1d(MatrixAType& matrixA, MatrixRType& matrixR, CommType&& CommInfo){
  TAU_FSTART(cacqr::invoke_1d);

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
  MPI_Allreduce(MPI_IN_PLACE, matrixR.getRawData(), localDimensionN*localDimensionN, MPI_DATATYPE, MPI_SUM, CommInfo.world);

  lapackEngineArgumentPackage_potrf potrfArgs(lapackEngineOrder::AlapackColumnMajor, lapackEngineUpLo::AlapackUpper);
  lapackEngineArgumentPackage_trtri trtriArgs(lapackEngineOrder::AlapackColumnMajor, lapackEngineUpLo::AlapackUpper, lapackEngineDiag::AlapackNonUnit);
  lapackEngine::_potrf(matrixR.getRawData(), localDimensionN, localDimensionN, potrfArgs);
  std::vector<T> RI = matrixR.getVectorData();
  lapackEngine::_trtri(&RI[0], localDimensionN, localDimensionN, trtriArgs);

  // Finish by performing local matrix multiplication Q = A*R^{-1}
  blasEngineArgumentPackage_trmm<T> trmmPack1(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasRight, blasEngineUpLo::AblasUpper, blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
  blasEngine::_trmm(&RI[0], matrixA.getRawData(), localDimensionM, localDimensionN, localDimensionN, localDimensionM, trmmPack1);
  TAU_FSTOP(cacqr::invoke_1d);
}

template<typename MatrixAType, typename MatrixRType, typename CommType>
void cacqr::invoke_3d(MatrixAType& matrixA, MatrixRType& matrixR, CommType&& CommInfo,
                      size_t inverseCutOffMultiplier, size_t baseCaseMultiplier, size_t panelDimensionMultiplier){

  TAU_FSTART(cacqr::invoke_3d);

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using Distribution = typename MatrixAType::DistributionType;
  using Offload = typename MatrixAType::OffloadType;

  // Need to perform the multiple steps to get our partition of matrixA
  U localDimensionN = matrixA.getNumColumnsLocal();		// no error check here, but hopefully 
  U localDimensionM = matrixA.getNumRowsLocal();		// no error check here, but hopefully 
  U globalDimensionN = matrixA.getNumColumnsGlobal();		// no error check here, but hopefully 
  std::vector<T>& dataA = matrixA.getVectorData();
  U sizeA = matrixA.getNumElems();
  std::vector<T> foreignA;	// dont fill with data first, because if root its a waste,
                                //   but need it to outside to get outside scope
  bool isRootRow = ((CommInfo.x == CommInfo.z) ? true : false);
  bool isRootColumn = ((CommInfo.y == CommInfo.z) ? true : false);

  // No optimization here I am pretty sure due to final result being symmetric, as it is cyclic and transpose isnt true as I have painfully found out before.
  broadcast_panels((isRootRow ? dataA : foreignA), sizeA, isRootRow, CommInfo.z, CommInfo.row);
  blasEngineArgumentPackage_gemm<T> gemmPack1(blasEngineOrder::AblasColumnMajor, blasEngineTranspose::AblasTrans, blasEngineTranspose::AblasNoTrans, 1., 0.);

  // I don't think I can run syrk here, so I will use gemm. Maybe in the future after I have it correct I can experiment.
  blasEngine::_gemm((isRootRow ? &dataA[0] : &foreignA[0]), &dataA[0], matrixR.getRawData(), localDimensionN, localDimensionN,
    localDimensionM, localDimensionM, localDimensionM, localDimensionN, gemmPack1);

  MPI_Reduce((isRootColumn ? MPI_IN_PLACE : matrixR.getRawData()), matrixR.getRawData(), localDimensionN*localDimensionN, MPI_DATATYPE, MPI_SUM, CommInfo.z, CommInfo.column);
  MPI_Bcast(matrixR.getRawData(), localDimensionN*localDimensionN, MPI_DATATYPE, CommInfo.y, CommInfo.depth);

  // Create an extra matrix for R-inverse
  MatrixRType matrixRI(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN, matrixA.getNumColumnsGlobal(), matrixA.getNumColumnsGlobal(), true);

  std::pair<bool,std::vector<U>> baseCaseDimList = cholesky::cholinv::invoke(matrixR, matrixRI, std::forward<CommType>(CommInfo),
                                                                             inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier, 'U');

// For now, comment this out, because I am experimenting with using TriangularSolve TRSM instead of MM3D
//   But later on once it works, use an integer or something to have both available, important when benchmarking
  // Need to be careful here. matrixRI must be truly upper-triangular for this to be correct as I found out in 1D case.

  if (baseCaseDimList.first){
    blasEngineArgumentPackage_trmm<T> trmmPack1(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasRight, blasEngineUpLo::AblasUpper,
      blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
    matmult::summa::invoke(matrixRI, matrixA, std::forward<CommType>(CommInfo), trmmPack1);
  }
  else{
    // Note: there are issues with serializing a square matrix into a rectangular. To bypass that,
    //        and also to communicate only the nonzeros, I will serialize into packed triangular buffers before calling TRSM
    Matrix<T,U,UpperTriangular,Distribution,Offload> rectR(std::vector<T>(), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN);
    Matrix<T,U,UpperTriangular,Distribution,Offload> rectRI(std::vector<T>(), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN);
    // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
    serialize<Square,UpperTriangular>::invoke(matrixR, rectR);
    serialize<Square,UpperTriangular>::invoke(matrixRI, rectRI);
    // alpha and beta fields don't matter. All I need from this struct are whether or not transposes are used.
    gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
    gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
    blasEngineArgumentPackage_trmm<T> trmmPackage(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasRight, blasEngineUpLo::AblasUpper,
      blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
    trsm::diaginvert::invoke(matrixA, rectR, rectRI, std::forward<CommType>(CommInfo), 'U', 'L', baseCaseDimList.second, gemmPack1);
  }
  TAU_FSTOP(cacqr::invoke_3d);
}

template<typename MatrixAType, typename MatrixRType, typename CommType>
void cacqr::invoke(MatrixAType& matrixA, MatrixRType& matrixR, CommType&& CommInfo,
                   size_t inverseCutOffMultiplier, size_t baseCaseMultiplier, size_t panelDimensionMultiplier){
  invoke(matrixA,matrixR,std::forward<CommType>(CommInfo),topo::square(CommInfo.cube,CommInfo.c),inverseCutOffMultiplier,baseCaseMultiplier,panelDimensionMultiplier);
}

template<typename MatrixAType, typename MatrixRType, typename RectCommType, typename SquareCommType>
void cacqr::invoke(MatrixAType& matrixA, MatrixRType& matrixR, RectCommType&& RectCommInfo, SquareCommType&& SquareCommInfo,
                   size_t inverseCutOffMultiplier, size_t baseCaseMultiplier, size_t panelDimensionMultiplier){
  TAU_FSTART(cacqr::invoke);

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using Distribution = typename MatrixAType::DistributionType;
  using Offload = typename MatrixAType::OffloadType;

  int columnContigRank;
  MPI_Comm_rank(CommInfo.column_contig, &columnContigRank);

  // Need to perform the multiple steps to get our partition of matrixA
  U globalDimensionN = matrixA.getNumColumnsGlobal();
  U localDimensionM = matrixA.getNumRowsLocal();//globalDimensionM/gridDimensionD;
  U localDimensionN = matrixA.getNumColumnsLocal();//globalDimensionN/gridDimensionC;
  std::vector<T>& dataA = matrixA.getVectorData();
  U sizeA = matrixA.getNumElems();
  std::vector<T> foreignA;	// dont fill with data first, because if root its a waste,
                                //   but need it to outside to get outside scope
  bool isRootRow = ((CommInfo.x == CommInfo.z) ? true : false);
  bool isRootColumn = ((columnContigRank == pCoordZ) ? true : false);

  // No optimization here I am pretty sure due to final result being symmetric, as it is cyclic and transpose isnt true as I have painfully found out before.
  broadcast_panels((isRootRow ? dataA : foreignA), sizeA, isRootRow, CommInfo.z, CommInfo.row);
  blasEngineArgumentPackage_gemm<T> gemmPack1(blasEngineOrder::AblasColumnMajor, blasEngineTranspose::AblasTrans, blasEngineTranspose::AblasNoTrans, 1., 0.);

  // I don't think I can run syrk here, so I will use gemm. Maybe in the future after I have it correct I can experiment.
  blasEngine::_gemm((isRootRow ? &dataA[0] : &foreignA[0]), &dataA[0], matrixR.getRawData(), localDimensionN, localDimensionN,
    localDimensionM, localDimensionM, localDimensionM, localDimensionN, gemmPack1);

  MPI_Reduce((isRootColumn ? MPI_IN_PLACE : matrixR.getRawData()), matrixR.getRawData(), localDimensionN*localDimensionN, MPI_DATATYPE, MPI_SUM, CommInfo.z, CommInfo.column_contig);
  MPI_Allreduce(MPI_IN_PLACE, matrixR.getRawData(), localDimensionN*localDimensionN, MPI_DATATYPE,MPI_SUM, CommInfo.column_alt);
  MPI_Bcast(matrixR.getRawData(), localDimensionN*localDimensionN, MPI_DATATYPE, columnContigRank, CommInfo.depth);

  // Create an extra matrix for R-inverse
  MatrixRType matrixRI(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN, true);

  std::pair<bool,std::vector<U>> baseCaseDimList = cholesky::cholinv::invoke(matrixR, matrixRI, std::forward<SquareCommType>(SquareCommInfo),
                                                                             inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier, 'U');

  if (baseCaseDimList.first){
    blasEngineArgumentPackage_trmm<T> trmmPack1(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasRight, blasEngineUpLo::AblasUpper,
      blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
    matmult::summa::invoke(matrixRI, matrixA, std::forward<SquareCommType>(SquareCommInfo), trmmPack1);
  }
  else{
    // Note: there are issues with serializing a square matrix into a rectangular. To bypass that,
    //        and also to communicate only the nonzeros, I will serialize into packed triangular buffers before calling TRSM
    Matrix<T,U,UpperTriangular,Distribution,Offload> rectR(std::vector<T>(), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN);
    Matrix<T,U,UpperTriangular,Distribution,Offload> rectRI(std::vector<T>(), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN);
    // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
    serialize<Square,UpperTriangular>::invoke(matrixR, rectR);
    serialize<Square,UpperTriangular>::invoke(matrixRI, rectRI);
    // alpha and beta fields don't matter. All I need from this struct are whether or not transposes are used.
    gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
    gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
    blasEngineArgumentPackage_trmm<T> trmmPackage(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasRight, blasEngineUpLo::AblasUpper,
      blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
    trsm::diaginvert::invoke(matrixA, rectR, rectRI, std::forward<CommType>(SquareCommInfo), 'U', 'L', baseCaseDimList.second, gemmPack1);
  }
  TAU_FSTOP(cacqr::invoke);
}

template<typename MatrixAType, typename MatrixRType, typename CommType>
void cacqr2::invoke_1d(MatrixAType& matrixA, MatrixRType& matrixR, CommType&& CommInfo){
  TAU_FSTART(cacqr2::invoke_1d);
  // We assume data is owned relative to a 1D processor grid, so every processor owns a chunk of data consisting of
  //   all columns and a block of rows.

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  U globalDimensionN = matrixA.getNumColumnsGlobal();
  U localDimensionN = matrixA.getNumColumnsLocal();

  MatrixRType matrixR2(std::vector<T>(localDimensionN*localDimensionN), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN, true);

  cacqr::invoke_1d(matrixA, matrixR, std::forward<CommType>(CommInfo));
  cacqr::invoke_1d(matrixA, matrixR2, std::forward<CommType>(CommInfo));

  blasEngineArgumentPackage_trmm<T> trmmPack1(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasLeft, blasEngineUpLo::AblasUpper,
    blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
  blasEngine::_trmm(matrixR2.getRawData(), matrixR.getRawData(), localDimensionN, localDimensionN,
    localDimensionN, localDimensionN, trmmPack1);
  TAU_FSTOP(cacqr2::invoke_1d);
}

template<typename MatrixAType, typename MatrixRType, typename CommType>
void cacqr2::invoke_3d(MatrixAType& matrixA, MatrixRType& matrixR, CommType&& CommInfo,
                       size_t inverseCutOffMultiplier, size_t baseCaseMultiplier, size_t panelDimensionMultiplier){
  TAU_FSTART(cacqr2::invoke_3d);

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  U globalDimensionN = matrixA.getNumColumnsGlobal();
  U localDimensionN = matrixA.getNumColumnsLocal();		// no error check here, but hopefully 

  MatrixRType matrixR2(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN, true);
  cacqr::invoke_3d(matrixA, matrixR, std::forward<CommType>(CommInfo), inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier);
  cacqr::invoke_3d(matrixA, matrixR2, std::forward<CommType>(CommInfo), inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier);

  blasEngineArgumentPackage_trmm<T> trmmPack1(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasLeft, blasEngineUpLo::AblasUpper,
    blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
  // Later optimization - Serialize all 3 matrices into UpperTriangular first, then call this with those matrices, so we don't have to
  //   send half of the data!
  matmult::summa::invoke(matrixR2, matrixR, std::forward<CommType>(CommInfo), trmmPack1);
  TAU_FSTOP(cacqr2::invoke_3d);
}

template<typename MatrixAType, typename MatrixRType, typename CommType>
void cacqr2::invoke(MatrixAType& matrixA, MatrixRType& matrixR, CommType&& CommInfo,
                    size_t inverseCutOffMultiplier, size_t baseCaseMultiplier, size_t panelDimensionMultiplier){
  TAU_FSTART(cacqr2::invoke);
  if (CommType.c == 1){
    cacqr2::invoke_1d(matrixA, matrixR, std::forward<CommType>(CommInfo));
    return;
  }
  if (CommInfo.c == CommInfo.d){
    // TODO: Can CommInfo be reused?
    cacqr2::invoke_3d(matrixA, matrixR, topo::square(CommInfo.cube,CommInfo.c), inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier);
    return;
  }

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  U globalDimensionN = matrixA.getNumColumnsGlobal();
  U localDimensionN = matrixA.getNumColumnsLocal();//globalDimensionN/gridDimensionC;

  // Need to get the right global dimensions here, use a tunable package struct or something
  MatrixRType matrixR2(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN, true);
  auto SquareTopo = topo::square(CommInfo.cube,CommInfo.c);
  cacqr::invoke(matrixA, matrixR, std::forward<CommType>(CommInfo), SquareTopo, inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier);
  cacqr::invoke(matrixA, matrixR2, std::forward<CommType>(CommInfo), SquareTopo, inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier);

  blasEngineArgumentPackage_trmm<T> trmmPack1(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasLeft, blasEngineUpLo::AblasUpper,
    blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
  // Later optimization - Serialize all 3 matrices into UpperTriangular first, then call this with those matrices, so we don't have to
  //   send half of the data!
  // TODO: Any other way to avoid building .. without complicating the interface?
  matmult::summa:invoke(matrixR2, matrixR, SquareTopo, trmmPack1);

  TAU_FSTOP(cacqr2::invoke);
}
}
