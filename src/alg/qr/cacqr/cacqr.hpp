/* Author: Edward Hutter */

namespace qr{

template<typename T, typename U>
void cacqr::broadcast_panels(std::vector<T>& data, U size, bool isRoot, size_t pGridCoordZ, MPI_Comm panelComm){
  TAU_FSTART(cacqr::broadcast_panels);
  if (isRoot){
    MPI_Bcast(&data[0], size, mpi_type<T>::type, pGridCoordZ, panelComm);
  }
  else{
    data.resize(size);
    MPI_Bcast(&data[0], size, mpi_type<T>::type, pGridCoordZ, panelComm);
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
  blas::ArgPack_syrk<T> syrkPack(blas::Order::AblasColumnMajor, blas::UpLo::AblasUpper, blas::Transpose::AblasTrans, 1., 0.);
  blas::engine::_syrk(matrixA.getRawData(), matrixR.getRawData(), localDimensionN, localDimensionM, localDimensionM, localDimensionN, syrkPack);

  // MPI_Allreduce to replicate the dimensionY x dimensionY matrix on each processor
  // Optimization potential: only Allreduce half of this matrix because its symmetric
  //   but only try this later to see if it actually helps, because to do this, I will have to serialize and re-serialize. Would only make sense if dimensionX is huge.
  MPI_Allreduce(MPI_IN_PLACE, matrixR.getRawData(), localDimensionN*localDimensionN, mpi_type<T>::type, MPI_SUM, CommInfo.world);

  lapack::ArgPack_potrf potrfArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper);
  lapack::ArgPack_trtri trtriArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper, lapack::Diag::AlapackNonUnit);
  lapack::engine::_potrf(matrixR.getRawData(), localDimensionN, localDimensionN, potrfArgs);
  std::vector<T> RI = matrixR.getVectorData();
  lapack::engine::_trtri(&RI[0], localDimensionN, localDimensionN, trtriArgs);

  // Finish by performing local matrix multiplication Q = A*R^{-1}
  blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper, blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
  blas::engine::_trmm(&RI[0], matrixA.getRawData(), localDimensionM, localDimensionN, localDimensionN, localDimensionM, trmmPack1);
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
  blas::ArgPack_gemm<T> gemmPack1(blas::Order::AblasColumnMajor, blas::Transpose::AblasTrans, blas::Transpose::AblasNoTrans, 1., 0.);

  // I don't think I can run syrk here, so I will use gemm. Maybe in the future after I have it correct I can experiment.
  blas::engine::_gemm((isRootRow ? &dataA[0] : &foreignA[0]), &dataA[0], matrixR.getRawData(), localDimensionN, localDimensionN,
    localDimensionM, localDimensionM, localDimensionM, localDimensionN, gemmPack1);

  MPI_Reduce((isRootColumn ? MPI_IN_PLACE : matrixR.getRawData()), matrixR.getRawData(), localDimensionN*localDimensionN, mpi_type<T>::type, MPI_SUM, CommInfo.z, CommInfo.column);
  MPI_Bcast(matrixR.getRawData(), localDimensionN*localDimensionN, mpi_type<T>::type, CommInfo.y, CommInfo.depth);

  // Create an extra matrix for R-inverse
  MatrixRType matrixRI(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN, matrixA.getNumColumnsGlobal(), matrixA.getNumColumnsGlobal(), true);

  std::pair<bool,std::vector<U>> baseCaseDimList = cholesky::cholinv::invoke(matrixR, matrixRI, std::forward<CommType>(CommInfo),
                                                                             inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier, 'U');

// For now, comment this out, because I am experimenting with using TriangularSolve TRSM instead of MM3D
//   But later on once it works, use an integer or something to have both available, important when benchmarking
  // Need to be careful here. matrixRI must be truly upper-triangular for this to be correct as I found out in 1D case.

  if (baseCaseDimList.first){
    blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper,
      blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    matmult::summa::invoke(matrixRI, matrixA, std::forward<CommType>(CommInfo), trmmPack1);
  }
  else{
    // Note: there are issues with serializing a square matrix into a rectangular. To bypass that,
    //        and also to communicate only the nonzeros, I will serialize into packed triangular buffers before calling TRSM
    matrix<T,U,uppertri,Distribution,Offload> rectR(std::vector<T>(), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN);
    matrix<T,U,uppertri,Distribution,Offload> rectRI(std::vector<T>(), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN);
    // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
    serialize<square,uppertri>::invoke(matrixR, rectR);
    serialize<square,uppertri>::invoke(matrixRI, rectRI);
    // alpha and beta fields don't matter. All I need from this struct are whether or not transposes are used.
    gemmPack1.transposeA = blas::Transpose::AblasNoTrans;
    gemmPack1.transposeB = blas::Transpose::AblasNoTrans;
    blas::ArgPack_trmm<T> trmmPackage(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper,
      blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
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
  MPI_Comm_rank(RectCommInfo.column_contig, &columnContigRank);

  // Need to perform the multiple steps to get our partition of matrixA
  U globalDimensionN = matrixA.getNumColumnsGlobal();
  U localDimensionM = matrixA.getNumRowsLocal();//globalDimensionM/gridDimensionD;
  U localDimensionN = matrixA.getNumColumnsLocal();//globalDimensionN/gridDimensionC;
  std::vector<T>& dataA = matrixA.getVectorData();
  U sizeA = matrixA.getNumElems();
  std::vector<T> foreignA;	// dont fill with data first, because if root its a waste,
                                //   but need it to outside to get outside scope
  bool isRootRow = ((RectCommInfo.x == RectCommInfo.z) ? true : false);
  bool isRootColumn = ((columnContigRank == RectCommInfo.z) ? true : false);

  // No optimization here I am pretty sure due to final result being symmetric, as it is cyclic and transpose isnt true as I have painfully found out before.
  broadcast_panels((isRootRow ? dataA : foreignA), sizeA, isRootRow, RectCommInfo.z, RectCommInfo.row);
  blas::ArgPack_gemm<T> gemmPack1(blas::Order::AblasColumnMajor, blas::Transpose::AblasTrans, blas::Transpose::AblasNoTrans, 1., 0.);

  // I don't think I can run syrk here, so I will use gemm. Maybe in the future after I have it correct I can experiment.
  blas::engine::_gemm((isRootRow ? &dataA[0] : &foreignA[0]), &dataA[0], matrixR.getRawData(), localDimensionN, localDimensionN,
    localDimensionM, localDimensionM, localDimensionM, localDimensionN, gemmPack1);

  MPI_Reduce((isRootColumn ? MPI_IN_PLACE : matrixR.getRawData()), matrixR.getRawData(), localDimensionN*localDimensionN, mpi_type<T>::type, MPI_SUM, RectCommInfo.z, RectCommInfo.column_contig);
  MPI_Allreduce(MPI_IN_PLACE, matrixR.getRawData(), localDimensionN*localDimensionN, mpi_type<T>::type,MPI_SUM, RectCommInfo.column_alt);
  MPI_Bcast(matrixR.getRawData(), localDimensionN*localDimensionN, mpi_type<T>::type, columnContigRank, RectCommInfo.depth);

  // Create an extra matrix for R-inverse
  MatrixRType matrixRI(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN, true);

  std::pair<bool,std::vector<U>> baseCaseDimList = cholesky::cholinv::invoke(matrixR, matrixRI, std::forward<SquareCommType>(SquareCommInfo),
                                                                             inverseCutOffMultiplier, baseCaseMultiplier, panelDimensionMultiplier, 'U');

  if (baseCaseDimList.first){
    blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper,
      blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    matmult::summa::invoke(matrixRI, matrixA, std::forward<SquareCommType>(SquareCommInfo), trmmPack1);
  }
  else{
    // Note: there are issues with serializing a square matrix into a rectangular. To bypass that,
    //        and also to communicate only the nonzeros, I will serialize into packed triangular buffers before calling TRSM
    matrix<T,U,uppertri,Distribution,Offload> rectR(std::vector<T>(), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN);
    matrix<T,U,uppertri,Distribution,Offload> rectRI(std::vector<T>(), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN);
    // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
    serialize<square,uppertri>::invoke(matrixR, rectR);
    serialize<square,uppertri>::invoke(matrixRI, rectRI);
    // alpha and beta fields don't matter. All I need from this struct are whether or not transposes are used.
    gemmPack1.transposeA = blas::Transpose::AblasNoTrans;
    gemmPack1.transposeB = blas::Transpose::AblasNoTrans;
    blas::ArgPack_trmm<T> trmmPackage(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper,
      blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    trsm::diaginvert::invoke(matrixA, rectR, rectRI, std::forward<SquareCommType>(SquareCommInfo), 'U', 'L', baseCaseDimList.second, gemmPack1);
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

  blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper,
    blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
  blas::engine::_trmm(matrixR2.getRawData(), matrixR.getRawData(), localDimensionN, localDimensionN,
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

  blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper,
    blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
  // Later optimization - Serialize all 3 matrices into UpperTriangular first, then call this with those matrices, so we don't have to
  //   send half of the data!
  matmult::summa::invoke(matrixR2, matrixR, std::forward<CommType>(CommInfo), trmmPack1);
  TAU_FSTOP(cacqr2::invoke_3d);
}

template<typename MatrixAType, typename MatrixRType, typename CommType>
void cacqr2::invoke(MatrixAType& matrixA, MatrixRType& matrixR, CommType&& CommInfo,
                    size_t inverseCutOffMultiplier, size_t baseCaseMultiplier, size_t panelDimensionMultiplier){
  TAU_FSTART(cacqr2::invoke);
  if (CommInfo.c == 1){
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

  blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper,
    blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
  // Later optimization - Serialize all 3 matrices into UpperTriangular first, then call this with those matrices, so we don't have to
  //   send half of the data!
  // TODO: Any other way to avoid building .. without complicating the interface?
  matmult::summa::invoke(matrixR2, matrixR, SquareTopo, trmmPack1);

  TAU_FSTOP(cacqr2::invoke);
}
}