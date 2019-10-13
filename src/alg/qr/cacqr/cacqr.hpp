/* Author: Edward Hutter */

namespace qr{

template<class SerializeSymmetricPolicy, class CholInvPolicy, class TrsmPolicy>
template<typename T, typename U>
void cacqr<SerializeSymmetricPolicy,CholInvPolicy,TrsmPolicy>::broadcast_panels(std::vector<T>& data, U size, bool isRoot, size_t pGridCoordZ, MPI_Comm panelComm){
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

template<class SerializeSymmetricPolicy, class CholInvPolicy, class TrsmPolicy>
template<typename MatrixAType, typename MatrixRType, typename CommType>
void cacqr<SerializeSymmetricPolicy,CholInvPolicy,TrsmPolicy>::invoke_1d(MatrixAType& MatrixA, MatrixRType& MatrixR, CommType&& CommInfo){
  TAU_FSTART(cacqr::invoke_1d);

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  // Changed from syrk to gemm
  U localDimensionM = MatrixA.getNumRowsLocal();
  U localDimensionN = MatrixA.getNumColumnsLocal();
  std::vector<T> localMMvec(localDimensionN*localDimensionN);
  blas::ArgPack_syrk<T> syrkPack(blas::Order::AblasColumnMajor, blas::UpLo::AblasUpper, blas::Transpose::AblasTrans, 1., 0.);
  blas::engine::_syrk(MatrixA.getRawData(), MatrixR.getRawData(), localDimensionN, localDimensionM, localDimensionM, localDimensionN, syrkPack);

  // MPI_Allreduce to replicate the dimensionY x dimensionY matrix on each processor
  policy::cacqr::ReduceSymmetricMatrix<SerializeSymmetricPolicy>::invoke(MatrixR,CommInfo);

  lapack::ArgPack_potrf potrfArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper);
  lapack::ArgPack_trtri trtriArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper, lapack::Diag::AlapackNonUnit);
  lapack::engine::_potrf(MatrixR.getRawData(), localDimensionN, localDimensionN, potrfArgs);
  std::vector<T> RI = MatrixR.getVectorData();
  lapack::engine::_trtri(&RI[0], localDimensionN, localDimensionN, trtriArgs);

  // Finish by performing local matrix multiplication Q = A*R^{-1}
  blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper, blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
  blas::engine::_trmm(&RI[0], MatrixA.getRawData(), localDimensionM, localDimensionN, localDimensionN, localDimensionM, trmmPack1);
  TAU_FSTOP(cacqr::invoke_1d);
}

template<class SerializeSymmetricPolicy, class CholInvPolicy, class TrsmPolicy>
template<typename MatrixAType, typename MatrixRType, typename CommType>
void cacqr<SerializeSymmetricPolicy,CholInvPolicy,TrsmPolicy>::invoke_3d(MatrixAType& MatrixA, MatrixRType& MatrixR, CommType&& CommInfo,
                      size_t inverseCutOffMultiplier){
  TAU_FSTART(cacqr::invoke_3d);

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using Distribution = typename MatrixAType::DistributionType;
  using Offload = typename MatrixAType::OffloadType;

  // Need to perform the multiple steps to get our partition of MatrixA
  U localDimensionN = MatrixA.getNumColumnsLocal();		// no error check here, but hopefully 
  U localDimensionM = MatrixA.getNumRowsLocal();		// no error check here, but hopefully 
  U globalDimensionN = MatrixA.getNumColumnsGlobal();		// no error check here, but hopefully 
  std::vector<T>& dataA = MatrixA.getVectorData();
  U sizeA = MatrixA.getNumElems();
  std::vector<T> foreignA;	// dont fill with data first, because if root its a waste,
                                //   but need it to outside to get outside scope
  bool isRootRow = ((CommInfo.x == CommInfo.z) ? true : false);
  bool isRootColumn = ((CommInfo.y == CommInfo.z) ? true : false);

  // No optimization here I am pretty sure due to final result being symmetric, as it is cyclic and transpose isnt true as I have painfully found out before.
  broadcast_panels((isRootRow ? dataA : foreignA), sizeA, isRootRow, CommInfo.z, CommInfo.row);
  blas::ArgPack_gemm<T> gemmPack1(blas::Order::AblasColumnMajor, blas::Transpose::AblasTrans, blas::Transpose::AblasNoTrans, 1., 0.);

  // I don't think I can run syrk here, so I will use gemm. Maybe in the future after I have it correct I can experiment.
  blas::engine::_gemm((isRootRow ? &dataA[0] : &foreignA[0]), &dataA[0], MatrixR.getRawData(), localDimensionN, localDimensionN,
    localDimensionM, localDimensionM, localDimensionM, localDimensionN, gemmPack1);

  MPI_Reduce((isRootColumn ? MPI_IN_PLACE : MatrixR.getRawData()), MatrixR.getRawData(), localDimensionN*localDimensionN, mpi_type<T>::type, MPI_SUM, CommInfo.z, CommInfo.column);
  MPI_Bcast(MatrixR.getRawData(), localDimensionN*localDimensionN, mpi_type<T>::type, CommInfo.y, CommInfo.depth);

  // Create an extra matrix for R-inverse
  MatrixRType MatrixRI(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN, MatrixA.getNumColumnsGlobal(), MatrixA.getNumColumnsGlobal(), true);

  std::pair<bool,std::vector<U>> baseCaseDimList = CholInvPolicy::invoke(MatrixR, MatrixRI, std::forward<CommType>(CommInfo),
                                                                             inverseCutOffMultiplier, 'U');

// For now, comment this out, because I am experimenting with using TriangularSolve TRSM instead of MM3D
//   But later on once it works, use an integer or something to have both available, important when benchmarking
  // Need to be careful here. MatrixRI must be truly upper-triangular for this to be correct as I found out in 1D case.

  if (baseCaseDimList.first){
    blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper,
      blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    matmult::summa::invoke(MatrixRI, MatrixA, std::forward<CommType>(CommInfo), trmmPack1);
  }
  else{
    // Note: there are issues with serializing a square matrix into a rectangular. To bypass that,
    //        and also to communicate only the nonzeros, I will serialize into packed triangular buffers before calling TRSM
    matrix<T,U,uppertri,Distribution,Offload> rectR(std::vector<T>(), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN);
    matrix<T,U,uppertri,Distribution,Offload> rectRI(std::vector<T>(), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN);
    // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
    serialize<square,uppertri>::invoke(MatrixR, rectR);
    serialize<square,uppertri>::invoke(MatrixRI, rectRI);
    // alpha and beta fields don't matter. All I need from this struct are whether or not transposes are used.
    gemmPack1.transposeA = blas::Transpose::AblasNoTrans;
    gemmPack1.transposeB = blas::Transpose::AblasNoTrans;
    blas::ArgPack_trmm<T> trmmPackage(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper,
      blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    TrsmPolicy::invoke(MatrixA, rectR, rectRI, std::forward<CommType>(CommInfo), 'U', 'L', baseCaseDimList.second, gemmPack1);
  }
  TAU_FSTOP(cacqr::invoke_3d);
}

template<class SerializeSymmetricPolicy, class CholInvPolicy, class TrsmPolicy>
template<typename MatrixAType, typename MatrixRType, typename CommType>
void cacqr<SerializeSymmetricPolicy,CholInvPolicy,TrsmPolicy>::invoke(MatrixAType& MatrixA, MatrixRType& MatrixR, CommType&& CommInfo,
                   size_t inverseCutOffMultiplier){
  invoke(MatrixA,MatrixR,std::forward<CommType>(CommInfo),topo::square(CommInfo.cube,CommInfo.c),inverseCutOffMultiplier);
}

template<class SerializeSymmetricPolicy, class CholInvPolicy, class TrsmPolicy>
template<typename MatrixAType, typename MatrixRType, typename RectCommType, typename SquareCommType>
void cacqr<SerializeSymmetricPolicy,CholInvPolicy,TrsmPolicy>::invoke(MatrixAType& MatrixA, MatrixRType& MatrixR, RectCommType&& RectCommInfo, SquareCommType&& SquareCommInfo,
                   size_t inverseCutOffMultiplier){
  TAU_FSTART(cacqr::invoke);

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using Distribution = typename MatrixAType::DistributionType;
  using Offload = typename MatrixAType::OffloadType;

  int columnContigRank;
  MPI_Comm_rank(RectCommInfo.column_contig, &columnContigRank);

  // Need to perform the multiple steps to get our partition of MatrixA
  U globalDimensionN = MatrixA.getNumColumnsGlobal();
  U localDimensionM = MatrixA.getNumRowsLocal();//globalDimensionM/gridDimensionD;
  U localDimensionN = MatrixA.getNumColumnsLocal();//globalDimensionN/gridDimensionC;
  std::vector<T>& dataA = MatrixA.getVectorData();
  U sizeA = MatrixA.getNumElems();
  std::vector<T> foreignA;	// dont fill with data first, because if root its a waste,
                                //   but need it to outside to get outside scope
  bool isRootRow = ((RectCommInfo.x == RectCommInfo.z) ? true : false);
  bool isRootColumn = ((columnContigRank == RectCommInfo.z) ? true : false);

  // No optimization here I am pretty sure due to final result being symmetric, as it is cyclic and transpose isnt true as I have painfully found out before.
  broadcast_panels((isRootRow ? dataA : foreignA), sizeA, isRootRow, RectCommInfo.z, RectCommInfo.row);
  blas::ArgPack_gemm<T> gemmPack1(blas::Order::AblasColumnMajor, blas::Transpose::AblasTrans, blas::Transpose::AblasNoTrans, 1., 0.);

  // I don't think I can run syrk here, so I will use gemm. Maybe in the future after I have it correct I can experiment.
  blas::engine::_gemm((isRootRow ? &dataA[0] : &foreignA[0]), &dataA[0], MatrixR.getRawData(), localDimensionN, localDimensionN,
    localDimensionM, localDimensionM, localDimensionM, localDimensionN, gemmPack1);

  MPI_Reduce((isRootColumn ? MPI_IN_PLACE : MatrixR.getRawData()), MatrixR.getRawData(), localDimensionN*localDimensionN, mpi_type<T>::type, MPI_SUM, RectCommInfo.z, RectCommInfo.column_contig);
  MPI_Allreduce(MPI_IN_PLACE, MatrixR.getRawData(), localDimensionN*localDimensionN, mpi_type<T>::type,MPI_SUM, RectCommInfo.column_alt);
  MPI_Bcast(MatrixR.getRawData(), localDimensionN*localDimensionN, mpi_type<T>::type, columnContigRank, RectCommInfo.depth);

  // Create an extra Matrix for R-inverse
  MatrixRType MatrixRI(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN, true);

  std::pair<bool,std::vector<U>> baseCaseDimList = CholInvPolicy::invoke(MatrixR, MatrixRI, std::forward<SquareCommType>(SquareCommInfo),
                                                                         inverseCutOffMultiplier, 'U');

  if (baseCaseDimList.first){
    blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper,
      blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    matmult::summa::invoke(MatrixRI, MatrixA, std::forward<SquareCommType>(SquareCommInfo), trmmPack1);
  }
  else{
    // Note: there are issues with serializing a square Matrix into a rectangular. To bypass that,
    //        and also to communicate only the nonzeros, I will serialize into packed triangular buffers before calling TRSM
    matrix<T,U,uppertri,Distribution,Offload> rectR(std::vector<T>(), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN);
    matrix<T,U,uppertri,Distribution,Offload> rectRI(std::vector<T>(), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN);
    // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
    serialize<square,uppertri>::invoke(MatrixR, rectR);
    serialize<square,uppertri>::invoke(MatrixRI, rectRI);
    // alpha and beta fields don't matter. All I need from this struct are whether or not transposes are used.
    gemmPack1.transposeA = blas::Transpose::AblasNoTrans;
    gemmPack1.transposeB = blas::Transpose::AblasNoTrans;
    blas::ArgPack_trmm<T> trmmPackage(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper,
      blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    TrsmPolicy::invoke(MatrixA, rectR, rectRI, std::forward<SquareCommType>(SquareCommInfo), 'U', 'L', baseCaseDimList.second, gemmPack1);
  }
  TAU_FSTOP(cacqr::invoke);
}

template<class SerializeSymmetricPolicy, class CholInvPolicy, class TrsmPolicy>
template<typename MatrixAType, typename MatrixRType, typename CommType>
void cacqr2<SerializeSymmetricPolicy,CholInvPolicy,TrsmPolicy>::invoke_1d(MatrixAType& MatrixA, MatrixRType& MatrixR, CommType&& CommInfo){
  TAU_FSTART(cacqr2::invoke_1d);
  // We assume data is owned relative to a 1D processor grid, so every processor owns a chunk of data consisting of
  //   all columns and a block of rows.

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  U globalDimensionN = MatrixA.getNumColumnsGlobal();
  U localDimensionN = MatrixA.getNumColumnsLocal();

  MatrixRType MatrixR2(std::vector<T>(localDimensionN*localDimensionN), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN, true);

  cacqr<SerializeSymmetricPolicy,CholInvPolicy>::invoke_1d(MatrixA, MatrixR, std::forward<CommType>(CommInfo));
  cacqr<SerializeSymmetricPolicy,CholInvPolicy>::invoke_1d(MatrixA, MatrixR2, std::forward<CommType>(CommInfo));

  // Later optimization - Serialize all 3 matrices into UpperTriangular first, then call this with those matrices, so we don't have to
  //   send half of the data!
  blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper,
    blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
  blas::engine::_trmm(MatrixR2.getRawData(), MatrixR.getRawData(), localDimensionN, localDimensionN,
    localDimensionN, localDimensionN, trmmPack1);
  TAU_FSTOP(cacqr2::invoke_1d);
}

template<class SerializeSymmetricPolicy, class CholInvPolicy, class TrsmPolicy>
template<typename MatrixAType, typename MatrixRType, typename CommType>
void cacqr2<SerializeSymmetricPolicy,CholInvPolicy,TrsmPolicy>::invoke_3d(MatrixAType& MatrixA, MatrixRType& MatrixR, CommType&& CommInfo,
                       size_t inverseCutOffMultiplier){
  TAU_FSTART(cacqr2::invoke_3d);

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  U globalDimensionN = MatrixA.getNumColumnsGlobal();
  U localDimensionN = MatrixA.getNumColumnsLocal();		// no error check here, but hopefully 

  MatrixRType MatrixR2(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN, true);
  cacqr<SerializeSymmetricPolicy,CholInvPolicy>::invoke_3d(MatrixA, MatrixR, std::forward<CommType>(CommInfo), inverseCutOffMultiplier);
  cacqr<SerializeSymmetricPolicy,CholInvPolicy>::invoke_3d(MatrixA, MatrixR2, std::forward<CommType>(CommInfo), inverseCutOffMultiplier);

  blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper,
    blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
  // Later optimization - Serialize all 3 matrices into UpperTriangular first, then call this with those matrices, so we don't have to
  //   send half of the data!
  matmult::summa::invoke(MatrixR2, MatrixR, std::forward<CommType>(CommInfo), trmmPack1);
  TAU_FSTOP(cacqr2::invoke_3d);
}

template<class SerializeSymmetricPolicy, class CholInvPolicy, class TrsmPolicy>
template<typename MatrixAType, typename MatrixRType, typename CommType>
void cacqr2<SerializeSymmetricPolicy,CholInvPolicy,TrsmPolicy>::invoke(MatrixAType& MatrixA, MatrixRType& MatrixR, CommType&& CommInfo,
                    size_t inverseCutOffMultiplier){
  TAU_FSTART(cacqr2::invoke);
  if (CommInfo.c == 1){
    cacqr2::invoke_1d(MatrixA, MatrixR, std::forward<CommType>(CommInfo));
    return;
  }
  if (CommInfo.c == CommInfo.d){
    cacqr2::invoke_3d(MatrixA, MatrixR, topo::square(CommInfo.cube,CommInfo.c), inverseCutOffMultiplier);
    return;
  }

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  U globalDimensionN = MatrixA.getNumColumnsGlobal();
  U localDimensionN = MatrixA.getNumColumnsLocal();//globalDimensionN/gridDimensionC;

  // Need to get the right global dimensions here, use a tunable package struct or something
  MatrixRType MatrixR2(std::vector<T>(localDimensionN*localDimensionN,0), localDimensionN, localDimensionN, globalDimensionN, globalDimensionN, true);
  auto SquareTopo = topo::square(CommInfo.cube,CommInfo.c);
  cacqr<SerializeSymmetricPolicy,CholInvPolicy>::invoke(MatrixA, MatrixR, std::forward<CommType>(CommInfo), SquareTopo, inverseCutOffMultiplier);
  cacqr<SerializeSymmetricPolicy,CholInvPolicy>::invoke(MatrixA, MatrixR2, std::forward<CommType>(CommInfo), SquareTopo, inverseCutOffMultiplier);

  blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper,
    blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
  // Later optimization - Serialize all 3 matrices into UpperTriangular first, then call this with those matrices, so we don't have to
  //   send half of the data!
  // TODO: Any other way to avoid building .. without complicating the interface?
  matmult::summa::invoke(MatrixR2, MatrixR, SquareTopo, trmmPack1);

  TAU_FSTOP(cacqr2::invoke);
}
}
