/* Author: Edward Hutter */

namespace qr{

template<class SerializeSymmetricPolicy>
template<typename MatrixAType, typename MatrixRType, typename ArgType, typename CommType>
void cacqr<SerializeSymmetricPolicy>::invoke_1d(MatrixAType& A, MatrixRType& R, typename MatrixRType::ScalarType* RI, ArgType&& args, CommType&& CommInfo){

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  // Changed from syrk to gemm
  U localDimensionM = A.num_rows_local();
  U localDimensionN = A.num_columns_local();
  blas::ArgPack_syrk<T> syrkPack(blas::Order::AblasColumnMajor, blas::UpLo::AblasUpper, blas::Transpose::AblasTrans, 1., 0.);
  blas::engine::_syrk(A.data(), R.data(), localDimensionN, localDimensionM, localDimensionM, localDimensionN, syrkPack);

  // MPI_Allreduce to replicate the dimensionY x dimensionY matrix on each processor
  SerializeSymmetricPolicy::invoke(R,CommInfo);

  lapack::ArgPack_potrf potrfArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper);
  lapack::ArgPack_trtri trtriArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper, lapack::Diag::AlapackNonUnit);
  lapack::engine::_potrf(R.data(), localDimensionN, localDimensionN, potrfArgs);
  std::memcpy(RI, R.data(), sizeof(T)*localDimensionN*localDimensionN);
  lapack::engine::_trtri(RI, localDimensionN, localDimensionN, trtriArgs);

  // Finish by performing local matrix multiplication Q = A*R^{-1}
  blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper, blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
  blas::engine::_trmm(RI, A.data(), localDimensionM, localDimensionN, localDimensionN, localDimensionM, trmmPack1);
}

template<class SerializeSymmetricPolicy>
template<typename MatrixAType, typename MatrixRType, typename ArgType, typename CommType>
void cacqr<SerializeSymmetricPolicy>::invoke_3d(MatrixAType& A, MatrixRType& R, MatrixRType& RI, ArgType&& args, CommType&& CommInfo){

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using Distribution = typename MatrixAType::DistributionType;
  using Offload = typename MatrixAType::OffloadType;

  // Need to perform the multiple steps to get our partition of A
  U localDimensionN = A.num_columns_local();		// no error check here, but hopefully 
  U localDimensionM = A.num_rows_local();		// no error check here, but hopefully 
  U globalDimensionN = A.num_columns_global();		// no error check here, but hopefully 
  U globalDimensionM = A.num_rows_global();		// no error check here, but hopefully 
  U sizeA = A.num_elems();
  bool isRootRow = ((CommInfo.x == CommInfo.z) ? true : false);
  bool isRootColumn = ((CommInfo.y == CommInfo.z) ? true : false);

  // No optimization here I am pretty sure due to final result being symmetric, as it is cyclic and transpose isnt true as I have painfully found out before.
  if (isRootRow) { A.swap(); }
  MPI_Bcast(A.scratch(), sizeA, mpi_type<T>::type, CommInfo.z, CommInfo.row);
  blas::ArgPack_gemm<T> gemmPack1(blas::Order::AblasColumnMajor, blas::Transpose::AblasTrans, blas::Transpose::AblasNoTrans, 1., 0.);
  if (isRootRow) { A.swap(); }

  // I don't think I can run syrk here, so I will use gemm. Maybe in the future after I have it correct I can experiment.
  blas::engine::_gemm((isRootRow ? A.data() : A.scratch()), A.data(), R.data(), localDimensionN, localDimensionN,
                      localDimensionM, localDimensionM, localDimensionM, localDimensionN, gemmPack1);

  MPI_Reduce((isRootColumn ? MPI_IN_PLACE : R.data()), R.data(), localDimensionN*localDimensionN, mpi_type<T>::type, MPI_SUM, CommInfo.z, CommInfo.column);
  MPI_Bcast(R.data(), localDimensionN*localDimensionN, mpi_type<T>::type, CommInfo.y, CommInfo.depth);

  auto baseCaseDimList = std::remove_reference<ArgType>::type::cholesky_inverse_type::invoke(R, RI,
                                                   args.cholesky_inverse_args, std::forward<CommType>(CommInfo));
// For now, comment this out, because I am experimenting with using TriangularSolve TRSM instead of summa
//   But later on once it works, use an integer or something to have both available, important when benchmarking
  // Need to be careful here. RI must be truly upper-triangular for this to be correct as I found out in 1D case.

  if (1){//baseCaseDimList.first){
    blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper,
      blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    matmult::summa::invoke(RI, A, std::forward<CommType>(CommInfo), trmmPack1);
  }
  else{
    assert(0);
    // Note: there are issues with serializing a square matrix into a rectangular. To bypass that,
    //        and also to communicate only the nonzeros, I will serialize into packed triangular buffers before calling TRSM
    matrix<T,U,uppertri,Distribution,Offload> rectR(globalDimensionN, globalDimensionN, CommInfo.c, CommInfo.c);
    matrix<T,U,uppertri,Distribution,Offload> rectRI(globalDimensionN, globalDimensionN, CommInfo.c, CommInfo.c);
    // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
    serialize<square,uppertri>::invoke(R, rectR);
    serialize<square,uppertri>::invoke(RI, rectRI);
    // alpha and beta fields don't matter. All I need from this struct are whether or not transposes are used.
    gemmPack1.transposeA = blas::Transpose::AblasNoTrans;
    gemmPack1.transposeB = blas::Transpose::AblasNoTrans;
    blas::ArgPack_trmm<T> trmmPackage(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper,
      blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    solve_upper_left(A, rectR, rectRI, std::forward<CommType>(CommInfo), baseCaseDimList.second, gemmPack1);
  }
}

template<class SerializeSymmetricPolicy>
template<typename MatrixAType, typename MatrixRType, typename ArgType, typename RectCommType, typename SquareCommType>
void cacqr<SerializeSymmetricPolicy>::invoke(MatrixAType& A, MatrixRType& R, MatrixRType& RI, ArgType&& args, RectCommType&& RectCommInfo, SquareCommType&& SquareCommInfo){

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using Distribution = typename MatrixAType::DistributionType;
  using Offload = typename MatrixAType::OffloadType;

  int columnContigRank;
  MPI_Comm_rank(RectCommInfo.column_contig, &columnContigRank);

  // Need to perform the multiple steps to get our partition of A
  U localDimensionM = A.num_rows_local();//globalDimensionM/gridDimensionD;
  U localDimensionN = A.num_columns_local();//globalDimensionN/gridDimensionC;
  U globalDimensionN = A.num_columns_global();
  U globalDimensionM = A.num_rows_global();//globalDimensionN/gridDimensionC;
  U sizeA = A.num_elems();
  bool isRootRow = ((RectCommInfo.x == RectCommInfo.z) ? true : false);
  bool isRootColumn = ((columnContigRank == RectCommInfo.z) ? true : false);

  // No optimization here I am pretty sure due to final result being symmetric, as it is cyclic and transpose isnt true as I have painfully found out before.
  if (isRootRow) { A.swap(); }
  MPI_Bcast(A.scratch(), sizeA, mpi_type<T>::type, RectCommInfo.z, RectCommInfo.row);
  blas::ArgPack_gemm<T> gemmPack1(blas::Order::AblasColumnMajor, blas::Transpose::AblasTrans, blas::Transpose::AblasNoTrans, 1., 0.);
  if (isRootRow) { A.swap(); }

  // I don't think I can run syrk here, so I will use gemm. Maybe in the future after I have it correct I can experiment.
  blas::engine::_gemm((isRootRow ? A.data() : A.scratch()), A.data(), R.data(), localDimensionN, localDimensionN,
                      localDimensionM, localDimensionM, localDimensionM, localDimensionN, gemmPack1);

  MPI_Reduce((isRootColumn ? MPI_IN_PLACE : R.data()), R.data(), localDimensionN*localDimensionN, mpi_type<T>::type, MPI_SUM, RectCommInfo.z, RectCommInfo.column_contig);
  MPI_Allreduce(MPI_IN_PLACE, R.data(), localDimensionN*localDimensionN, mpi_type<T>::type,MPI_SUM, RectCommInfo.column_alt);
  MPI_Bcast(R.data(), localDimensionN*localDimensionN, mpi_type<T>::type, columnContigRank, RectCommInfo.depth);

  auto baseCaseDimList = std::remove_reference<ArgType>::type::cholesky_inverse_type::invoke(R, RI,
                                                   args.cholesky_inverse_args, std::forward<SquareCommType>(SquareCommInfo));
  if (1){//baseCaseDimList.first){
    blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper,
      blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    matmult::summa::invoke(RI, A, std::forward<SquareCommType>(SquareCommInfo), trmmPack1);
  }
  else{
    assert(0);
    // Note: there are issues with serializing a square matrix into a rectangular. To bypass that,
    //        and also to communicate only the nonzeros, I will serialize into packed triangular buffers before calling TRSM
    matrix<T,U,uppertri,Distribution,Offload> rectR(globalDimensionN, globalDimensionN, SquareCommInfo.c, SquareCommInfo.c);
    matrix<T,U,uppertri,Distribution,Offload> rectRI(globalDimensionN, globalDimensionN, SquareCommInfo.c, SquareCommInfo.c);
    // Note: packedMatrix has no data right now. It will modify its buffers when serialized below
    serialize<square,uppertri>::invoke(R, rectR);
    serialize<square,uppertri>::invoke(RI, rectRI);
    // alpha and beta fields don't matter. All I need from this struct are whether or not transposes are used.
    gemmPack1.transposeA = blas::Transpose::AblasNoTrans;
    gemmPack1.transposeB = blas::Transpose::AblasNoTrans;
    blas::ArgPack_trmm<T> trmmPackage(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper,
      blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    solve_upper_left(A, rectR, rectRI, std::forward<SquareCommType>(SquareCommInfo), baseCaseDimList.second, gemmPack1);
  }
}

template<class SerializeSymmetricPolicy>
template<typename MatrixAType, typename MatrixRType, typename ArgType, typename CommType>
void cacqr<SerializeSymmetricPolicy>::invoke(MatrixAType& A, MatrixRType& R, ArgType&& args, CommType&& CommInfo){
  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  U globalDimensionN = A.num_columns_global();
  U localDimensionN = A.num_columns_local();//globalDimensionN/gridDimensionC;
  if (CommInfo.c == 1){
    std::vector<T> RI(localDimensionN*localDimensionN);
    invoke_1d(A,R,&RI[0],std::forward<ArgType>(args),std::forward<CommType>(CommInfo));
    return;
  }
  if (CommInfo.c == CommInfo.d){
    MatrixRType RI = R;
    invoke_3d(A,R,RI,std::forward<ArgType>(args),topo::square(CommInfo.cube,CommInfo.c));
    return;
  }
  else{
    MatrixRType RI = R;
    invoke(A,R,RI,std::forward<CommType>(CommInfo),topo::square(CommInfo.cube,CommInfo.c));
  }
}

template<class SerializeSymmetricPolicy>
template<typename T, typename U, typename ArgType, typename CommType>
std::pair<T*,T*> cacqr<SerializeSymmetricPolicy>::invoke(T* A, T* R, U localNumRows, U localNumColumns, U globalNumRows, U globalNumColumns, ArgType&& args, CommType&& CommInfo){
  //TODO: Test with non-power-of-2 global matrix dimensions
  matrix<T,U,rect,cyclic> mA(A,localNumColumns,localNumRows,globalNumColumns,globalNumRows,CommInfo.c,CommInfo.d);
  matrix<T,U,rect,cyclic> mR(A,localNumColumns,localNumColumns,globalNumColumns,globalNumColumns,CommInfo.c,CommInfo.c);
  invoke(mA,mR,std::forward<ArgType>(args),std::forward<CommType>(CommInfo));
  return std::make_pair(mA.get_data(),mR.get_data());
}

template<class SerializeSymmetricPolicy>
template<typename MatrixAType, typename MatrixRType, typename ArgType, typename CommType>
void cacqr2<SerializeSymmetricPolicy>::invoke_1d(MatrixAType& A, MatrixRType& R, ArgType&& args, CommType&& CommInfo){
  // We assume data is owned relative to a 1D processor grid, so every processor owns a chunk of data consisting of
  //   all columns and a block of rows.

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  U globalDimensionN = A.num_columns_global();
  U localDimensionN = A.num_columns_local();

  // Pre-allocate a buffer to use in each invoke_1d call to avoid allocating at each invocation
  std::vector<T> RI(localDimensionN*localDimensionN);
  cacqr<SerializeSymmetricPolicy>::invoke_1d(A, R, &RI[0], std::forward<ArgType>(args), std::forward<CommType>(CommInfo));
  R.swap();
  cacqr<SerializeSymmetricPolicy>::invoke_1d(A, R, &RI[0], std::forward<ArgType>(args), std::forward<CommType>(CommInfo));
  R.swap();

  // Later optimization - Serialize all 3 matrices into UpperTriangular first, then call this with those matrices, so we don't have to
  //   send half of the data!
  blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper,
    blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
  blas::engine::_trmm(R.scratch(), R.data(), localDimensionN, localDimensionN, localDimensionN, localDimensionN, trmmPack1);
}

template<class SerializeSymmetricPolicy>
template<typename MatrixAType, typename MatrixRType, typename ArgType, typename CommType>
void cacqr2<SerializeSymmetricPolicy>::invoke_3d(MatrixAType& A, MatrixRType& R, ArgType&& args, CommType&& CommInfo){

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  U globalDimensionN = A.num_columns_global();
  U localDimensionN = A.num_columns_local();		// no error check here, but hopefully 

  // Pre-allocate a buffer to use for Rinv in each invoke_1d call to avoid allocating at each invocation
  MatrixRType RI(globalDimensionN, globalDimensionN, CommInfo.c, CommInfo.c);
  MatrixRType R2(globalDimensionN, globalDimensionN, CommInfo.c, CommInfo.c);
  cacqr<SerializeSymmetricPolicy>::invoke_3d(A, R, RI, std::forward<ArgType>(args), std::forward<CommType>(CommInfo));
  cacqr<SerializeSymmetricPolicy>::invoke_3d(A, R2, RI, std::forward<ArgType>(args), std::forward<CommType>(CommInfo));

  blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper,
    blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
  // Later optimization - Serialize all 3 matrices into UpperTriangular first, then call this with those matrices, so we don't have to
  //   send half of the data!
  matmult::summa::invoke(R2, R, std::forward<CommType>(CommInfo), trmmPack1);
}

template<class SerializeSymmetricPolicy>
template<typename MatrixAType, typename MatrixRType, typename ArgType, typename CommType>
void cacqr2<SerializeSymmetricPolicy>::invoke(MatrixAType& A, MatrixRType& R, ArgType&& args, CommType&& CommInfo){
  if (CommInfo.c == 1){
    cacqr2::invoke_1d(A, R, std::forward<ArgType>(args), std::forward<CommType>(CommInfo));
    return;
  }
  if (CommInfo.c == CommInfo.d){
    cacqr2::invoke_3d(A, R, std::forward<ArgType>(args), topo::square(CommInfo.cube,CommInfo.c));
    return;
  }
  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  U globalDimensionN = A.num_columns_global();
  U localDimensionN = A.num_columns_local();

  auto SquareTopo = topo::square(CommInfo.cube,CommInfo.c);
  // Pre-allocate a buffer to use for Rinv in each invoke_1d call to avoid allocating at each invocation
  MatrixRType RI(globalDimensionN, globalDimensionN, SquareTopo.c, SquareTopo.c);
  MatrixRType R2(globalDimensionN, globalDimensionN, SquareTopo.c, SquareTopo.c);
  cacqr<SerializeSymmetricPolicy>::invoke(A, R, RI, std::forward<ArgType>(args), std::forward<CommType>(CommInfo), SquareTopo);
  cacqr<SerializeSymmetricPolicy>::invoke(A, R2, RI, std::forward<ArgType>(args), std::forward<CommType>(CommInfo), SquareTopo);

  blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper,
    blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
  // Later optimization - Serialize all 3 matrices into UpperTriangular first, then call this with those matrices, so we don't have to
  //   send half of the data!
  matmult::summa::invoke(R2, R, SquareTopo, trmmPack1);
}

template<class SerializeSymmetricPolicy>
template<typename MatrixAType, typename MatrixUType, typename MatrixUIType, typename CommType>
void cacqr<SerializeSymmetricPolicy>::solve_upper_left(MatrixAType& A, MatrixUType& U, MatrixUIType& UI, CommType&& CommInfo,
                                 std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                                 blas::ArgPack_gemm<typename MatrixAType::ScalarType>& gemmPackage){

  using T = typename MatrixAType::ScalarType;
  using V = typename MatrixAType::DimensionType;
  using StructureTri = typename MatrixUType::StructureType;
  using Distribution = typename MatrixAType::DistributionType;
  using Offload = typename MatrixAType::OffloadType;

  blas::ArgPack_trmm<T> trmmPackage(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasLower,
      blas::Transpose::AblasTrans, blas::Diag::AblasNonUnit, 1.);
  // to catch debugging issues, assert that this has at least one size
  assert(baseCaseDimList.size());

  V matAendX = A.num_columns_local();
  V matAendY = A.num_rows_local();
  V matUendX = U.num_columns_local();

  // Lets operate on individual columns at a time
  // Potential optimization 1): Don't use summa if the columns are too skinny in relation to the block size!
    // Or this could just be taken care of when we tune block sizes?
  // Potential optimization 2) Lots of serializing going on with each summa, this needs to be reduced.
  // Communicate A and U and UI immediately.
  // These 3 matrices should never need to be communicated again.
  //   B however will need to be AllReduced at each iteration so that final results can be summed and updated before next iteration

  V offset1 = 0; V offset2 = (baseCaseDimList.size() < 1 ? matAendX : baseCaseDimList[0]); V offset3 = 0;
  V arg1 = (gemmPackage.transposeB == blas::Transpose::AblasNoTrans ? offset1 : offset3);
  V arg2 = (gemmPackage.transposeB == blas::Transpose::AblasNoTrans ? matUendX : offset1);
  V arg3 = (gemmPackage.transposeB == blas::Transpose::AblasNoTrans ? offset3 : offset1);
  V arg4 = (gemmPackage.transposeB == blas::Transpose::AblasNoTrans ? offset1 : matUendX);
  V save1 = offset2-offset1;
  matrix<T,V,rect,Distribution,Offload> Upartition(nullptr, baseCaseDimList[0], arg4-arg3, CommInfo.d, CommInfo.d);
  matrix<T,V,StructureTri,Distribution,Offload> UIpartition(nullptr, save1, save1, CommInfo.d, CommInfo.d);

  for (V i=0; i<baseCaseDimList.size()/*numBlockColumns*/; i++){
    // Update the current column by accumulating the updates via MM
    gemmPackage.alpha = -1;
    gemmPackage.beta = 1.;
    // Only update once first panel is solved
    if (i>0){
      // As i increases, the size of these updates gets smaller.
      // Special handling. This might only work since the triangular matrix is square, which should be ok
      arg1 = (gemmPackage.transposeB == blas::Transpose::AblasNoTrans ? offset1 : offset3);
      arg2 = (gemmPackage.transposeB == blas::Transpose::AblasNoTrans ? matUendX : offset1);
      arg3 = (gemmPackage.transposeB == blas::Transpose::AblasNoTrans ? offset3 : offset1);
      arg4 = (gemmPackage.transposeB == blas::Transpose::AblasNoTrans ? offset1 : matUendX);

      serialize<StructureTri,rect>::invoke(U, Upartition, arg1, arg2, arg3, arg4);

//      matmult::summa::invoke(A.getRawData()+(offset3*matAendY), Upartition, A.getRawData()+(offset1*matAendY),
//        offset1-offset3, matAendY, arg2-arg1, arg4-arg3, matAendX-offset1, matAendY, std::forward<CommType>(CommInfo), gemmPackage);
    }

    // Solve via TRMM
    save1 = offset2-offset1;
    // New optimization: prevent this copy if we are doing TRSM only at the top level
    // Note: this change might be rendered useless now that I modified CFR3D.hpp with a similar optimization for that top level of TRSM
    if (baseCaseDimList.size() <= 1){
//      matmult::summa::invoke(UI, A.data()+(offset1*matAendY), save1, save1, save1, matAendY, std::forward<CommType>(CommInfo), trmmPackage);
    }
    else{
      serialize<StructureTri,StructureTri>::invoke(UI, UIpartition, offset1, offset2, offset1, offset2);
//      matmult::summa::invoke(UIpartition, A.getRawData()+(offset1*matAendY), save1, save1, save1, matAendY, std::forward<CommType>(CommInfo), trmmPackage);
    }
    if ((i+1) < baseCaseDimList.size()){
      // Update the offsets
      offset3 = offset1;
      offset1 = offset2;
      offset2 += baseCaseDimList[i+1];
    }
  }
}

template<class SerializeSymmetricPolicy>
template<typename T, typename U, typename ArgType, typename CommType>
std::pair<T*,T*> cacqr2<SerializeSymmetricPolicy>::invoke(T* A, T* R, U localNumRows, U localNumColumns, U globalNumRows, U globalNumColumns, ArgType&& args, CommType&& CommInfo){
  //TODO: Test with non-power-of-2 global matrix dimensions
  matrix<T,U,rect,cyclic> mA(A,localNumColumns,localNumRows,globalNumColumns,globalNumRows,CommInfo.c,CommInfo.d);
  matrix<T,U,rect,cyclic> mR(R,localNumColumns,localNumColumns,globalNumColumns,globalNumColumns,CommInfo.c,CommInfo.c);
  invoke(mA,mR,std::forward<ArgType>(args),std::forward<CommType>(CommInfo));
  return std::make_pair(mA.get_data(),mR.get_data());
}
}
