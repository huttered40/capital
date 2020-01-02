/* Author: Edward Hutter */

namespace qr{

template<class SerializePolicy, class IntermediatesPolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void cacqr<SerializePolicy,IntermediatesPolicy>::sweep_1d(MatrixType& A, MatrixType& R, MatrixType& RI, ArgType&& args, CommType&& CommInfo){

  using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType; using SP = SerializePolicy; using IP = IntermediatesPolicy;
  U localDimensionM = A.num_rows_local(); U localDimensionN = A.num_columns_local(); U globalDimensionN = A.num_columns_global();
  blas::ArgPack_syrk<T> syrkPack(blas::Order::AblasColumnMajor, blas::UpLo::AblasUpper, blas::Transpose::AblasTrans, 1., 0.);
  blas::engine::_syrk(A.data(), R.data(), localDimensionN, localDimensionM, localDimensionM, localDimensionN, syrkPack);

  // MPI_Allreduce to replicate the dimensionY x dimensionY matrix on each processor
  SP::gram(R,IP::invoke(args.policy_table1,std::make_pair(globalDimensionN,globalDimensionN)),CommInfo);

  lapack::ArgPack_potrf potrfArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper);
  lapack::ArgPack_trtri trtriArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper, lapack::Diag::AlapackNonUnit);
  lapack::engine::_potrf(R.data(), localDimensionN, localDimensionN, potrfArgs);
  std::memcpy(RI.data(), R.data(), sizeof(T)*localDimensionN*localDimensionN);
  lapack::engine::_trtri(RI.data(), localDimensionN, localDimensionN, trtriArgs);

  // Finish by performing local matrix multiplication Q = A*R^{-1}
  blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper, blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
  blas::engine::_trmm(RI.data(), A.data(), localDimensionM, localDimensionN, localDimensionN, localDimensionM, trmmPack1);
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void cacqr<SerializePolicy,IntermediatesPolicy>::sweep_3d(MatrixType& A, MatrixType& R, MatrixType& RI, ArgType&& args, CommType&& CommInfo){

  using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType; using SP = SerializePolicy; using IP = IntermediatesPolicy;

  // Need to perform the multiple steps to get our partition of A
  U localDimensionN = A.num_columns_local(); U localDimensionM = A.num_rows_local();
  U globalDimensionN = A.num_columns_global(); U globalDimensionM = A.num_rows_global(); U sizeA = A.num_elems();
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

  std::remove_reference<ArgType>::type::cholesky_inverse_type::invoke(R, RI, args.cholesky_inverse_args, std::forward<CommType>(CommInfo));
// For now, comment this out, because I am experimenting with using TriangularSolve TRSM instead of summa
//   But later on once it works, use an integer or something to have both available, important when benchmarking
  // Need to be careful here. RI must be truly upper-triangular for this to be correct as I found out in 1D case.

  if (args.cholesky_inverse_args.complete_inv){
    blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper,
                                    blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    matmult::summa::invoke(SP::invoke(RI,IP::invoke(args.policy_table1,std::make_pair(globalDimensionN,globalDimensionN))), A, std::forward<CommType>(CommInfo), trmmPack1);
  }
  else{
    solve(A,R,RI,std::forward<ArgType>(args),std::forward<CommType>(CommInfo));
  }
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename MatrixType, typename ArgType, typename RectCommType, typename SquareCommType>
void cacqr<SerializePolicy,IntermediatesPolicy>::sweep_tune(MatrixType& A, MatrixType& R, MatrixType& RI, ArgType&& args, RectCommType&& RectCommInfo, SquareCommType&& SquareCommInfo){

  using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType; using SP = SerializePolicy; using IP = IntermediatesPolicy;
  int columnContigRank; MPI_Comm_rank(RectCommInfo.column_contig, &columnContigRank);

  // Need to perform the multiple steps to get our partition of A
  U localDimensionM = A.num_rows_local(); U localDimensionN = A.num_columns_local();
  U globalDimensionN = A.num_columns_global(); U globalDimensionM = A.num_rows_global(); U sizeA = A.num_elems();
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

  std::remove_reference<ArgType>::type::cholesky_inverse_type::invoke(R, RI, args.cholesky_inverse_args, std::forward<SquareCommType>(SquareCommInfo));
  if (args.cholesky_inverse_args.complete_inv){
    blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper,
                                    blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    matmult::summa::invoke(SP::invoke(RI,IP::invoke(args.policy_table1,std::make_pair(globalDimensionN,globalDimensionN))), A, std::forward<SquareCommType>(SquareCommInfo), trmmPack1);
  }
  else{
    solve(A,R,RI,std::forward<ArgType>(args),std::forward<SquareCommType>(SquareCommInfo));
  }
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void cacqr<SerializePolicy,IntermediatesPolicy>::invoke_1d(MatrixType& A, MatrixType& R, ArgType&& args, CommType&& CommInfo){
  // We assume data is owned relative to a 1D processor grid, so every processor owns a chunk of data consisting of
  //   all columns and a block of rows.

  using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType; using SP = SerializePolicy; using IP = IntermediatesPolicy;
  U globalDimensionN = A.num_columns_global(); U localDimensionN = A.num_columns_local();

  // Pre-allocate a buffer to use in each invoke_1d call to avoid allocating at each invocation
  sweep_1d(A, R, IP::invoke(args.rect_table1,std::make_pair(globalDimensionN,globalDimensionN)), std::forward<ArgType>(args), std::forward<CommType>(CommInfo));
  if (args.num_iter>1){
    sweep_1d(A, IP::invoke(args.rect_table2,std::make_pair(globalDimensionN,globalDimensionN)), IP::invoke(args.rect_table1,std::make_pair(globalDimensionN,globalDimensionN)), std::forward<ArgType>(args), std::forward<CommType>(CommInfo));
    blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper,
                                    blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    blas::engine::_trmm(IP::invoke(args.rect_table2,std::make_pair(globalDimensionN,globalDimensionN)).data(), R.data(), localDimensionN, localDimensionN, localDimensionN, localDimensionN, trmmPack1);
  }
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void cacqr<SerializePolicy,IntermediatesPolicy>::invoke_3d(MatrixType& A, MatrixType& R, ArgType&& args, CommType&& CommInfo){

  using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType; using SP = SerializePolicy; using IP = IntermediatesPolicy;
  U globalDimensionN = A.num_columns_global(); U localDimensionN = A.num_columns_local();
  sweep_3d(A, R, IP::invoke(args.rect_table1,std::make_pair(globalDimensionN,globalDimensionN)), std::forward<ArgType>(args), std::forward<CommType>(CommInfo));
  if (args.num_iter>1){
    sweep_3d(A, IP::invoke(args.rect_table2,std::make_pair(globalDimensionN,globalDimensionN)), IP::invoke(args.rect_table1,std::make_pair(globalDimensionN,globalDimensionN)),
                           std::forward<ArgType>(args), std::forward<CommType>(CommInfo));
    blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper, blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    matmult::summa::invoke(SP::invoke(IP::invoke(args.rect_table2,std::make_pair(globalDimensionN,globalDimensionN)),IP::invoke(args.policy_table1,std::make_pair(globalDimensionN,globalDimensionN))),
                           SP::invoke(R, IP::invoke(args.policy_table2,std::make_pair(globalDimensionN,globalDimensionN))), std::forward<CommType>(CommInfo), trmmPack1);
    SP::complete(R, IP::invoke(args.policy_table2,std::make_pair(globalDimensionN,globalDimensionN)));
  }
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void cacqr<SerializePolicy,IntermediatesPolicy>::invoke(MatrixType& A, MatrixType& R, ArgType&& args, CommType&& CommInfo){
  using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType; using SP = SerializePolicy; using IP = IntermediatesPolicy;
  static_assert(std::is_same<typename MatrixType::StructureType,rect>::value,"qr::cacqr requires matrices of rect structure");
  U globalDimensionN = A.num_columns_global(); U localDimensionN = A.num_columns_local();
  if (std::is_same<SerializePolicy,policy::cacqr::Serialize>::value){ IP::init(args.policy_table1,std::make_pair(globalDimensionN,globalDimensionN),globalDimensionN,globalDimensionN,CommInfo.c,CommInfo.c); }
  IP::init(args.rect_table1,std::make_pair(globalDimensionN,globalDimensionN),globalDimensionN,globalDimensionN,CommInfo.c,CommInfo.c);
  IP::init(args.rect_table2,std::make_pair(globalDimensionN,globalDimensionN),globalDimensionN,globalDimensionN,CommInfo.c,CommInfo.c);
  if (CommInfo.c == 1){ invoke_1d(A, R, std::forward<ArgType>(args), std::forward<CommType>(CommInfo)); }
  else{
    if (std::is_same<SerializePolicy,policy::cacqr::Serialize>::value){ IP::init(args.policy_table2,std::make_pair(globalDimensionN,globalDimensionN),globalDimensionN,globalDimensionN,CommInfo.c,CommInfo.c); }
    if (!args.cholesky_inverse_args.complete_inv) simulate_solve(A,R,std::forward<ArgType>(args),std::forward<CommType>(CommInfo));
    if (CommInfo.c == CommInfo.d){ invoke_3d(A, R, std::forward<ArgType>(args), topo::square(CommInfo.cube,CommInfo.c)); }
    else{
      auto SquareTopo = topo::square(CommInfo.cube,CommInfo.c);
      sweep_tune(A, R, IP::invoke(args.rect_table1,std::make_pair(globalDimensionN,globalDimensionN)), std::forward<ArgType>(args), std::forward<CommType>(CommInfo), SquareTopo);
      if (args.num_iter>1){
        sweep_tune(A, IP::invoke(args.rect_table2,std::make_pair(globalDimensionN,globalDimensionN)), IP::invoke(args.rect_table1,std::make_pair(globalDimensionN,globalDimensionN)),
                                 std::forward<ArgType>(args), std::forward<CommType>(CommInfo), SquareTopo);

        blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasLeft, blas::UpLo::AblasUpper, blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
        matmult::summa::invoke(SP::invoke(IP::invoke(args.rect_table2,std::make_pair(globalDimensionN,globalDimensionN)),IP::invoke(args.policy_table1,std::make_pair(globalDimensionN,globalDimensionN))),
                               SP::invoke(R, IP::invoke(args.policy_table2,std::make_pair(globalDimensionN,globalDimensionN))), SquareTopo, trmmPack1);
        SP::complete(R, IP::invoke(args.policy_table2,std::make_pair(globalDimensionN,globalDimensionN)));
      }
    }
  }
  IP::flush(args.rect_table1[std::make_pair(globalDimensionN,globalDimensionN)]); IP::flush(args.rect_table2[std::make_pair(globalDimensionN,globalDimensionN)]);
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void cacqr<SerializePolicy,IntermediatesPolicy>::simulate_solve(MatrixType& A, MatrixType& R, ArgType&& args, CommType&& CommInfo){
  using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType; using SP = SerializePolicy; using IP = IntermediatesPolicy;
  U localDimensionN = R.num_rows_local(); U localDimensionM = A.num_rows_local();
  U split1 = (localDimensionN>>args.cholesky_inverse_args.split); split1 = util::get_next_power2(split1); U split2 = localDimensionN-split1;
  IP::init(args.rect_table1,std::make_pair(split1,localDimensionM),nullptr,split1,localDimensionM,CommInfo.c,CommInfo.c);
  IP::init(args.rect_table2,std::make_pair(split2,localDimensionM),nullptr,split2,localDimensionM,CommInfo.c,CommInfo.c);
  IP::init(args.rect_table2,std::make_pair(split2,split1),nullptr,split2,split1,CommInfo.c,CommInfo.c);
  IP::init(args.policy_table1,std::make_pair(split1,split1),nullptr,split1,split1,CommInfo.c,CommInfo.c);
  IP::init(args.policy_table1,std::make_pair(split2,split2),nullptr,split2,split2,CommInfo.c,CommInfo.c);
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void cacqr<SerializePolicy,IntermediatesPolicy>::solve(MatrixType& A, MatrixType& R, MatrixType& RI, ArgType&& args, CommType&& CommInfo){

  using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType; using SP = SerializePolicy; using IP = IntermediatesPolicy;
  U localDimensionN = R.num_rows_local(); U localDimensionM = A.num_rows_local();
  U split1 = (localDimensionN>>args.cholesky_inverse_args.split); split1 = util::get_next_power2(split1); U split2 = localDimensionN-split1;
  serialize<rect,rect>::invoke(A,IP::invoke(args.rect_table1,std::make_pair(split1,localDimensionM)),0,split1,0,localDimensionM);
  serialize<rect,rect>::invoke(A,IP::invoke(args.rect_table2,std::make_pair(split2,localDimensionM)),split1,localDimensionN,0,localDimensionM);
  serialize<rect,typename SP::structure>::invoke(RI,IP::invoke(args.policy_table1,std::make_pair(split1,split1)),0,split1,0,split1);
  serialize<rect,rect>::invoke(R,IP::invoke(args.rect_table2,std::make_pair(split2,split1)),split1,localDimensionN,0,split1);
  blas::ArgPack_gemm<T> gemmPack(blas::Order::AblasColumnMajor, blas::Transpose::AblasNoTrans, blas::Transpose::AblasNoTrans, 1., -1.);
  blas::ArgPack_trmm<T> trmmPack(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper, blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
  matmult::summa::invoke(IP::invoke(args.policy_table1,std::make_pair(split1,split1)),IP::invoke(args.rect_table1,std::make_pair(split1,localDimensionM)),
                         std::forward<CommType>(CommInfo), trmmPack);
  serialize<rect,typename SP::structure>::invoke(RI,IP::invoke(args.policy_table1,std::make_pair(split2,split2)),split1,localDimensionN,split1,localDimensionN);
  matmult::summa::invoke(IP::invoke(args.rect_table1,std::make_pair(split1,localDimensionM)),IP::invoke(args.rect_table2,std::make_pair(split2,split1)),
                         IP::invoke(args.rect_table2,std::make_pair(split2,localDimensionM)), std::forward<CommType>(CommInfo), gemmPack);
  matmult::summa::invoke(IP::invoke(args.policy_table1,std::make_pair(split2,split2)),IP::invoke(args.rect_table2,std::make_pair(split2,localDimensionM)),
                         std::forward<CommType>(CommInfo), trmmPack);
  serialize<rect,rect>::invoke(A,IP::invoke(args.rect_table1,std::make_pair(split1,localDimensionM)),0,split1,0,localDimensionM,true);
  serialize<rect,rect>::invoke(A,IP::invoke(args.rect_table2,std::make_pair(split2,localDimensionM)),split1,localDimensionN,0,localDimensionM,true);
  IP::flush(args.rect_table1[std::make_pair(split1,localDimensionM)]); IP::flush(args.rect_table2[std::make_pair(split2,localDimensionM)]);
  IP::flush(args.policy_table1[std::make_pair(split1,split1)]); IP::flush(args.policy_table1[std::make_pair(split2,split2)]);
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename ScalarType, typename DimensionType, typename ArgType, typename CommType>
std::pair<ScalarType*,ScalarType*> cacqr<SerializePolicy,IntermediatesPolicy>::invoke(ScalarType* A, ScalarType* R, DimensionType localNumRows, DimensionType localNumColumns,
                                                                                      DimensionType globalNumRows, DimensionType globalNumColumns, ArgType&& args, CommType&& CommInfo){
  //TODO: Test with non-power-of-2 global matrix dimensions
  matrix<ScalarType,DimensionType,rect> mA(A,localNumColumns,localNumRows,globalNumColumns,globalNumRows,CommInfo.c,CommInfo.d);
  matrix<ScalarType,DimensionType,rect> mR(A,localNumColumns,localNumColumns,globalNumColumns,globalNumColumns,CommInfo.c,CommInfo.c);
  invoke(mA,mR,std::forward<ArgType>(args),std::forward<CommType>(CommInfo));
  return std::make_pair(mA.get_data(),mR.get_data());
}
}
