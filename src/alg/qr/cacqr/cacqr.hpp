/* Author: Edward Hutter */

namespace qr{

template<class SerializePolicy, class IntermediatesPolicy>
template<typename ArgType, typename CommType>
void cacqr<SerializePolicy,IntermediatesPolicy>::sweep_1d(ArgType& args, CommType&& CommInfo){
#ifdef FUNCTION_SYMBOLS
  CRITTER_START(CQR::sweep_1d);
#endif
  using T = typename ArgType::ScalarType; using SP = SerializePolicy; using IP = IntermediatesPolicy;
  auto localDimensionM = args.Q.num_rows_local(); auto localDimensionN = args.R.num_columns_local(); auto globalDimensionN = args.R.num_columns_global();
  auto& buffer = SP::buffer(args.R,IP::invoke(args.rect_table1,std::make_pair(globalDimensionN,globalDimensionN)));
  blas::ArgPack_syrk<T> syrkPack(blas::Order::AblasColumnMajor, blas::UpLo::AblasUpper, blas::Transpose::AblasTrans, 1., 0.);
  blas::engine::_syrk(args.Q.data(), buffer.data(), localDimensionN, localDimensionM, localDimensionM, localDimensionN, syrkPack);
  // MPI_Allreduce to replicate the gram matrix on each process
  SP::compute_gram(args.R,IP::invoke(args.rect_table1,std::make_pair(globalDimensionN,globalDimensionN)),CommInfo);
  lapack::ArgPack_potrf potrfArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper);
  lapack::ArgPack_trtri trtriArgs(lapack::Order::AlapackColumnMajor, lapack::UpLo::AlapackUpper, lapack::Diag::AlapackNonUnit);
  lapack::engine::_potrf(buffer.data(), localDimensionN, localDimensionN, potrfArgs);
  std::memcpy(buffer.scratch(), buffer.data(), sizeof(T)*localDimensionN*localDimensionN);
  lapack::engine::_trtri(buffer.scratch(), localDimensionN, localDimensionN, trtriArgs);
  // Finish by performing local matrix multiplication Q = A*R^{-1}
  blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper, blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
  blas::engine::_trmm(buffer.scratch(), args.Q.data(), localDimensionM, localDimensionN, localDimensionN, localDimensionM, trmmPack1);
#ifdef FUNCTION_SYMBOLS
  CRITTER_STOP(CQR::sweep_1d);
#endif
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename ArgType, typename CommType>
void cacqr<SerializePolicy,IntermediatesPolicy>::simulate_solve(ArgType& args, CommType&& CommInfo){
  using SP = SerializePolicy; using IP = IntermediatesPolicy;
  auto localDimensionN = args.R.num_rows_local(); auto localDimensionM = args.Q.num_rows_local();
  auto split1 = (localDimensionN>>args.cholesky_inverse_args.split); auto split2 = localDimensionN-split1;
  IP::init(args.rect_table1,std::make_pair(split1,localDimensionM),nullptr,split1,localDimensionM,CommInfo.c,CommInfo.c);
  IP::init(args.rect_table2,std::make_pair(split2,localDimensionM),nullptr,split2,localDimensionM,CommInfo.c,CommInfo.c);
  IP::init(args.rect_table2,std::make_pair(split2,split1),nullptr,split2,split1,CommInfo.c,CommInfo.c);
  IP::init(args.policy_table,std::make_pair(split1,split1),nullptr,split1,split1,CommInfo.c,CommInfo.c);
  IP::init(args.policy_table,std::make_pair(split2,split2),nullptr,split2,split2,CommInfo.c,CommInfo.c);
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename ArgType, typename CommType>
void cacqr<SerializePolicy,IntermediatesPolicy>::solve(ArgType& args, CommType&& CommInfo){
#ifdef FUNCTION_SYMBOLS
  CRITTER_START(CQR::solve);
#endif
  using T = typename ArgType::ScalarType; using SP = SerializePolicy; using IP = IntermediatesPolicy;
  auto localDimensionN = args.R.num_rows_local(); auto localDimensionM = args.Q.num_rows_local();
  auto split1 = (localDimensionN>>args.cholesky_inverse_args.split); auto split2 = localDimensionN-split1;
  serialize<rect,rect>::invoke(args.Q,IP::invoke(args.rect_table1,std::make_pair(split1,localDimensionM)),0,split1,0,localDimensionM,0,split1,0,localDimensionM);
  serialize<rect,rect>::invoke(args.Q,IP::invoke(args.rect_table2,std::make_pair(split2,localDimensionM)),split1,localDimensionN,0,localDimensionM,0,split2,0,localDimensionM);
  serialize<uppertri,uppertri>::invoke(args.cholesky_inverse_args.Rinv,IP::invoke(args.policy_table,std::make_pair(split1,split1)),0,split1,0,split1,0,split1,0,split1);
  serialize<rect,rect>::invoke(args.cholesky_inverse_args.R,IP::invoke(args.rect_table2,std::make_pair(split2,split1)),split1,localDimensionN,0,split1,0,split2,0,split1);
  blas::ArgPack_gemm<T> gemmPack(blas::Order::AblasColumnMajor, blas::Transpose::AblasNoTrans, blas::Transpose::AblasNoTrans, 1., -1.);
  blas::ArgPack_trmm<T> trmmPack(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper, blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
  matmult::summa::invoke(IP::invoke(args.policy_table,std::make_pair(split1,split1)),IP::invoke(args.rect_table1,std::make_pair(split1,localDimensionM)),
                         std::forward<CommType>(CommInfo), trmmPack);
  serialize<uppertri,uppertri>::invoke(args.cholesky_inverse_args.Rinv,IP::invoke(args.policy_table,std::make_pair(split2,split2)),split1,localDimensionN,split1,localDimensionN,0,split2,0,split2);
  matmult::summa::invoke(IP::invoke(args.rect_table1,std::make_pair(split1,localDimensionM)),IP::invoke(args.rect_table2,std::make_pair(split2,split1)),
                         IP::invoke(args.rect_table2,std::make_pair(split2,localDimensionM)), std::forward<CommType>(CommInfo), gemmPack);
  matmult::summa::invoke(IP::invoke(args.policy_table,std::make_pair(split2,split2)),IP::invoke(args.rect_table2,std::make_pair(split2,localDimensionM)),
                         std::forward<CommType>(CommInfo), trmmPack);
  serialize<rect,rect>::invoke(IP::invoke(args.rect_table1,std::make_pair(split1,localDimensionM)),args.Q,0,split1,0,localDimensionM,0,split1,0,localDimensionM);
  serialize<rect,rect>::invoke(IP::invoke(args.rect_table2,std::make_pair(split2,localDimensionM)),args.Q,0,split2,0,localDimensionM,split1,localDimensionN,0,localDimensionM);
  IP::flush(args.rect_table1[std::make_pair(split1,localDimensionM)]); IP::flush(args.rect_table2[std::make_pair(split2,localDimensionM)]);
  IP::flush(args.policy_table[std::make_pair(split1,split1)]); IP::flush(args.policy_table[std::make_pair(split2,split2)]);
#ifdef FUNCTION_SYMBOLS
  CRITTER_STOP(CQR::solve);
#endif
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename ArgType, typename CommType>
void cacqr<SerializePolicy,IntermediatesPolicy>::sweep_3d(ArgType& args, CommType&& CommInfo){
#ifdef FUNCTION_SYMBOLS
  CRITTER_START(CQR::sweep_3d);
#endif
  using T = typename ArgType::ScalarType; using SP = SerializePolicy; using IP = IntermediatesPolicy;
#ifdef ALGORITHMIC_SYMBOLS
  CRITTER_START(CQR::gram);
#endif
  // Need to perform the multiple steps to obtain partition of A
  auto localDimensionN = args.Q.num_columns_local(); auto localDimensionM = args.Q.num_rows_local();
  auto globalDimensionN = args.Q.num_columns_global(); auto globalDimensionM = args.Q.num_rows_global(); auto sizeA = args.Q.num_elems();
  auto& buffer = SP::buffer(args.R,IP::invoke(args.rect_table1,std::make_pair(globalDimensionN,globalDimensionN)));
  bool isRootRow = ((CommInfo.x == CommInfo.z) ? true : false);
  bool isRootColumn = ((CommInfo.y == CommInfo.z) ? true : false);
  if (isRootRow) { args.Q.swap(); }
  MPI_Bcast(args.Q.scratch(), sizeA, mpi_type<T>::type, CommInfo.z, CommInfo.row);
  blas::ArgPack_gemm<T> gemmPack1(blas::Order::AblasColumnMajor, blas::Transpose::AblasTrans, blas::Transpose::AblasNoTrans, 1., 0.);
  if (isRootRow) { args.Q.swap(); }
  blas::engine::_gemm((isRootRow ? args.Q.data() : args.Q.scratch()), args.Q.data(), buffer.data(), localDimensionN, localDimensionN,
                      localDimensionM, localDimensionM, localDimensionM, localDimensionN, gemmPack1);
  SP::transfer_start(args.R,buffer);
  MPI_Reduce((isRootColumn ? MPI_IN_PLACE : args.R.data()), args.R.data(), args.R.num_elems(), mpi_type<T>::type, MPI_SUM, CommInfo.z, CommInfo.column);
  MPI_Bcast(args.R.data(), args.R.num_elems(), mpi_type<T>::type, CommInfo.y, CommInfo.depth);
#ifdef ALGORITHMIC_SYMBOLS
  CRITTER_STOP(CQR::gram);
#endif
  std::remove_reference<ArgType>::type::cholesky_inverse_type::factor(args.R, args.cholesky_inverse_args, std::forward<CommType>(CommInfo));
  SP::transfer_end(args.R);
#ifdef ALGORITHMIC_SYMBOLS
  CRITTER_START(CQR::formR);
#endif
  if (args.cholesky_inverse_args.complete_inv){
    blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper,
                                    blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    matmult::summa::invoke(args.cholesky_inverse_args.Rinv,args.Q, std::forward<CommType>(CommInfo), trmmPack1);
  }
  else{ solve(args,std::forward<CommType>(CommInfo)); }
#ifdef ALGORITHMIC_SYMBOLS
  CRITTER_STOP(CQR::formR);
#endif
#ifdef FUNCTION_SYMBOLS
  CRITTER_STOP(CQR::sweep_3d);
#endif
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename ArgType, typename RectCommType, typename SquareCommType>
void cacqr<SerializePolicy,IntermediatesPolicy>::sweep_tune(ArgType& args, RectCommType&& RectCommInfo, SquareCommType&& SquareCommInfo){
#ifdef FUNCTION_SYMBOLS
  CRITTER_START(CQR::sweep_tune);
#endif
  using T = typename ArgType::ScalarType; using SP = SerializePolicy; using IP = IntermediatesPolicy;
  int columnContigRank; MPI_Comm_rank(RectCommInfo.column_contig, &columnContigRank);

#ifdef ALGORITHMIC_SYMBOLS
  CRITTER_START(CQR::gram);
#endif
  // Need to perform the multiple steps to obtain partition of A
  auto localDimensionM = args.Q.num_rows_local(); auto localDimensionN = args.Q.num_columns_local();
  auto globalDimensionN = args.Q.num_columns_global(); auto globalDimensionM = args.Q.num_rows_global(); auto sizeA = args.Q.num_elems();
  auto& buffer = SP::buffer(args.R,IP::invoke(args.rect_table1,std::make_pair(globalDimensionN,globalDimensionN)));
  bool isRootRow = ((RectCommInfo.x == RectCommInfo.z) ? true : false);
  bool isRootColumn = ((columnContigRank == RectCommInfo.z) ? true : false);
  if (isRootRow) { args.Q.swap(); }
  MPI_Bcast(args.Q.scratch(), sizeA, mpi_type<T>::type, RectCommInfo.z, RectCommInfo.row);
  blas::ArgPack_gemm<T> gemmPack1(blas::Order::AblasColumnMajor, blas::Transpose::AblasTrans, blas::Transpose::AblasNoTrans, 1., 0.);
  if (isRootRow) { args.Q.swap(); }
  blas::engine::_gemm((isRootRow ? args.Q.data() : args.Q.scratch()), args.Q.data(), buffer.data(), localDimensionN, localDimensionN,
                      localDimensionM, localDimensionM, localDimensionM, localDimensionN, gemmPack1);
  SP::transfer_start(args.R,buffer);
  MPI_Reduce((isRootColumn ? MPI_IN_PLACE : args.R.data()), args.R.data(), args.R.num_elems(), mpi_type<T>::type, MPI_SUM, RectCommInfo.z, RectCommInfo.column_contig);
  MPI_Allreduce(MPI_IN_PLACE, args.R.data(), args.R.num_elems(), mpi_type<T>::type,MPI_SUM, RectCommInfo.column_alt);
  MPI_Bcast(args.R.data(), args.R.num_elems(), mpi_type<T>::type, columnContigRank, RectCommInfo.depth);
#ifdef ALGORITHMIC_SYMBOLS
  CRITTER_STOP(CQR::gram);
#endif
  std::remove_reference<ArgType>::type::cholesky_inverse_type::factor(args.R, args.cholesky_inverse_args, std::forward<SquareCommType>(SquareCommInfo));
  SP::transfer_end(args.R);
#ifdef ALGORITHMIC_SYMBOLS
  CRITTER_START(CQR::formR);
#endif
  if (args.cholesky_inverse_args.complete_inv){
    blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper,
                                    blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    matmult::summa::invoke(args.cholesky_inverse_args.Rinv,args.Q, std::forward<SquareCommType>(SquareCommInfo), trmmPack1);
  }
  else{ solve(args,std::forward<SquareCommType>(SquareCommInfo)); }
#ifdef ALGORITHMIC_SYMBOLS
  CRITTER_STOP(CQR::formR);
#endif
#ifdef FUNCTION_SYMBOLS
  CRITTER_STOP(CQR::sweep_tune);
#endif
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename ArgType, typename CommType>
void cacqr<SerializePolicy,IntermediatesPolicy>::invoke_1d(ArgType& args, CommType&& CommInfo){
#ifdef FUNCTION_SYMBOLS
  CRITTER_START(CQR::invoke_1d);
#endif
  using T = typename ArgType::ScalarType; using SP = SerializePolicy; using IP = IntermediatesPolicy;
  auto globalDimensionN = args.R.num_columns_global(); auto localDimensionN = args.R.num_columns_local();
  sweep_1d(args, std::forward<CommType>(CommInfo));
  if (args.num_iter>1){
    SP::save_R_1d(args.R,IP::invoke(args.rect_table1,std::make_pair(globalDimensionN,globalDimensionN)));
    sweep_1d(args, std::forward<CommType>(CommInfo));
    blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper, blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    blas::engine::_trmm(SP::retrieve_intermediate_R_1d(args.R,IP::invoke(args.rect_table1,std::make_pair(globalDimensionN,globalDimensionN))),
                        SP::retrieve_final_R_1d(args.R,IP::invoke(args.rect_table1,std::make_pair(globalDimensionN,globalDimensionN))),
                        localDimensionN, localDimensionN, localDimensionN, localDimensionN, trmmPack1);
    SP::complete_1d(args.R,IP::invoke(args.rect_table1,std::make_pair(globalDimensionN,globalDimensionN)));
  }
#ifdef FUNCTION_SYMBOLS
  CRITTER_STOP(CQR::invoke_1d);
#endif
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename ArgType, typename CommType>
void cacqr<SerializePolicy,IntermediatesPolicy>::invoke_3d(ArgType& args, CommType&& CommInfo){
#ifdef FUNCTION_SYMBOLS
  CRITTER_START(CQR::invoke_3d);
#endif
  using T = typename ArgType::ScalarType; using SP = SerializePolicy; using IP = IntermediatesPolicy;
  auto globalDimensionN = args.Q.num_columns_global(); auto localDimensionN = args.Q.num_columns_local();
  sweep_3d(args, std::forward<CommType>(CommInfo));
  if (args.num_iter>1){
    SP::save_R_3d(args.cholesky_inverse_args.R,args.R,IP::invoke(args.rect_table1,std::make_pair(globalDimensionN,globalDimensionN)));
    sweep_3d(args, std::forward<CommType>(CommInfo));
    blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper, blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
    if (std::is_same<typename std::remove_reference<ArgType>::type::cholesky_inverse_type::SP,cholesky::policy::cholinv::NoSerialize>::value) { util::remove_triangle_local(args.cholesky_inverse_args.R, CommInfo.x, CommInfo.y, CommInfo.c, 'U'); }
    matmult::summa::invoke(SP::retrieve_intermediate_R_3d(args.R,IP::invoke(args.rect_table1,std::make_pair(globalDimensionN,globalDimensionN))), args.cholesky_inverse_args.R, std::forward<CommType>(CommInfo), trmmPack1);
  }
  serialize<uppertri,uppertri>::invoke(args.cholesky_inverse_args.R,args.R,0,localDimensionN,0,localDimensionN,0,localDimensionN,0,localDimensionN);
#ifdef FUNCTION_SYMBOLS
  CRITTER_STOP(CQR::invoke_3d);
#endif
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void cacqr<SerializePolicy,IntermediatesPolicy>::factor(const MatrixType& A, ArgType& args, CommType&& CommInfo){
  CRITTER_START(CQR::factor);
  using T = typename MatrixType::ScalarType; using SP = SerializePolicy; using IP = IntermediatesPolicy;
  static_assert(std::is_same<typename MatrixType::StructureType,rect>::value,"qr::cacqr requires matrices of rect structure");
  auto globalDimensionN = A.num_columns_global(); auto globalDimensionM = A.num_rows_global(); auto localDimensionN = A.num_columns_local(); auto localDimensionM = A.num_rows_local();
  args.Q._register_(globalDimensionN,globalDimensionM,CommInfo.c,CommInfo.d);
  args.R._register_(globalDimensionN,globalDimensionN,CommInfo.c,CommInfo.c);
  serialize<rect,rect>::invoke(A,args.Q,0,localDimensionN,0,localDimensionM,0,localDimensionN,0,localDimensionM);

  IP::init(args.rect_table1,std::make_pair(globalDimensionN,globalDimensionN),globalDimensionN,globalDimensionN,CommInfo.c,CommInfo.c);
  if (CommInfo.c == 1){ invoke_1d(args, std::forward<CommType>(CommInfo)); }
  else{
    if (!args.cholesky_inverse_args.complete_inv) simulate_solve(args,std::forward<CommType>(CommInfo));
    if (CommInfo.c == CommInfo.d){ invoke_3d(args, topo::square(CommInfo.cube,CommInfo.c,CommInfo.layout,CommInfo.num_chunks)); }
    else{
      auto SquareTopo = topo::square(CommInfo.cube,CommInfo.c,CommInfo.layout,CommInfo.num_chunks);
      sweep_tune(args, std::forward<CommType>(CommInfo), SquareTopo);
      if (args.num_iter>1){
        SP::save_R_3d(args.cholesky_inverse_args.R,args.R,IP::invoke(args.rect_table1,std::make_pair(globalDimensionN,globalDimensionN)));
        sweep_tune(args, std::forward<CommType>(CommInfo), SquareTopo);
        blas::ArgPack_trmm<T> trmmPack1(blas::Order::AblasColumnMajor, blas::Side::AblasRight, blas::UpLo::AblasUpper, blas::Transpose::AblasNoTrans, blas::Diag::AblasNonUnit, 1.);
        if (std::is_same<typename std::remove_reference<ArgType>::type::cholesky_inverse_type::SP,cholesky::policy::cholinv::NoSerialize>::value) { util::remove_triangle_local(args.cholesky_inverse_args.R, SquareTopo.x, SquareTopo.y, SquareTopo.c, 'U'); }
        matmult::summa::invoke(SP::retrieve_intermediate_R_3d(args.R,IP::invoke(args.rect_table1,std::make_pair(globalDimensionN,globalDimensionN))), args.cholesky_inverse_args.R, SquareTopo, trmmPack1);
      }
      serialize<uppertri,uppertri>::invoke(args.cholesky_inverse_args.R,args.R,0,localDimensionN,0,localDimensionN,0,localDimensionN,0,localDimensionN);
    }
  }
  IP::flush(args.rect_table1[std::make_pair(globalDimensionN,globalDimensionN)]);
  CRITTER_STOP(CQR::factor);
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename ArgType, typename CommType>
matrix<typename ArgType::ScalarType,typename ArgType::DimensionType,rect> cacqr<SerializePolicy,IntermediatesPolicy>::construct_Q(ArgType& args, CommType&& CommInfo){
  CRITTER_START(qr::cacqr::construct_Q);
  auto localDimensionM = args.Q.num_rows_local(); auto localDimensionN = args.Q.num_columns_local();
  matrix<typename ArgType::ScalarType,typename ArgType::DimensionType,rect> ret(args.Q.num_columns_global(),args.Q.num_rows_global(),CommInfo.c, CommInfo.d);
  serialize<rect,rect>::invoke(args.Q, ret,0,localDimensionN,0,localDimensionM,0,localDimensionN,0,localDimensionM);
  CRITTER_STOP(qr::cacqr::construct_Q);
  return ret;
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename ArgType, typename CommType>
matrix<typename ArgType::ScalarType,typename ArgType::DimensionType,rect> cacqr<SerializePolicy,IntermediatesPolicy>::construct_R(ArgType& args, CommType&& CommInfo){
  CRITTER_START(qr::cacqr::construct_R);
  auto localDimensionM = args.R.num_rows_local(); auto localDimensionN = args.R.num_columns_local();
  matrix<typename ArgType::ScalarType,typename ArgType::DimensionType,rect> ret(args.R.num_columns_global(),args.R.num_rows_global(),CommInfo.c, CommInfo.c);
  serialize<uppertri,uppertri>::invoke(args.R, ret,0,localDimensionN,0,localDimensionN,0,localDimensionN,0,localDimensionN);
  CRITTER_STOP(qr::cacqr::construct_R);
  return ret;
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void cacqr<SerializePolicy,IntermediatesPolicy>::apply_Q(MatrixType& src, ArgType& args,CommType&& CommInfo){
  CRITTER_START(qr::cacqr::apply_Q);
  using T = typename MatrixType::ScalarType;
  blas::ArgPack_gemm<T> gemmPack(blas::Order::AblasColumnMajor, blas::Transpose::AblasNoTrans, blas::Transpose::AblasNoTrans, 1., 0.);
  auto out = args.Q; matmult::summa::invoke(args.Q,src,out,gemmPack,std::forward<CommType>(CommInfo));
  CRITTER_STOP(qr::cacqr::apply_Q);
}

template<class SerializePolicy, class IntermediatesPolicy>
template<typename MatrixType, typename ArgType, typename CommType>
void cacqr<SerializePolicy,IntermediatesPolicy>::apply_QT(MatrixType& src, ArgType& args,CommType&& CommInfo) { assert(0) && "not implemented"; }

}
