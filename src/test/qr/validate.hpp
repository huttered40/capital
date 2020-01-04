/* Author: Edward Hutter */
namespace qr{

template<typename AlgType>
template<typename MatrixType, typename ArgType, typename RectCommType>
typename MatrixType::ScalarType
validate<AlgType>::orthogonality(const MatrixType& A, ArgType& args, RectCommType&& RectTopo){

  using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType;
  auto SquareTopo = topo::square(RectTopo.cube,RectTopo.c);
  auto Q = AlgType::construct_Q(args,RectTopo); auto  Qtrans = Q;
  util::transpose(Qtrans, SquareTopo);
  U localNumRows = Qtrans.num_columns_local(); U localNumColumns = Q.num_columns_local();
  U globalNumRows = Qtrans.num_columns_global(); U globalNumColumns = Q.num_columns_global();
  U numElems = localNumRows*localNumColumns;
  MatrixType I(globalNumColumns, globalNumRows, SquareTopo.d, SquareTopo.d);

  blas::ArgPack_gemm<T> blasArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasTrans, blas::Transpose::AblasNoTrans, 1., 0.);
  matmult::summa::invoke(Qtrans, Q, I, SquareTopo, blasArgs);
  if (RectTopo.column_alt != MPI_COMM_WORLD){
    MPI_Allreduce(MPI_IN_PLACE, I.data(), I.num_elems(), mpi_type<T>::type, MPI_SUM, RectTopo.column_alt);
  }

  auto Lambda = [](auto&& matrix, auto&& ref, size_t index, size_t sliceX, size_t sliceY){
    using T = typename std::remove_reference_t<decltype(matrix)>::ScalarType;
    T val,control;
    if (sliceX == sliceY){ val = std::abs(1. - matrix.data()[index]); control = 1; }
    else{ val = matrix.data()[index]; control = 1.; }
    return std::make_pair(val,control);
  };
  return util::residual_local(I, I, std::move(Lambda), SquareTopo.slice, SquareTopo.x, SquareTopo.y, SquareTopo.c, SquareTopo.d);
}

template<typename AlgType>
template<typename MatrixType, typename ArgType, typename RectCommType>
typename MatrixType::ScalarType
validate<AlgType>::residual(const MatrixType& A, ArgType& args, RectCommType&& RectTopo){

  using T = typename MatrixType::ScalarType; using U = typename MatrixType::DimensionType;
  auto SquareTopo = topo::square(RectTopo.cube,RectTopo.c);
  auto R = AlgType::construct_R(args,RectTopo); auto Q = AlgType::construct_Q(args,RectTopo);
  util::remove_triangle(R, SquareTopo.x, SquareTopo.y, SquareTopo.d, 'U'); auto Asave = A;
  blas::ArgPack_gemm<T> blasArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasNoTrans, blas::Transpose::AblasNoTrans, 1., -1.);
  matmult::summa::invoke(Q,R,Asave, SquareTopo, blasArgs);

  auto Lambda = [](auto&& matrix, auto&& ref, size_t index, size_t sliceX, size_t sliceY){
    using T = typename std::remove_reference_t<decltype(matrix)>::ScalarType;
    T val = matrix.data()[index]; T control = ref.data()[index];
    return std::make_pair(val,control);
  };
  return util::residual_local(Asave,A, std::move(Lambda), SquareTopo.slice, SquareTopo.x, SquareTopo.y, SquareTopo.c, SquareTopo.d);
}
}
