/* Author: Edward Hutter */
namespace qr{

template<typename AlgType>
template<typename MatrixType, typename RectCommType, typename SquareCommType>
typename MatrixType::ScalarType
validate<AlgType>::orth(MatrixType& Q, RectCommType&& RectCommInfo, SquareCommType&& SquareCommInfo){

  using T = typename MatrixType::ScalarType;
  using U = typename MatrixType::DimensionType;

  MatrixType Qtrans = Q;
  util::transpose(Qtrans, SquareCommInfo);
  U localNumRows = Qtrans.num_columns_local();
  U localNumColumns = Q.num_columns_local();
  U globalNumRows = Qtrans.num_columns_global();
  U globalNumColumns = Q.num_columns_global();
  U numElems = localNumRows*localNumColumns;
  MatrixType I(globalNumColumns, globalNumRows, SquareCommInfo.d, SquareCommInfo.d);

  blas::ArgPack_gemm<T> blasArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasTrans, blas::Transpose::AblasNoTrans, 1., 0.);
  matmult::summa::invoke(Qtrans, Q, I, std::forward<SquareCommType>(SquareCommInfo), blasArgs);
  if (RectCommInfo.column_alt != MPI_COMM_WORLD){
    MPI_Allreduce(MPI_IN_PLACE, I.data(), I.num_elems(), mpi_type<T>::type, MPI_SUM, RectCommInfo.column_alt);
  }

  auto Lambda = [](auto&& matrix, auto&& ref, size_t index, size_t sliceX, size_t sliceY){
    using T = typename std::remove_reference_t<decltype(matrix)>::ScalarType;
    T val,control;
    if (sliceX == sliceY){
      val = std::abs(1. - matrix.data()[index]);
      control = 1;
    }
    else{
      val = matrix.data()[index];
      control = 0;
    }
    return std::make_pair(val,control);
  };
  return util::residual_local(I, I, std::move(Lambda), SquareCommInfo.slice, SquareCommInfo.x, SquareCommInfo.y, SquareCommInfo.c, SquareCommInfo.d);
}

template<typename AlgType>
template<typename MatrixQType, typename MatrixRType, typename MatrixAType, typename RectCommType, typename SquareCommType>
typename MatrixAType::ScalarType
validate<AlgType>::residual(MatrixQType& Q, MatrixRType& R, MatrixAType& A, RectCommType&& RectCommInfo, SquareCommType&& SquareCommInfo){

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  MatrixAType saveMatA = A;
  blas::ArgPack_gemm<T> blasArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasNoTrans, blas::Transpose::AblasNoTrans, 1., -1.);
  matmult::summa::invoke(Q, R, saveMatA, std::forward<SquareCommType>(SquareCommInfo), blasArgs);

  auto Lambda = [](auto&& matrix, auto&& ref, size_t index, size_t sliceX, size_t sliceY){
    using T = typename std::remove_reference_t<decltype(matrix)>::ScalarType;
    T val = matrix.data()[index];
    T control = ref.data()[index];
    return std::make_pair(val,control);
  };
  return util::residual_local(saveMatA, A, std::move(Lambda), SquareCommInfo.slice, SquareCommInfo.x, SquareCommInfo.y, SquareCommInfo.c, SquareCommInfo.d);
}

/* Validation against sequential BLAS/LAPACK constructs */
template<typename AlgType>
template<typename MatrixAType, typename MatrixQType, typename MatrixRType, typename CommType>
std::pair<typename MatrixAType::ScalarType,typename MatrixAType::ScalarType>
validate<AlgType>::invoke(MatrixAType& A, MatrixQType& Q, MatrixRType& R, CommType&& CommInfo){
  using T = typename MatrixAType::ScalarType;

  auto SquareTopo = topo::square(CommInfo.cube,CommInfo.c);
  util::remove_triangle(R, SquareTopo.x, SquareTopo.y, SquareTopo.d, 'U');
  T error1 = residual(Q, R, A, std::forward<CommType>(CommInfo), SquareTopo);
  T error2 = orth(Q, std::forward<CommType>(CommInfo), SquareTopo);
  return std::make_pair(error1,error2);
}
}
