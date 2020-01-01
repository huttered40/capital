/* Author: Edward Hutter */

namespace cholesky{

template<typename AlgType>
template<typename MatrixAType, typename MatrixTriType, typename CommType>
typename MatrixAType::ScalarType validate<AlgType>::invoke(MatrixAType& A, MatrixTriType& Tri, char dir, CommType&& CommInfo){

  using T = typename MatrixAType::ScalarType;

  PMPI_Barrier(MPI_COMM_WORLD);
  util::remove_triangle(Tri, CommInfo.x, CommInfo.y, CommInfo.d, dir);
  MatrixTriType TriTrans = Tri;
  util::transpose(TriTrans, std::forward<CommType>(CommInfo));
  MatrixAType saveMatA = A;

  if (dir == 'L'){
    blas::ArgPack_gemm<T> blasArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasNoTrans, blas::Transpose::AblasTrans, 1., -1.);
    matmult::summa::invoke(Tri, TriTrans, A, std::forward<CommType>(CommInfo), blasArgs);
    auto Lambda = [](auto&& matrix, auto&& ref, size_t index, size_t sliceX, size_t sliceY){
      using T = typename std::remove_reference_t<decltype(matrix)>::ScalarType;
      T val=0;
      T control=0;
      if (sliceY >= sliceX){
        val = matrix.data()[index];
        control = ref.data()[index];
      }
      return std::make_pair(val,control);
    };
    return util::residual_local(A, saveMatA, std::move(Lambda), CommInfo.slice, CommInfo.x, CommInfo.y, CommInfo.d, CommInfo.d);
  }
  else if (dir == 'U'){
    blas::ArgPack_gemm<T> blasArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasTrans, blas::Transpose::AblasNoTrans, 1., -1.);
    PMPI_Barrier(MPI_COMM_WORLD);
    matmult::summa::invoke(TriTrans, Tri, A, std::forward<CommType>(CommInfo), blasArgs);
    auto Lambda = [](auto&& matrix, auto&& ref, size_t index, size_t sliceX, size_t sliceY){
      using T = typename std::remove_reference_t<decltype(matrix)>::ScalarType;
      T val=0;
      T control=0;
      if (sliceY <= sliceX){
        val = matrix.data()[index];
        control = ref.data()[index];
      }
      return std::make_pair(val,control);
    };
    return util::residual_local(A, saveMatA, std::move(Lambda), CommInfo.slice, CommInfo.x, CommInfo.y, CommInfo.d, CommInfo.d);
  }
  return 0.;	// prevent compiler complaints
}
}
