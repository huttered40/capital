/* Author: Edward Hutter */

namespace cholesky{

template<typename AlgType>
template<typename MatrixType, typename TriMatrixType, typename ArgType, typename CommType>
typename MatrixType::ScalarType validate<AlgType>::invoke(const MatrixType& A, TriMatrixType&& Tri, ArgType&& args, CommType&& CommInfo){

  using T = typename MatrixType::ScalarType;
  using TriType = typename std::remove_reference<TriMatrixType>::type;
  static_assert(std::is_same<typename TriType::StructureType,rect>::value,"matrix instances must be of rect structure");
  static_assert(std::is_same<typename MatrixType::StructureType,rect>::value,"matrix instances must be of rect structure");

  PMPI_Barrier(MPI_COMM_WORLD);
  util::remove_triangle(Tri, CommInfo.x, CommInfo.y, CommInfo.d, args.dir);
  auto TriTrans = Tri;
  util::transpose(TriTrans, std::forward<CommType>(CommInfo));
  MatrixType Asave1 = A;
  MatrixType Asave2 = A;

  if (args.dir == 'L'){
    blas::ArgPack_gemm<T> blasArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasNoTrans, blas::Transpose::AblasTrans, 1., -1.);
    matmult::summa::invoke(Tri, TriTrans, Asave1, std::forward<CommType>(CommInfo), blasArgs);
    auto Lambda = [](auto&& matrix, auto&& ref, size_t index, size_t sliceX, size_t sliceY){
      using T = typename std::remove_reference_t<decltype(matrix)>::ScalarType;
      T val=0;
      T control=1.;
      if (sliceY >= sliceX){
        val = matrix.data()[index];
        control = ref.data()[index];
      }
      return std::make_pair(val,control);
    };
    return util::residual_local(Asave1, Asave2, std::move(Lambda), CommInfo.slice, CommInfo.x, CommInfo.y, CommInfo.d, CommInfo.d);
  }
  else if (args.dir == 'U'){
    blas::ArgPack_gemm<T> blasArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasTrans, blas::Transpose::AblasNoTrans, 1., -1.);
    PMPI_Barrier(MPI_COMM_WORLD);
    matmult::summa::invoke(TriTrans, Tri, Asave1, std::forward<CommType>(CommInfo), blasArgs);
    auto Lambda = [](auto&& matrix, auto&& ref, size_t index, size_t sliceX, size_t sliceY){
      using T = typename std::remove_reference_t<decltype(matrix)>::ScalarType;
      T val=0;
      T control=1.;
      if (sliceY <= sliceX){
        val = matrix.data()[index];
        control = ref.data()[index];
      }
      return std::make_pair(val,control);
    };
    return util::residual_local(Asave1, Asave2, std::move(Lambda), CommInfo.slice, CommInfo.x, CommInfo.y, CommInfo.d, CommInfo.d);
  }
  return 0.;	// prevent compiler complaints
}
}
