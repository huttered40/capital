/* Author: Edward Hutter */

namespace inverse{

template<typename AlgType>
template<typename MatrixType, typename CommType>
typename MatrixType::ScalarType validate<AlgType>::invoke(MatrixType& matrixA, MatrixType& matrixB, CommType&& CommInfo){

  using T = typename MatrixType::ScalarType;
  using U = typename MatrixType::DimensionType;

  U localNumRows = matrixA.getNumColumnsLocal();
  U localNumColumns = matrixA.getNumColumnsLocal();
  U globalNumRows = matrixA.getNumColumnsGlobal();
  U globalNumColumns = matrixA.getNumColumnsGlobal();
  U numElems = localNumRows*localNumColumns;
  MatrixType matrixI(std::vector<T>(numElems,0), localNumColumns, localNumRows, globalNumColumns, globalNumRows, true);
  blas::ArgPack_gemm<T> blasArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasNoTrans, blas::Transpose::AblasTrans, 1., 0);
  matmult::summa::invoke(matrixA, matrixB, matrixI, std::forward<CommType>(CommInfo), blasArgs);
  auto Lambda = [](auto&& matrix, auto&& ref, size_t index, size_t sliceX, size_t sliceY){
    using T = typename std::remove_reference_t<decltype(matrix)>::ScalarType;
    T val,control;
    if (sliceX == sliceY){
      val = std::abs(1. - matrix.getRawData()[index]);
      control = 1;
    }
    else{
      val = matrix.getRawData()[index];
      control = 0;
    }
    return std::make_pair(val,control);
  };
  return util::residual_local(matrixI, matrixI, std::move(Lambda), CommInfo.slice, CommInfo.x, CommInfo.y, CommInfo.c, CommInfo.d);
}

}
