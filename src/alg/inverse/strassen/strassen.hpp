/* Author: Edward Hutter */

namespace inverse{

template<typename MatrixType, typename CommType>
void strassen::invoke(MatrixType& matrix, CommType&& CommInfo){
  // `matrix` is modified in-place
  TAU_FSTART(strassen::invoke);
  // TODO: Assuming CommType is an instance of SquareTopo
  using T = typename MatrixType::ScalarType;
  using U = typename MatrixType::DimensionType;
  assert(matrix.getNumColumnsGlobal() == matrix.getNumRowsGlobal());
  U globalDimensionN = matrix.getNumColumnsGlobal();
  U localDimensionN = matrix.getNumColumnsLocal();

  // obtain starting iterate by getting infinity norm
  auto ptr_data = matrix.getMatrixData();
  std::vector<double> save_partial_row_sums(localDimensionN,0.);
  auto norm=0.;
  for (auto col=0; col<localDimensionN; col++){
    for (auto row=0; row<localDimensionN; row++){
      save_partial_row_sums[row]+=ptr_data[row][col];
    }
  }
  MPI_Allreduce(MPI_IN_PLACE,&save_partial_row_sums[0],localDimensionN,mpi_type<T>::type, MPI_SUM, CommInfo.row);// sum along rows to complete row-sum. Could be reduce instead of allreduce
  T max_row_sum = 0;
  for (auto i=0; i<save_partial_row_sums.size(); i++){
    max_row_sum = std::max(max_row_sum,save_partial_row_sums[i]);
  }
  MPI_Allreduce(MPI_IN_PLACE,&max_row_sum,1,mpi_type<T>::type, MPI_MAX, CommInfo.slice);// max across slice in sub-communicator. Could be reduce instead of allreduce
  MatrixType iterate(globalDimensionN,globalDimensionN, CommInfo.c, CommInfo.d);
  MatrixType intermediate(globalDimensionN,globalDimensionN, CommInfo.c, CommInfo.d);
  iterate.DistributeIdentity(CommInfo.x, CommInfo.y, CommInfo.c, CommInfo.d,1./max_row_sum);
  std::cout << "max_row_sum - " << max_row_sum << std::endl;
  // Each process now knows the starting iterate
  // But, we need to make sure that each process knows which of its elements are global diagonal elements
  
  blas::ArgPack_gemm<T> pack1(blas::Order::AblasColumnMajor, blas::Transpose::AblasNoTrans, blas::Transpose::AblasNoTrans, 1., 0.);
  blas::ArgPack_gemm<T> pack2(blas::Order::AblasColumnMajor, blas::Transpose::AblasNoTrans, blas::Transpose::AblasNoTrans, -1., 2.);
  int i=0;
  while (i<100){
    matmult::summa::invoke(iterate, matrix, intermediate, CommInfo, pack1);
    // TODO: check stopping criterion here ..
    matmult::summa::invoke(intermediate, iterate, iterate, CommInfo, pack2);	// TODO: Make sure having `iterate` as both in/out is ok
    i++;
  }
  std::memcpy(matrix.getRawData(),iterate.getRawData(),sizeof(T)*matrix.getNumElems()); 

  TAU_FSTOP(strassen::invoke);
}
}
