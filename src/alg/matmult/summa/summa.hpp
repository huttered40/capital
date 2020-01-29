/* Author: Edward Hutter */

namespace matmult{

// Invariant: it is assumed that the matrix data is stored in the _data member, and the _scratch member is available for exploiting
template<typename MatrixAType, typename MatrixBType, typename MatrixCType, typename CommType>
void summa::invoke(MatrixAType& A, MatrixBType& B, MatrixCType& C, CommType&& CommInfo, blas::ArgPack_gemm<typename MatrixAType::ScalarType>& srcPackage){
  TAU_START(matmult::summa::invoke);

  // Use tuples so we don't have to pass multiple things by reference.
  // Also this way, we can take advantage of the new pass-by-value move semantics that are efficient
  using T = typename MatrixAType::ScalarType;
  using StructureA = typename MatrixAType::StructureType; using StructureB = typename MatrixBType::StructureType;

  bool isRootRow = ((CommInfo.x == CommInfo.z) ? true : false);
  bool isRootColumn = ((CommInfo.y == CommInfo.z) ? true : false);
  if (isRootRow){ A.swap(); } if (isRootColumn){ B.swap(); }
  auto localDimensionM = (srcPackage.transposeA == blas::Transpose::AblasNoTrans ? A.num_rows_local() : A.num_columns_local());
  auto localDimensionN = (srcPackage.transposeB == blas::Transpose::AblasNoTrans ? B.num_columns_local() : B.num_rows_local());
  auto localDimensionK = (srcPackage.transposeA == blas::Transpose::AblasNoTrans ? A.num_columns_local() : A.num_rows_local());

  // Communicated data lives in the _scratch members of A,B
  distribute(A,B,std::forward<CommType>(CommInfo));

  // Assume, for now, that C has Rectangular Structure. In the future, we can always do the same procedure as above, and add a invoke after the AllReduce
  decltype(srcPackage.beta) save_beta = srcPackage.beta; srcPackage.beta = 0;
  blas::engine::_gemm(A.scratch(), B.scratch(), C.scratch(), localDimensionM, localDimensionN, localDimensionK,
                      (srcPackage.transposeA == blas::Transpose::AblasNoTrans ? localDimensionM : localDimensionK),
                      (srcPackage.transposeB == blas::Transpose::AblasNoTrans ? localDimensionK : localDimensionN), localDimensionM, srcPackage);
  collect(C,std::forward<CommType>(CommInfo));
  if (save_beta != 0){
    for (auto i=0; i<C.num_elems(); i++){ C.data()[i] = save_beta*C.data()[i] + C.scratch()[i]; }
  }
  else{ C.swap(); }
  // Reset before returning
  srcPackage.beta = save_beta;
  if (!std::is_same<StructureA,rect>::value){ A.swap_pad(); }
  if (!std::is_same<StructureB,rect>::value){ B.swap_pad(); }
  if (isRootRow){ A.swap(); } if (isRootColumn){ B.swap(); }
  TAU_STOP(matmult::summa::invoke);
}
  
template<typename MatrixAType, typename MatrixBType, typename CommType>
void summa::invoke(MatrixAType& A, MatrixBType& B, CommType&& CommInfo, blas::ArgPack_trmm<typename MatrixAType::ScalarType>& srcPackage){
  TAU_START(matmult::summa::invoke);

  // Use tuples so we don't have to pass multiple things by reference.
  // Also this way, we can take advantage of the new pass-by-value move semantics that are efficient
  using T = typename MatrixAType::ScalarType;
  using StructureA = typename MatrixAType::StructureType; using StructureB = typename MatrixBType::StructureType;

  bool isRootRow = ((CommInfo.x == CommInfo.z) ? true : false);
  bool isRootColumn = ((CommInfo.y == CommInfo.z) ? true : false);
  auto localDimensionM = B.num_rows_local(); auto localDimensionN = B.num_columns_local();

  // Communicated data lives in the _scratch members of A,B
  if (srcPackage.side == blas::Side::AblasLeft){
    if (isRootRow){ A.swap(); } if (isRootColumn){ B.swap(); }
    distribute(A, B, std::forward<CommType>(CommInfo));
    blas::engine::_trmm(A.scratch(), B.scratch(), localDimensionM, localDimensionN, localDimensionM, localDimensionM, srcPackage);
  }
  else{
    if (isRootRow){ B.swap(); } if (isRootColumn){ A.swap(); }
    distribute(B,A,std::forward<CommType>(CommInfo));
    if (std::is_same<StructureB,uppertri>::value){ B.swap_pad(); util::remove_triangle_local(B,CommInfo.x,CommInfo.y,CommInfo.d,'U'); B.swap_pad(); }
    if (std::is_same<StructureB,lowertri>::value){ B.swap_pad(); util::remove_triangle_local(B,CommInfo.x,CommInfo.y,CommInfo.d,'L'); B.swap_pad(); }
    blas::engine::_trmm(A.scratch(), B.scratch(), localDimensionM, localDimensionN, localDimensionN, localDimensionM, srcPackage);
  }
  // We will follow the standard here: A is always the triangular matrix. B is always the rectangular matrix
  if (!std::is_same<StructureB,rect>::value){ B.swap_pad(); serialize<StructureB,StructureB>::invoke(B,B,0,localDimensionN,0,localDimensionN,0,localDimensionN,0,localDimensionN,2,1); }
  collect(B,std::forward<CommType>(CommInfo));
  // Reset before returning
  if (!std::is_same<StructureA,rect>::value){ A.swap_pad(); }
  if (isRootRow && srcPackage.side == blas::Side::AblasLeft){ A.swap(); }
  B.swap();	// unconditional swap, since B holds output
  TAU_STOP(matmult::summa::invoke);
}

template<typename MatrixSrcType, typename MatrixDestType, typename CommType>
void summa::invoke(MatrixSrcType& A, MatrixDestType& C, CommType&& CommInfo, blas::ArgPack_syrk<typename MatrixSrcType::ScalarType>& srcPackage){
  TAU_START(matmult::summa::invoke);
  // No choice but to incur the copy cost below.
  MatrixSrcType B = A; util::transpose(B, std::forward<CommType>(CommInfo));
  syrk_internal(A,B,C,std::forward<CommType>(CommInfo),srcPackage);
  TAU_STOP(matmult::summa::invoke);
}

template<typename MatrixSrcType, typename MatrixDestType, typename CommType>
void summa::invoke(MatrixSrcType& A, MatrixSrcType& B, MatrixDestType& C, CommType&& CommInfo, blas::ArgPack_syrk<typename MatrixSrcType::ScalarType>& srcPackage){
  TAU_START(matmult::summa::invoke);
  util::transpose(B, std::forward<CommType>(CommInfo));
  syrk_internal(A,B,C,std::forward<CommType>(CommInfo),srcPackage);
  TAU_STOP(matmult::summa::invoke);
}

template<typename MatrixSrcType, typename MatrixDestType, typename CommType>
void summa::syrk_internal(MatrixSrcType& A, MatrixSrcType& B, MatrixDestType& C, CommType&& CommInfo, blas::ArgPack_syrk<typename MatrixSrcType::ScalarType>& srcPackage){
  // Note: Internally, this routine uses gemm, not syrk, as its not possible for each processor to perform local MM with symmetric matrices
  //         given the data layout over the processor grid.

  using T = typename MatrixSrcType::ScalarType;
  using StructureA = typename MatrixSrcType::StructureType; using StructureC = typename MatrixDestType::StructureType;

  bool isRootRow = ((CommInfo.x == CommInfo.z) ? true : false);
  bool isRootColumn = ((CommInfo.y == CommInfo.z) ? true : false);
  // Note: The routine will be C <- BA or AB, depending on the order in the srcPackage. B will always be the transposed matrix
  auto localDimensionN = C.num_columns_local();  // rows or columns, doesn't matter. They should be the same. C is meant to be square
  auto localDimensionK = (srcPackage.transposeA == blas::Transpose::AblasNoTrans ? A.num_columns_local() : A.num_rows_local());

  if (srcPackage.transposeA == blas::Transpose::AblasNoTrans){
    if (isRootRow){ A.swap(); } if (isRootColumn){ B.swap(); }
    distribute(A,B,std::forward<CommType>(CommInfo)); }
  else{
    if (isRootRow){ B.swap(); } if (isRootColumn){ A.swap(); }
    distribute(B,A,std::forward<CommType>(CommInfo)); }

  if (!std::is_same<StructureC,rect>::value) { C.swap_pad(); }
  for (auto i=0; i<(localDimensionN*localDimensionN); i++) { C.scratch()[i] = 0.; }
  // This cancels out any affect beta could have. Beta is just not compatable with summa and must be handled separately
  if (srcPackage.transposeA == blas::Transpose::AblasNoTrans){
    blas::ArgPack_gemm<T> gemmArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasNoTrans, blas::Transpose::AblasTrans, srcPackage.alpha,srcPackage.beta);
    blas::engine::_gemm(A.scratch(), B.scratch(), C.scratch(), localDimensionN, localDimensionN, localDimensionK,
                        localDimensionN, localDimensionN, localDimensionN, gemmArgs);
  }
  else{
    blas::ArgPack_gemm<T> gemmArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasTrans, blas::Transpose::AblasNoTrans,srcPackage.alpha,srcPackage.beta);
    blas::engine::_gemm(B.scratch(), A.scratch(), C.scratch(), localDimensionN, localDimensionN, localDimensionK,
                        localDimensionK, localDimensionK, localDimensionN, gemmArgs);
  }
  if (std::is_same<StructureC,uppertri>::value) { C.swap_pad(); auto counter=0; for (auto i=0; i<localDimensionN; i++) { for (auto j=0; j<(i+1); j++) C.scratch()[counter++] = C.pad()[i*localDimensionN+j]; } }
  if (std::is_same<StructureC,lowertri>::value) { C.swap_pad(); auto counter=0; for (auto i=0; i<localDimensionN; i++) { for (auto j=0; j<(localDimensionN-i); j++) C.scratch()[counter++] = C.pad()[i*localDimensionN+j]; } }
  collect(C,std::forward<CommType>(CommInfo));

  if (srcPackage.beta != 0.){
    // Future optimization: Reduce loop length by half since the update will be a symmetric matrix and only half will be used going forward.
    for (auto i=0; i<C.num_elems(); i++){ C.data()[i] = srcPackage.beta*C.data()[i] + C.scratch()[i]; }
  } else{ C.swap(); }
  // Reset before returning
  if (!std::is_same<StructureA,rect>::value) { A.swap_pad(); }
  if (isRootRow){ A.swap(); }
}

template<typename MatrixAType, typename MatrixBType, typename CommType>
void summa::distribute(MatrixAType& A, MatrixBType& B, CommType&& CommInfo){

  using T = typename MatrixAType::ScalarType;
  using StructureA = typename MatrixAType::StructureType; using StructureB = typename MatrixBType::StructureType;

  auto localDimensionM = A.num_rows_local(); auto localDimensionN = B.num_columns_local();
  auto localDimensionK = A.num_columns_local(); auto sizeA  = A.num_elems(); auto sizeB  = B.num_elems();
  bool isRootRow = ((CommInfo.x == CommInfo.z) ? true : false);
  bool isRootColumn = ((CommInfo.y == CommInfo.z) ? true : false);

  // Check chunk size. If its 0, then bcast across rows and columns with no overlap
  if (CommInfo.num_chunks == 0){
    // distribute across rows
    MPI_Bcast(A.scratch(), sizeA, mpi_type<T>::type, CommInfo.z, CommInfo.row);
    // distribute across columns
    MPI_Bcast(B.scratch(), sizeB, mpi_type<T>::type, CommInfo.z, CommInfo.column);
  }
  else{
    // initiate distribution across rows
    std::vector<MPI_Request> row_req(CommInfo.num_chunks); std::vector<MPI_Request> column_req(CommInfo.num_chunks);
    std::vector<MPI_Status> row_stat(CommInfo.num_chunks); std::vector<MPI_Status> column_stat(CommInfo.num_chunks);
    int64_t offset = sizeA%CommInfo.num_chunks; int64_t progress=0;
    for (int64_t idx=0; idx < CommInfo.num_chunks; idx++){
      MPI_Ibcast(&A.scratch()[progress], idx==(CommInfo.num_chunks-1) ? sizeA/CommInfo.num_chunks+offset : sizeA/CommInfo.num_chunks,
                 mpi_type<T>::type, CommInfo.z, CommInfo.row, &row_req[idx]);
      progress += sizeA/CommInfo.num_chunks;
    }
    // initiate distribution along columns and complete distribution across rows
    offset = sizeB%CommInfo.num_chunks; progress=0;
    for (int64_t idx=0; idx < CommInfo.num_chunks; idx++){
     MPI_Ibcast(&B.scratch()[progress], idx==(CommInfo.num_chunks-1) ? sizeB/CommInfo.num_chunks+offset : sizeB/CommInfo.num_chunks,
                mpi_type<T>::type, CommInfo.z, CommInfo.column, &column_req[idx]);
      progress += sizeB/CommInfo.num_chunks;
      MPI_Wait(&row_req[idx],&row_stat[idx]);
    }
    // complete distribution along columns
    for (int64_t idx=0; idx < CommInfo.num_chunks; idx++){ MPI_Wait(&column_req[idx],&column_stat[idx]); }
  }
  if (!std::is_same<StructureA,rect>::value){ serialize<StructureA,StructureA>::invoke(A,A,0,localDimensionK,0,localDimensionM,0,localDimensionK,0,localDimensionM,1,2); A.swap_pad(); }
  if (!std::is_same<StructureB,rect>::value){ serialize<StructureB,StructureB>::invoke(B,B,0,localDimensionN,0,localDimensionN,0,localDimensionN,0,localDimensionN,1,2); B.swap_pad(); }
}

template<typename MatrixType, typename CommType>
void summa::collect(MatrixType& matrix, CommType&& CommInfo){

  using T = typename MatrixType::ScalarType;
  if (CommInfo.num_chunks == 0){
    MPI_Allreduce(MPI_IN_PLACE,matrix.scratch(), matrix.num_elems(), mpi_type<T>::type, MPI_SUM, CommInfo.depth);
  }
  else{
    // initiate collection along depth
    std::vector<MPI_Request> req(CommInfo.num_chunks); std::vector<MPI_Status> stat(CommInfo.num_chunks);
    int64_t offset = matrix.num_elems()%CommInfo.num_chunks; int64_t progress=0;
    for (int64_t idx=0; idx < CommInfo.num_chunks; idx++){
      MPI_Iallreduce(MPI_IN_PLACE, &matrix.scratch()[progress], idx==(CommInfo.num_chunks-1) ? matrix.num_elems()/CommInfo.num_chunks+offset : matrix.num_elems()/CommInfo.num_chunks,
                     mpi_type<T>::type, MPI_SUM, CommInfo.depth, &req[idx]);
      progress += matrix.num_elems()/CommInfo.num_chunks;
    }
    // complete
    for (int64_t idx=0; idx < CommInfo.num_chunks; idx++){ MPI_Wait(&req[idx],&stat[idx]); }
  }
}
}
