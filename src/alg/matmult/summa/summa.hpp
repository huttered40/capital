/* Author: Edward Hutter */

namespace matmult{

// Invariant: it is assumed that the matrix data is stored in the _data member, and the _scratch member is available for exploiting
template<typename MatrixAType, typename MatrixBType, typename MatrixCType, typename CommType>
void summa::invoke(MatrixAType& A, MatrixBType& B, MatrixCType& C, CommType&& CommInfo,
                   blas::ArgPack_gemm<typename MatrixAType::ScalarType>& srcPackage){

  // Use tuples so we don't have to pass multiple things by reference.
  // Also this way, we can take advantage of the new pass-by-value move semantics that are efficient
  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using StructureA = typename MatrixAType::StructureType;
  using StructureB = typename MatrixBType::StructureType;

  bool isRootRow = ((CommInfo.x == CommInfo.z) ? true : false);
  bool isRootColumn = ((CommInfo.y == CommInfo.z) ? true : false);
  if (isRootRow){ A.swap(); }
  if (isRootColumn){ B.swap(); }
  U localDimensionM = (srcPackage.transposeA == blas::Transpose::AblasNoTrans ? A.num_rows_local() : A.num_columns_local());
  U localDimensionN = (srcPackage.transposeB == blas::Transpose::AblasNoTrans ? B.num_columns_local() : B.num_rows_local());
  U localDimensionK = (srcPackage.transposeA == blas::Transpose::AblasNoTrans ? A.num_columns_local() : A.num_rows_local());

  // Communicated data lives in the _scratch members of A,B
  distribute(A,B,std::forward<CommType>(CommInfo));

  // Assume, for now, that C has Rectangular Structure. In the future, we can always do the same procedure as above, and add a invoke after the AllReduce
  decltype(srcPackage.beta) save_beta = srcPackage.beta; srcPackage.beta = 0;
  blas::engine::_gemm(A.scratch(), B.scratch(), C.scratch(), localDimensionM, localDimensionN, localDimensionK,
                      (srcPackage.transposeA == blas::Transpose::AblasNoTrans ? localDimensionM : localDimensionK),
                      (srcPackage.transposeB == blas::Transpose::AblasNoTrans ? localDimensionK : localDimensionN), localDimensionM, srcPackage);
  collect(C,std::forward<CommType>(CommInfo));
  if (save_beta != 0){
    for (U i=0; i<C.num_elems(); i++){
      C.data()[i] = save_beta*C.data()[i] + C.scratch()[i];
    }
  }
  else{
    C.swap();
  }
  // Reset before returning
  srcPackage.beta = save_beta;
  if (!std::is_same<StructureA,rect>::value){ A.swap_pad(); }
  if (!std::is_same<StructureB,rect>::value){ B.swap_pad(); }
  if (isRootRow){ A.swap(); }
  if (isRootColumn){ B.swap(); }
}
  
template<typename T, typename U, typename ArgType, typename CommType>
T* summa::invoke(T* A, T* B, T* C, U localNumRowsA, U localNumColumnsA, U localNumColumnsB, U globalNumRowsA, U globalNumColumnsA, U globalNumColumnsB, ArgType&& args, CommType&& CommInfo){
  matrix<T,U,rect,cyclic> mA(A,localNumColumnsA,localNumRowsA,globalNumColumnsA,globalNumRowsA,CommInfo.d,CommInfo.d);
  matrix<T,U,rect,cyclic> mB(B,localNumColumnsB,localNumColumnsA,globalNumColumnsB,globalNumColumnsA,CommInfo.d,CommInfo.d);
  matrix<T,U,rect,cyclic> mC(C,localNumColumnsB,localNumRowsA,globalNumColumnsB,globalNumRowsA,CommInfo.d,CommInfo.d);
  invoke(mA,mB,mC,std::forward<ArgType>(args),std::forward<CommType>(CommInfo));
  return mC.get_data();
}

template<typename MatrixAType, typename MatrixBType, typename CommType>
void summa::invoke(MatrixAType& A, MatrixBType& B, CommType&& CommInfo,
                   blas::ArgPack_trmm<typename MatrixAType::ScalarType>& srcPackage){

  // Use tuples so we don't have to pass multiple things by reference.
  // Also this way, we can take advantage of the new pass-by-value move semantics that are efficient
  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using StructureA = typename MatrixAType::StructureType;
  using StructureB = typename MatrixBType::StructureType;

  bool isRootRow = ((CommInfo.x == CommInfo.z) ? true : false);
  bool isRootColumn = ((CommInfo.y == CommInfo.z) ? true : false);
  U localDimensionM = B.num_rows_local();
  U localDimensionN = B.num_columns_local();

  // Communicated data lives in the _scratch members of A,B
  if (srcPackage.side == blas::Side::AblasLeft){
    if (isRootRow){ A.swap(); }
    if (isRootColumn){ B.swap(); }
    distribute(A, B, std::forward<CommType>(CommInfo));
    blas::engine::_trmm(A.scratch(), B.scratch(), localDimensionM, localDimensionN, localDimensionM,
    (srcPackage.order == blas::Order::AblasColumnMajor ? localDimensionM : localDimensionN), srcPackage);
  }
  else{
    if (isRootRow){ B.swap(); }
    if (isRootColumn){ A.swap(); }
    distribute(B,A,std::forward<CommType>(CommInfo));
    blas::engine::_trmm(A.scratch(), B.scratch(), localDimensionM, localDimensionN, localDimensionN,
    (srcPackage.order == blas::Order::AblasColumnMajor ? localDimensionM : localDimensionN), srcPackage);
  }
  // We will follow the standard here: A is always the triangular matrix. B is always the rectangular matrix
  collect(B,std::forward<CommType>(CommInfo));
  // Reset before returning
  if (!std::is_same<StructureA,rect>::value){ A.swap_pad(); }
  if (!std::is_same<StructureB,rect>::value){ B.swap_pad(); }
  if (isRootRow && srcPackage.side == blas::Side::AblasLeft){ A.swap(); }
  B.swap();	// unconditional swap, since B holds output
}

template<typename MatrixAType, typename MatrixCType, typename CommType>
void summa::invoke(MatrixAType& A, MatrixCType& C, CommType&& CommInfo,
                   blas::ArgPack_syrk<typename MatrixAType::ScalarType>& srcPackage){
  // Note: Internally, this routine uses gemm, not syrk, as its not possible for each processor to perform local MM with symmetric matrices
  //         given the data layout over the processor grid.

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using StructureA = typename MatrixAType::StructureType;
  using StructureC = typename MatrixCType::StructureType;

  // No choice but to incur the copy cost below.
  MatrixAType B = A;
  util::transpose(B, std::forward<CommType>(CommInfo));

  bool isRootRow = ((CommInfo.x == CommInfo.z) ? true : false);
  bool isRootColumn = ((CommInfo.y == CommInfo.z) ? true : false);
  // Note: The routine will be C <- BA or AB, depending on the order in the srcPackage. B will always be the transposed matrix
  U localDimensionN = C.num_columns_local();  // rows or columns, doesn't matter. They should be the same. C is meant to be square
  U localDimensionK = (srcPackage.transposeA == blas::Transpose::AblasNoTrans ? A.num_columns_local() : A.num_rows_local());

  if (srcPackage.transposeA == blas::Transpose::AblasNoTrans){
    if (isRootRow){ A.swap(); }
    if (isRootColumn){ B.swap(); }
    distribute(A,B,std::forward<CommType>(CommInfo)); }
  else{
    if (isRootRow){ B.swap(); }
    if (isRootColumn){ A.swap(); }
    distribute(B,A,std::forward<CommType>(CommInfo)); }

  for (U i=0; i<C.num_elems(); i++) { C.scratch()[i] = 0.; }
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
  collect(C,std::forward<CommType>(CommInfo));

  if (srcPackage.beta != 0.){
    // Future optimization: Reduce loop length by half since the update will be a symmetric matrix and only half will be used going forward.
    for (U i=0; i<C.num_elems(); i++){
      C.data()[i] = srcPackage.beta*C.data()[i] + C.scratch()[i];
    }
  } else{ C.swap(); }
  // Reset before returning
  if (!std::is_same<StructureA,rect>::value) { A.swap_pad(); }
  if (isRootRow){ A.swap(); }
}

template<typename MatrixAType, typename MatrixBType, typename MatrixCType, typename CommType>
void summa::invoke(MatrixAType& A, MatrixBType& B, MatrixCType& C, typename MatrixAType::DimensionType AcutXstart,
                    typename MatrixAType::DimensionType AcutXend, typename MatrixAType::DimensionType AcutYstart,
                    typename MatrixAType::DimensionType AcutYend, typename MatrixBType::DimensionType BcutZstart,
                    typename MatrixBType::DimensionType BcutZend, typename MatrixBType::DimensionType BcutXstart,
                    typename MatrixBType::DimensionType BcutXend, typename MatrixCType::DimensionType CcutZstart,
                    typename MatrixCType::DimensionType CcutZend, typename MatrixCType::DimensionType CcutYstart,
                    typename MatrixCType::DimensionType CcutYend, CommType&& CommInfo,
                    blas::ArgPack_gemm<typename MatrixAType::ScalarType>& srcPackage, bool cutA, bool cutB, bool cutC){
  // We will set up 3 matrices and call the method above.

  using StructureC = typename MatrixCType::StructureType;

  // I cannot use a fast-pass-by-value via move constructor because I don't want to corrupt the true matrices A,B,C. Other reasons as well.
  MatrixAType matA = extract(A, AcutXstart, AcutXend, AcutYstart, AcutYend, CommInfo.d, cutA);
  MatrixBType matB = extract(B, BcutZstart, BcutZend, BcutXstart, BcutXend, CommInfo.d, cutB);
  MatrixCType matC = extract(C, CcutZstart, CcutZend, CcutYstart, CcutYend, CommInfo.d, cutC);

  invoke((cutA ? matA : A), (cutB ? matB : B), (cutC ? matC : C), std::forward<CommType>(CommInfo), srcPackage);

  // reverse serialize, to put the solved piece of C into where it should go.
  if (cutC){
    serialize<StructureC,StructureC>::invoke(C, matC, CcutZstart, CcutZend, CcutYstart, CcutYend, true);
  }
}


template<typename MatrixAType, typename MatrixBType, typename CommType>
void summa::invoke(MatrixAType& A, MatrixBType& B, typename MatrixAType::DimensionType AcutXstart,
                    typename MatrixAType::DimensionType AcutXend, typename MatrixAType::DimensionType AcutYstart,
                    typename MatrixAType::DimensionType AcutYend, typename MatrixBType::DimensionType BcutZstart,
                    typename MatrixBType::DimensionType BcutZend, typename MatrixBType::DimensionType BcutXstart,
                    typename MatrixBType::DimensionType BcutXend, CommType&& CommInfo,
                    blas::ArgPack_trmm<typename MatrixAType::ScalarType>& srcPackage, bool cutA, bool cutB){
  // We will set up 2 matrices and call the method above.

  using StructureB = typename MatrixBType::StructureType;

  // I cannot use a fast-pass-by-value via move constructor because I don't want to corrupt the true matrices A,B,C. Other reasons as well.
  MatrixAType matA = extract(A, AcutXstart, AcutXend, AcutYstart, AcutYend, CommInfo.d, cutA);
  MatrixBType matB = extract(B, BcutZstart, BcutZend, BcutXstart, BcutXend, CommInfo.d, cutB);
  invoke((cutA ? matA : A), (cutB ? matB : B), std::forward<CommType>(CommInfo), srcPackage);

  // reverse serialize, to put the solved piece of C into where it should go. Only if we need to
  if (cutB){
    serialize<StructureB,StructureB>::invoke(B, matB, BcutZstart, BcutZend, BcutXstart, BcutXend, true);
  }
}

template<typename MatrixAType, typename MatrixCType, typename CommType>
void summa::invoke(MatrixAType& A, MatrixCType& C, typename MatrixAType::DimensionType AcutXstart,
                   typename MatrixAType::DimensionType AcutXend, typename MatrixAType::DimensionType AcutYstart,
                   typename MatrixAType::DimensionType AcutYend, typename MatrixCType::DimensionType CcutZstart,
                   typename MatrixCType::DimensionType CcutZend, typename MatrixCType::DimensionType CcutXstart,
                   typename MatrixCType::DimensionType CcutXend, CommType&& CommInfo,
                   blas::ArgPack_syrk<typename MatrixAType::ScalarType>& srcPackage, bool cutA, bool cutC){
  // We will set up 2 matrices and call the method above.

  using StructureC = typename MatrixCType::StructureType;

  // I cannot use a fast-pass-by-value via move constructor because I don't want to corrupt the true matrices A,B,C. Other reasons as well.
  MatrixAType matA = extract(A, AcutXstart, AcutXend, AcutYstart, AcutYend, CommInfo.d, cutA);
  MatrixAType matC = extract(C, CcutZstart, CcutZend, CcutXstart, CcutXend, CommInfo.d, cutC);

  invoke((cutA ? matA : A), (cutC ? matC : C), std::forward<CommType>(CommInfo), srcPackage);

  // reverse serialize, to put the solved piece of C into where it should go.
  if (cutC){
    serialize<StructureC,StructureC>::invoke(C, matC, CcutZstart, CcutZend, CcutXstart, CcutXend, true);
  }
}

template<typename MatrixAType, typename MatrixBType, typename CommType>
void summa::distribute(MatrixAType& A, MatrixBType& B, CommType&& CommInfo){

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using StructureA = typename MatrixAType::StructureType;
  using StructureB = typename MatrixBType::StructureType;
  using Offload = typename MatrixAType::OffloadType;

  U localDimensionM = A.num_rows_local();
  U localDimensionN = B.num_columns_local();
  U localDimensionK = A.num_columns_local();
  U sizeA  = A.num_elems();
  U sizeB  = B.num_elems();
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
    std::vector<MPI_Request> row_req(CommInfo.num_chunks);
    std::vector<MPI_Request> column_req(CommInfo.num_chunks);
    std::vector<MPI_Status> row_stat(CommInfo.num_chunks);
    std::vector<MPI_Status> column_stat(CommInfo.num_chunks);
    int64_t offset = sizeA%CommInfo.num_chunks;
    int64_t progress=0;
    for (int64_t idx=0; idx < CommInfo.num_chunks; idx++){
      MPI_Ibcast(&A.scratch()[progress], idx==(CommInfo.num_chunks-1) ? sizeA/CommInfo.num_chunks+offset : sizeA/CommInfo.num_chunks,
                 mpi_type<T>::type, CommInfo.z, CommInfo.row, &row_req[idx]);
      progress += sizeA/CommInfo.num_chunks;
    }
    // initiate distribution along columns and complete distribution across rows
    offset = sizeB%CommInfo.num_chunks;
    progress=0;
    for (int64_t idx=0; idx < CommInfo.num_chunks; idx++){
     MPI_Ibcast(&B.scratch()[progress], idx==(CommInfo.num_chunks-1) ? sizeB/CommInfo.num_chunks+offset : sizeB/CommInfo.num_chunks,
                mpi_type<T>::type, CommInfo.z, CommInfo.column, &column_req[idx]);
      progress += sizeB/CommInfo.num_chunks;
      MPI_Wait(&row_req[idx],&row_stat[idx]);
    }
    // complete distribution along columns
    for (int64_t idx=0; idx < CommInfo.num_chunks; idx++){
      MPI_Wait(&column_req[idx],&column_stat[idx]);
    }
  }

  if (!std::is_same<StructureA,rect>::value){
    serialize<StructureA,rect>::invoke(A); A.swap_pad();
  }
  if (!std::is_same<StructureB,rect>::value){
    serialize<StructureB,rect>::invoke(B); B.swap_pad();
  }
}

template<typename MatrixType, typename CommType>
void summa::collect(MatrixType& matrix, CommType&& CommInfo){

  using T = typename MatrixType::ScalarType;
  using U = typename MatrixType::DimensionType;

  U numElems = matrix.num_elems();
  MPI_Allreduce(MPI_IN_PLACE,matrix.scratch(), numElems, mpi_type<T>::type, MPI_SUM, CommInfo.depth);
}

template<typename MatrixType>
MatrixType summa::extract(MatrixType& srcMatrix, typename MatrixType::DimensionType matrixArgColumnStart, typename MatrixType::DimensionType matrixArgColumnEnd,
                              typename MatrixType::DimensionType matrixArgRowStart, typename MatrixType::DimensionType matrixArgRowEnd,
		              int64_t sliceDim, bool getSub){

  using T = typename MatrixType::ScalarType;
  using U = typename MatrixType::DimensionType;
  using Structure = typename MatrixType::StructureType;

  if (getSub){
    U numColumns = matrixArgColumnEnd - matrixArgColumnStart;
    U numRows = matrixArgRowEnd - matrixArgRowStart;
    MatrixType fillMatrix(numColumns*sliceDim,numRows*sliceDim,sliceDim,sliceDim);
    serialize<Structure,Structure>::invoke(srcMatrix, fillMatrix, matrixArgColumnStart, matrixArgColumnEnd, matrixArgRowStart, matrixArgRowEnd);
    return fillMatrix;			// I am returning an rvalue
  }
  else{
    // return cheap garbage.
    return MatrixType(1,1,2,2);
  }
}
}
