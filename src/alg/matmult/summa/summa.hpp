/* Author: Edward Hutter */

namespace matmult{
template<typename MatrixBType, typename CommType>
void summa::invoke(typename MatrixBType::ScalarType* matrixA, MatrixBType& matrixB, typename MatrixBType::ScalarType* matrixC,
                    typename MatrixBType::DimensionType matrixAnumColumns, typename MatrixBType::DimensionType matrixAnumRows,
                    typename MatrixBType::DimensionType matrixBnumColumns, typename MatrixBType::DimensionType matrixBnumRows,
                    typename MatrixBType::DimensionType matrixCnumColumns, typename MatrixBType::DimensionType matrixCnumRows,
                    CommType&& CommInfo, const blas::ArgPack_gemm<typename MatrixBType::ScalarType>& srcPackage){

  // Note: this is a temporary method that simplifies optimizations by bypassing the Matrix interface
  //       Later on, I can make this prettier and merge with the Matrix-explicit method below.
  //       Also, I only allow method1, not Allgather-based method2
  using T = typename MatrixBType::ScalarType;
  using U = typename MatrixBType::DimensionType;
  using StructureB = typename MatrixBType::StructureType;
  using Distribution = typename MatrixBType::DistributionType;
  using Offload = typename MatrixBType::OffloadType;

  T* matrixAEnginePtr;
  T* matrixBEnginePtr;
  T* foreignA;
  std::vector<T> foreignB;
  U localDimensionM = (srcPackage.transposeA == blas::Transpose::AblasNoTrans ? matrixAnumRows : matrixAnumColumns);
  U localDimensionN = (srcPackage.transposeB == blas::Transpose::AblasNoTrans ? matrixBnumColumns : matrixBnumRows);
  U localDimensionK = (srcPackage.transposeA == blas::Transpose::AblasNoTrans ? matrixAnumColumns : matrixAnumRows);

  U sizeA = matrixAnumRows*matrixAnumColumns;
  U sizeB = matrixB.getNumElems();
  U sizeC = matrixCnumRows*matrixCnumColumns;
  bool isRootRow = ((CommInfo.x == CommInfo.z) ? true : false);
  bool isRootColumn = ((CommInfo.y == CommInfo.z) ? true : false);

  //BroadcastPanels((isRootRow ? matrixA : foreignA), sizeA, isRootRow, CommInfo.z, CommInfo.row);
  matrixAEnginePtr = (isRootRow ? matrixA : foreignA);
  //BroadcastPanels((isRootColumn ? matrixB.getVectorData() : foreignB), sizeB, isRootColumn, CommInfo.z, CommInfo.column);
  if ((!std::is_same<StructureB,rect>::value) && (!std::is_same<StructureB,square>::value)){
    matrix<T,U,rect,Distribution,Offload> helperB(std::vector<T>(), matrixBnumColumns, matrixBnumRows, matrixBnumColumns, matrixBnumRows);
    getEnginePtr(matrixB, helperB, (isRootColumn ? matrixB.getVectorData() : foreignB), isRootColumn);
    matrixBEnginePtr = helperB.getRawData();
  }
  else{
    matrixBEnginePtr = (isRootColumn ? matrixB.getRawData() : &foreignB[0]);
  }

  // Massive bug fix. Need to use a separate array if beta != 0

  T* matrixCforEnginePtr = matrixC;
  if (srcPackage.beta == 0){
    blas::engine::_gemm(matrixAEnginePtr, matrixBEnginePtr, matrixCforEnginePtr, localDimensionM, localDimensionN, localDimensionK,
      (srcPackage.transposeA == blas::Transpose::AblasNoTrans ? localDimensionM : localDimensionK),
      (srcPackage.transposeB == blas::Transpose::AblasNoTrans ? localDimensionK : localDimensionN),
      localDimensionM, srcPackage);
    MPI_Allreduce(MPI_IN_PLACE,matrixCforEnginePtr, sizeC, mpi_type<T>::type, MPI_SUM, CommInfo.depth);
  }
  else{
    // This cancels out any affect beta could have. Beta is just not compatable with summa and must be handled separately
     std::vector<T> holdProduct(sizeC,0);
     blas::engine::_gemm(matrixAEnginePtr, matrixBEnginePtr, &holdProduct[0], localDimensionM, localDimensionN, localDimensionK,
       (srcPackage.transposeA == blas::Transpose::AblasNoTrans ? localDimensionM : localDimensionK),
       (srcPackage.transposeB == blas::Transpose::AblasNoTrans ? localDimensionK : localDimensionN),
       localDimensionM, srcPackage); 
    MPI_Allreduce(MPI_IN_PLACE, &holdProduct[0], sizeC, mpi_type<T>::type, MPI_SUM, CommInfo.depth);
    for (U i=0; i<sizeC; i++){
      matrixC[i] = srcPackage.beta*matrixC[i] + holdProduct[i];
    }
  }
  if (!isRootRow) delete[] foreignA;
}


// This algorithm with underlying gemm BLAS routine will allow any Matrix Structure.
//   Of course we will serialize into Square Structure if not in Square Structure already in order to be compatible
//   with BLAS-3 routines.

template<typename MatrixAType, typename MatrixBType, typename MatrixCType, typename CommType>
void summa::invoke(MatrixAType& matrixA, MatrixBType& matrixB, MatrixCType& matrixC, CommType&& CommInfo,
                     const blas::ArgPack_gemm<typename MatrixAType::ScalarType>& srcPackage, size_t methodKey){

  // Use tuples so we don't have to pass multiple things by reference.
  // Also this way, we can take advantage of the new pass-by-value move semantics that are efficient
  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  T* matrixAEnginePtr;
  T* matrixBEnginePtr;
  std::vector<T> matrixAEngineVector;
  std::vector<T> matrixBEngineVector;
  std::vector<T> foreignA;
  std::vector<T> foreignB;
  bool serializeKeyA = false;
  bool serializeKeyB = false;
  U localDimensionM = (srcPackage.transposeA == blas::Transpose::AblasNoTrans ? matrixA.getNumRowsLocal() : matrixA.getNumColumnsLocal());
  U localDimensionN = (srcPackage.transposeB == blas::Transpose::AblasNoTrans ? matrixB.getNumColumnsLocal() : matrixB.getNumRowsLocal());
  U localDimensionK = (srcPackage.transposeA == blas::Transpose::AblasNoTrans ? matrixA.getNumColumnsLocal() : matrixA.getNumRowsLocal());

  if (methodKey == 0){
    distribute_bcast(matrixA,matrixB,std::forward<CommType>(CommInfo),matrixAEnginePtr,matrixBEnginePtr,matrixAEngineVector,matrixBEngineVector,foreignA,foreignB,serializeKeyA,serializeKeyB);
  }
  else if (methodKey == 1){
    serializeKeyA = true;
    serializeKeyB = true;
    distribute_allgather(matrixA,matrixB,std::forward<CommType>(CommInfo), matrixAEngineVector,matrixBEngineVector,serializeKeyA,serializeKeyB);
  }

  // Assume, for now, that matrixC has Rectangular Structure. In the future, we can always do the same procedure as above, and add a invoke after the AllReduce

  // Massive bug fix. Need to use a separate array if beta != 0

  T* matrixCforEnginePtr = matrixC.getRawData();
  if (srcPackage.beta == 0){
    blas::engine::_gemm((serializeKeyA ? &matrixAEngineVector[0] : matrixAEnginePtr), (serializeKeyB ? &matrixBEngineVector[0] : matrixBEnginePtr),
      matrixCforEnginePtr, localDimensionM, localDimensionN, localDimensionK,
      (srcPackage.transposeA == blas::Transpose::AblasNoTrans ? localDimensionM : localDimensionK),
      (srcPackage.transposeB == blas::Transpose::AblasNoTrans ? localDimensionK : localDimensionN),
      localDimensionM, srcPackage);
    collect(matrixCforEnginePtr,matrixC,std::forward<CommType>(CommInfo));
   }
   else{
     // This cancels out any affect beta could have. Beta is just not compatable with summa and must be handled separately
     std::vector<T> holdProduct(matrixC.getNumElems(),0);
     blas::engine::_gemm((serializeKeyA ? &matrixAEngineVector[0] : matrixAEnginePtr), (serializeKeyB ? &matrixBEngineVector[0] : matrixBEnginePtr),
       &holdProduct[0], localDimensionM, localDimensionN, localDimensionK,
       (srcPackage.transposeA == blas::Transpose::AblasNoTrans ? localDimensionM : localDimensionK),
       (srcPackage.transposeB == blas::Transpose::AblasNoTrans ? localDimensionK : localDimensionN),
       localDimensionM, srcPackage); 
    collect(&holdProduct[0],matrixC,std::forward<CommType>(CommInfo),1);
    for (U i=0; i<holdProduct.size(); i++){
      matrixC.getRawData()[i] = srcPackage.beta*matrixC.getRawData()[i] + holdProduct[i];
    }
  }
}

template<typename MatrixAType, typename MatrixBType, typename CommType>
void summa::invoke(MatrixAType& matrixA, MatrixBType& matrixB, CommType&& CommInfo,
                     const blas::ArgPack_trmm<typename MatrixAType::ScalarType>& srcPackage, size_t methodKey){

  // Use tuples so we don't have to pass multiple things by reference.
  // Also this way, we can take advantage of the new pass-by-value move semantics that are efficient
  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  T* matrixAEnginePtr;
  T* matrixBEnginePtr;
  std::vector<T> matrixAEngineVector;
  std::vector<T> matrixBEngineVector;
  std::vector<T> foreignA;
  std::vector<T> foreignB;
  bool serializeKeyA = false;
  bool serializeKeyB = false;
  U localDimensionM = matrixB.getNumRowsLocal();
  U localDimensionN = matrixB.getNumColumnsLocal();

  // soon, we will need a methodKey for the different MM algs
  if (srcPackage.side == blas::Side::AblasLeft){
    if (methodKey == 0){
      distribute_bcast(matrixA, matrixB, std::forward<CommType>(CommInfo), matrixAEnginePtr, matrixBEnginePtr, matrixAEngineVector, matrixBEngineVector, foreignA, foreignB,
        serializeKeyA, serializeKeyB);
    }
    else if (methodKey == 1){
      serializeKeyA = true;
      serializeKeyB = true;
      distribute_allgather(matrixA, matrixB, std::forward<CommType>(CommInfo), matrixAEngineVector, matrixBEngineVector,
        serializeKeyA, serializeKeyB);
    }
    blas::engine::_trmm((serializeKeyA ? &matrixAEngineVector[0] : matrixAEnginePtr), (serializeKeyB ? &matrixBEngineVector[0] : matrixBEnginePtr),
      localDimensionM, localDimensionN, localDimensionM, (srcPackage.order == blas::Order::AblasColumnMajor ? localDimensionM : localDimensionN),
      srcPackage);
  }
  else{
    if (methodKey == 0){
      distribute_bcast(matrixB, matrixA, std::forward<CommType>(CommInfo), matrixBEnginePtr, matrixAEnginePtr, matrixBEngineVector, matrixAEngineVector, foreignB, foreignA, serializeKeyB, serializeKeyA);
    }
    else if (methodKey == 1){
      serializeKeyA = true;
      serializeKeyB = true;
      distribute_allgather(matrixB, matrixA, std::forward<CommType>(CommInfo), matrixBEngineVector, matrixAEngineVector, serializeKeyB, serializeKeyA);
    }
    blas::engine::_trmm((serializeKeyA ? &matrixAEngineVector[0] : matrixAEnginePtr), (serializeKeyB ? &matrixBEngineVector[0] : matrixBEnginePtr),
      localDimensionM, localDimensionN, localDimensionN, (srcPackage.order == blas::Order::AblasColumnMajor ? localDimensionM : localDimensionN),
      srcPackage);
  }
  // We will follow the standard here: matrixA is always the triangular matrix. matrixB is always the rectangular matrix
  collect((serializeKeyB ? &matrixBEngineVector[0] : matrixBEnginePtr),matrixB,std::forward<CommType>(CommInfo));
}

template<typename MatrixAType, typename CommType>
void summa::invoke(MatrixAType& matrixA, typename MatrixAType::ScalarType* matrixB, typename MatrixAType::DimensionType matrixAnumColumns,
                    typename MatrixAType::DimensionType matrixAnumRows, typename MatrixAType::DimensionType matrixBnumColumns,
                    typename MatrixAType::DimensionType matrixBnumRows, CommType&& CommInfo,
                    const blas::ArgPack_trmm<typename MatrixAType::ScalarType>& srcPackage){
  // Note: this is a temporary method that simplifies optimizations by bypassing the Matrix interface
  //       Later on, I can make this prettier and merge with the Matrix-explicit method below.
  //       Also, I only allow method1, not Allgather-based method2

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using StructureA = typename MatrixAType::StructureType;
  using Distribution = typename MatrixAType::DistributionType;
  using Offload = typename MatrixAType::OffloadType;

  std::vector<T> matrixAEnginePtr;
  T* matrixBEnginePtr;
  std::vector<T> foreignA;
  T* foreignB;
  U localDimensionM = matrixBnumRows;
  U localDimensionN = matrixBnumColumns;

  U sizeA = matrixA.getNumElems();
  U sizeB = matrixBnumRows*matrixBnumColumns;
  bool isRootRow = ((CommInfo.x == CommInfo.z) ? true : false);
  bool isRootColumn = ((CommInfo.y == CommInfo.z) ? true : false);

  // soon, we will need a methodKey for the different MM algs
  if (srcPackage.side == blas::Side::AblasLeft){
    //BroadcastPanels((isRootRow ? matrixA.getVectorData() : foreignA), sizeA, isRootRow, CommInfo.z, CommInfo.row);
    if ((!std::is_same<StructureA,rect>::value) && (!std::is_same<StructureA,square>::value)){
      matrix<T,U,rect,Distribution,Offload> helperA(std::vector<T>(), matrixAnumColumns, matrixAnumRows, matrixAnumColumns, matrixAnumRows);
      getEnginePtr(matrixA, helperA, (isRootRow ? matrixA.getVectorData() : foreignA), isRootRow);
      matrixAEnginePtr = std::move(helperA.getVectorData());
    }
    else{
      matrixAEnginePtr = std::move((isRootRow ? matrixA.getVectorData() : foreignA));
    }
    //BroadcastPanels((isRootColumn ? matrixB : foreignB), sizeB, isRootColumn, CommInfo.z, CommInfo.column);
    matrixBEnginePtr = (isRootColumn ? matrixB : foreignB);
    blas::engine::_trmm(&matrixAEnginePtr[0], matrixBEnginePtr, localDimensionM, localDimensionN, localDimensionM,
      (srcPackage.order == blas::Order::AblasColumnMajor ? localDimensionM : localDimensionN), srcPackage);
  }
  else{
    //BroadcastPanels((isRootColumn ? matrixA.getVectorData() : foreignA), sizeA, isRootColumn, CommInfo.z, CommInfo.column);
    if ((!std::is_same<StructureA,rect>::value) && (!std::is_same<StructureA,square>::value)){
      matrix<T,U,rect,Distribution,Offload> helperA(std::vector<T>(), matrixAnumColumns, matrixAnumRows, matrixAnumColumns, matrixAnumRows);
      getEnginePtr(matrixA, helperA, (isRootColumn ? matrixA.getVectorData() : foreignA), isRootColumn);
      matrixAEnginePtr = std::move(helperA.getVectorData());
    }
    else{
      matrixAEnginePtr = std::move((isRootColumn ? matrixA.getVectorData() : foreignA));
    }
    //BroadcastPanels((isRootRow ? matrixB : foreignB), sizeB, isRootRow, CommInfo.z, CommInfo.row);
    matrixBEnginePtr = (isRootRow ? matrixB : foreignB);
    blas::engine::_trmm(&matrixAEnginePtr[0], matrixBEnginePtr, localDimensionM, localDimensionN, localDimensionN,
      (srcPackage.order == blas::Order::AblasColumnMajor ? localDimensionM : localDimensionN), srcPackage);
  }
  MPI_Allreduce(MPI_IN_PLACE,matrixBEnginePtr, sizeB, mpi_type<T>::type, MPI_SUM, CommInfo.depth);
  std::memcpy(matrixB, matrixBEnginePtr, sizeB*sizeof(T));
  if ((srcPackage.side == blas::Side::AblasLeft) && (!isRootColumn)) delete[] foreignB;
  if ((srcPackage.side == blas::Side::AblasRight) && (!isRootRow)) delete[] foreignB;
}

template<typename MatrixAType, typename MatrixCType, typename CommType>
void summa::invoke(MatrixAType& matrixA, MatrixCType& matrixC, CommType&& CommInfo,
                     const blas::ArgPack_syrk<typename MatrixAType::ScalarType>& srcPackage, size_t methodKey){
  // Note: Internally, this routine uses gemm, not syrk, as its not possible for each processor to perform local MM with symmetric matrices
  //         given the data layout over the processor grid.

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  // Note: The routine will be C <- BA or AB, depending on the order in the srcPackage. B will always be the transposed matrix
  T* matrixAEnginePtr;
  T* matrixBEnginePtr;
  std::vector<T> matrixAEngineVector;
  std::vector<T> matrixBEngineVector;
  std::vector<T> foreignA;
  std::vector<T> foreignB;
  bool serializeKeyA = false;
  bool serializeKeyB = false;
  U localDimensionN = matrixC.getNumColumnsLocal();  // rows or columns, doesn't matter. They should be the same. matrixC is meant to be square
  U localDimensionK = (srcPackage.transposeA == blas::Transpose::AblasNoTrans ? matrixA.getNumColumnsLocal() : matrixA.getNumRowsLocal());

  MatrixAType matrixB = matrixA;
  util::transposeSwap(matrixB, std::forward<CommType>(CommInfo));

  if (methodKey == 0){
    if (srcPackage.transposeA == blas::Transpose::AblasNoTrans){
      distribute_bcast(matrixA,matrixB,std::forward<CommType>(CommInfo),matrixAEnginePtr,matrixBEnginePtr,
        matrixAEngineVector,matrixBEngineVector,foreignA,foreignB,serializeKeyA,serializeKeyB);
    }
    else{
      distribute_bcast(matrixB,matrixA,std::forward<CommType>(CommInfo),matrixBEnginePtr,matrixAEnginePtr,
        matrixBEngineVector,matrixAEngineVector,foreignB,foreignA,serializeKeyB,serializeKeyA);
    }
  }
  // No option for methodKey == 1

  T* matrixCforEnginePtr = matrixC.getRawData();
  if (srcPackage.beta == 0){
    if (srcPackage.transposeA == blas::Transpose::AblasNoTrans){
      blas::ArgPack_gemm<T> gemmArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasNoTrans, blas::Transpose::AblasTrans, -1., 1.);
      blas::engine::_gemm((serializeKeyA ? &matrixAEngineVector[0] : matrixAEnginePtr), (serializeKeyB ? &matrixBEngineVector[0] : matrixBEnginePtr),
        matrixCforEnginePtr, localDimensionN, localDimensionN, localDimensionK,
        localDimensionN, localDimensionN, localDimensionN, gemmArgs);
    }
    else{
      blas::ArgPack_gemm<T> gemmArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasTrans, blas::Transpose::AblasNoTrans, -1., 1.);
      blas::engine::_gemm((serializeKeyB ? &matrixBEngineVector[0] : matrixBEnginePtr), (serializeKeyA ? &matrixAEngineVector[0] : matrixAEnginePtr),
        matrixCforEnginePtr, localDimensionN, localDimensionN, localDimensionK,
        localDimensionK, localDimensionK, localDimensionN, gemmArgs);
    }
    collect(matrixCforEnginePtr,matrixC,std::forward<CommType>(CommInfo));
  }
  else{
    // This cancels out any affect beta could have. Beta is just not compatable with summa and must be handled separately
    std::vector<T> holdProduct(matrixC.getNumElems(),0);
    if (srcPackage.transposeA == blas::Transpose::AblasNoTrans){
      blas::ArgPack_gemm<T> gemmArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasNoTrans, blas::Transpose::AblasTrans, -1., 1.);
      blas::engine::_gemm((serializeKeyA ? &matrixAEngineVector[0] : matrixAEnginePtr), (serializeKeyB ? &matrixBEngineVector[0] : matrixBEnginePtr),
        &holdProduct[0], localDimensionN, localDimensionN, localDimensionK,
        localDimensionN, localDimensionN, localDimensionN, gemmArgs);
    }
    else{
      blas::ArgPack_gemm<T> gemmArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasTrans, blas::Transpose::AblasNoTrans, -1., 1.);
      blas::engine::_gemm((serializeKeyB ? &matrixBEngineVector[0] : matrixBEnginePtr), (serializeKeyA ? &matrixAEngineVector[0] : matrixAEnginePtr),
        &holdProduct[0], localDimensionN, localDimensionN, localDimensionK,
        localDimensionK, localDimensionK, localDimensionN, gemmArgs);
    }
    collect(&holdProduct[0],matrixC,std::forward<CommType>(CommInfo),1);

    // Future optimization: Reduce loop length by half since the update will be a symmetric matrix and only half will be used going forward.
    for (U i=0; i<holdProduct.size(); i++){
      matrixC.getRawData()[i] = srcPackage.beta*matrixC.getRawData()[i] + holdProduct[i];
    }
  }
}

template<typename MatrixAType, typename MatrixBType, typename MatrixCType, typename CommType>
void summa::invoke(MatrixAType& matrixA, MatrixBType& matrixB, MatrixCType& matrixC, typename MatrixAType::DimensionType matrixAcutXstart,
                    typename MatrixAType::DimensionType matrixAcutXend, typename MatrixAType::DimensionType matrixAcutYstart,
                    typename MatrixAType::DimensionType matrixAcutYend, typename MatrixBType::DimensionType matrixBcutZstart,
                    typename MatrixBType::DimensionType matrixBcutZend, typename MatrixBType::DimensionType matrixBcutXstart,
                    typename MatrixBType::DimensionType matrixBcutXend, typename MatrixCType::DimensionType matrixCcutZstart,
                    typename MatrixCType::DimensionType matrixCcutZend, typename MatrixCType::DimensionType matrixCcutYstart,
                    typename MatrixCType::DimensionType matrixCcutYend, CommType&& CommInfo,
                    const blas::ArgPack_gemm<typename MatrixAType::ScalarType>& srcPackage, bool cutA, bool cutB, bool cutC, size_t methodKey){
  // We will set up 3 matrices and call the method above.

  using StructureC = typename MatrixCType::StructureType;

  // I cannot use a fast-pass-by-value via move constructor because I don't want to corrupt the true matrices A,B,C. Other reasons as well.
  MatrixAType matA = getSubMatrix(matrixA, matrixAcutXstart, matrixAcutXend, matrixAcutYstart, matrixAcutYend, CommInfo.d, cutA);
  MatrixBType matB = getSubMatrix(matrixB, matrixBcutZstart, matrixBcutZend, matrixBcutXstart, matrixBcutXend, CommInfo.d, cutB);
  MatrixCType matC = getSubMatrix(matrixC, matrixCcutZstart, matrixCcutZend, matrixCcutYstart, matrixCcutYend, CommInfo.d, cutC);

  invoke((cutA ? matA : matrixA), (cutB ? matB : matrixB), (cutC ? matC : matrixC), std::forward<CommType>(CommInfo), srcPackage, methodKey);

  // reverse serialize, to put the solved piece of matrixC into where it should go.
  if (cutC){
    serialize<StructureC,StructureC>::invoke(matrixC, matC, matrixCcutZstart, matrixCcutZend, matrixCcutYstart, matrixCcutYend, true);
  }
}


template<typename MatrixAType, typename MatrixBType, typename CommType>
void summa::invoke(MatrixAType& matrixA, MatrixBType& matrixB, typename MatrixAType::DimensionType matrixAcutXstart,
                    typename MatrixAType::DimensionType matrixAcutXend, typename MatrixAType::DimensionType matrixAcutYstart,
                    typename MatrixAType::DimensionType matrixAcutYend, typename MatrixBType::DimensionType matrixBcutZstart,
                    typename MatrixBType::DimensionType matrixBcutZend, typename MatrixBType::DimensionType matrixBcutXstart,
                    typename MatrixBType::DimensionType matrixBcutXend, CommType&& CommInfo,
                    const blas::ArgPack_trmm<typename MatrixAType::ScalarType>& srcPackage, bool cutA, bool cutB, size_t methodKey){
  // We will set up 2 matrices and call the method above.

  using StructureB = typename MatrixBType::StructureType;

  // I cannot use a fast-pass-by-value via move constructor because I don't want to corrupt the true matrices A,B,C. Other reasons as well.
  MatrixAType matA = getSubMatrix(matrixA, matrixAcutXstart, matrixAcutXend, matrixAcutYstart, matrixAcutYend, CommInfo.d, cutA);
  MatrixBType matB = getSubMatrix(matrixB, matrixBcutZstart, matrixBcutZend, matrixBcutXstart, matrixBcutXend, CommInfo.d, cutB);
  invoke((cutA ? matA : matrixA), (cutB ? matB : matrixB), std::forward<CommType>(CommInfo), srcPackage, methodKey);

  // reverse serialize, to put the solved piece of matrixC into where it should go. Only if we need to
  if (cutB){
    serialize<StructureB,StructureB>::invoke(matrixB, matB, matrixBcutZstart, matrixBcutZend, matrixBcutXstart, matrixBcutXend, true);
  }
}

template<typename MatrixAType, typename MatrixCType, typename CommType>
void summa::invoke(MatrixAType& matrixA, MatrixCType& matrixC, typename MatrixAType::DimensionType matrixAcutXstart,
                    typename MatrixAType::DimensionType matrixAcutXend, typename MatrixAType::DimensionType matrixAcutYstart,
                    typename MatrixAType::DimensionType matrixAcutYend, typename MatrixCType::DimensionType matrixCcutZstart,
                    typename MatrixCType::DimensionType matrixCcutZend, typename MatrixCType::DimensionType matrixCcutXstart,
                    typename MatrixCType::DimensionType matrixCcutXend, CommType&& CommInfo,
                    const blas::ArgPack_syrk<typename MatrixAType::ScalarType>& srcPackage, bool cutA, bool cutC, size_t methodKey){
  // We will set up 2 matrices and call the method above.

  using StructureC = typename MatrixCType::StructureType;

  // I cannot use a fast-pass-by-value via move constructor because I don't want to corrupt the true matrices A,B,C. Other reasons as well.
  MatrixAType matA = getSubMatrix(matrixA, matrixAcutXstart, matrixAcutXend, matrixAcutYstart, matrixAcutYend, CommInfo.d, cutA);
  MatrixAType matC = getSubMatrix(matrixC, matrixCcutZstart, matrixCcutZend, matrixCcutXstart, matrixCcutXend, CommInfo.d, cutC);

  invoke((cutA ? matA : matrixA), (cutC ? matC : matrixC), std::forward<CommType>(CommInfo), srcPackage, methodKey);

  // reverse serialize, to put the solved piece of matrixC into where it should go.
  if (cutC){
    serialize<StructureC,StructureC>::invoke(matrixC, matC, matrixCcutZstart, matrixCcutZend, matrixCcutXstart, matrixCcutXend, true);
  }
}

template<typename MatrixAType, typename MatrixBType, typename CommType>
void summa::distribute_bcast(MatrixAType& matrixA, MatrixBType& matrixB, CommType&& CommInfo, typename MatrixAType::ScalarType*& matrixAEnginePtr,
                   typename MatrixBType::ScalarType*& matrixBEnginePtr, std::vector<typename MatrixAType::ScalarType>& matrixAEngineVector,
                   std::vector<typename MatrixBType::ScalarType>& matrixBEngineVector, std::vector<typename MatrixAType::ScalarType>& foreignA,
                   std::vector<typename MatrixBType::ScalarType>& foreignB, bool& serializeKeyA, bool& serializeKeyB){

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;
  using StructureA = typename MatrixAType::StructureType;
  using StructureB = typename MatrixBType::StructureType;
  using Distribution = typename MatrixAType::DistributionType;
  using Offload = typename MatrixAType::OffloadType;

  U localDimensionM = matrixA.getNumRowsLocal();
  U localDimensionN = matrixB.getNumColumnsLocal();
  U localDimensionK = matrixA.getNumColumnsLocal();
  std::vector<T>& dataA = matrixA.getVectorData(); 
  std::vector<T>& dataB = matrixB.getVectorData();
  U sizeA = matrixA.getNumElems();
  U sizeB = matrixB.getNumElems();
  bool isRootRow = ((CommInfo.x == CommInfo.z) ? true : false);
  bool isRootColumn = ((CommInfo.y == CommInfo.z) ? true : false);
  if (!isRootRow){ foreignA.resize(sizeA); }
  if (!isRootColumn){ foreignB.resize(sizeB); }

  // Check chunk size. If its 0, then bcast across rows and columns with no overlap
  if (CommInfo.num_chunks == 0){
    // distribute across rows
    MPI_Bcast((isRootRow ? &dataA[0] : &foreignA[0]), sizeA, mpi_type<T>::type, CommInfo.z, CommInfo.row);
    // distribute across columns
    MPI_Bcast((isRootColumn ? &dataB[0] : &foreignB[0]), sizeB, mpi_type<T>::type, CommInfo.z, CommInfo.column);
  }
  else{
    // initiate distribution across rows
    std::vector<MPI_Request> row_req(CommInfo.num_chunks);
    std::vector<MPI_Request> column_req(CommInfo.num_chunks);
    std::vector<MPI_Status> row_stat(CommInfo.num_chunks);
    std::vector<MPI_Status> column_stat(CommInfo.num_chunks);
    size_t offset = sizeA%CommInfo.num_chunks;
    size_t progress=0;
    for (size_t idx=0; idx < CommInfo.num_chunks; idx++){
      MPI_Ibcast((isRootRow ? &dataA[progress] : &foreignA[progress]), idx==(CommInfo.num_chunks-1) ? sizeA/CommInfo.num_chunks+offset : sizeA/CommInfo.num_chunks,
                 mpi_type<T>::type, CommInfo.z, CommInfo.row, &row_req[idx]);
      progress += sizeA/CommInfo.num_chunks;
    }
    // initiate distribution along columns and complete distribution across rows
    offset = sizeB%CommInfo.num_chunks;
    progress=0;
    for (size_t idx=0; idx < CommInfo.num_chunks; idx++){
      MPI_Ibcast((isRootColumn ? &dataB[progress] : &foreignB[progress]), idx==(CommInfo.num_chunks-1) ? sizeB/CommInfo.num_chunks+offset : sizeB/CommInfo.num_chunks,
                 mpi_type<T>::type, CommInfo.z, CommInfo.column, &column_req[idx]);
      progress += sizeB/CommInfo.num_chunks;
      MPI_Wait(&row_req[idx],&row_stat[idx]);
    }
    // complete distribution along columns
    for (size_t idx=0; idx < CommInfo.num_chunks; idx++){
      MPI_Wait(&column_req[idx],&column_stat[idx]);
    }
  }

  matrixAEnginePtr = (isRootRow ? &dataA[0] : &foreignA[0]);
  matrixBEnginePtr = (isRootColumn ? &dataB[0] : &foreignB[0]);
  if ((!std::is_same<StructureA,rect>::value) && (!std::is_same<StructureA,square>::value)){
    serializeKeyA = true;
    matrix<T,U,rect,Distribution,Offload> helperA(std::vector<T>(), localDimensionK, localDimensionM, localDimensionK, localDimensionM);
    getEnginePtr(matrixA, helperA, (isRootRow ? dataA : foreignA), isRootRow);
    matrixAEngineVector = std::move(helperA.getVectorData());
  }
  if ((!std::is_same<StructureB,rect>::value) && (!std::is_same<StructureB,square>::value)){
    serializeKeyB = true;
    matrix<T,U,rect,Distribution,Offload> helperB(std::vector<T>(), localDimensionN, localDimensionK, localDimensionN, localDimensionK);
    getEnginePtr(matrixB, helperB, (isRootColumn ? dataB : foreignB), isRootColumn);
    matrixBEngineVector = std::move(helperB.getVectorData());
  }
}


template<typename MatrixType, typename CommType>
void summa::collect(typename MatrixType::ScalarType* matrixEnginePtr, MatrixType& matrix, CommType&& CommInfo, size_t dir){

  using T = typename MatrixType::ScalarType;
  using U = typename MatrixType::DimensionType;

  U numElems = matrix.getNumElems();
  // Prevents buffer aliasing, which MPI does not like.
  if ((dir) || (matrixEnginePtr == matrix.getRawData())){
    MPI_Allreduce(MPI_IN_PLACE,matrixEnginePtr, numElems, mpi_type<T>::type, MPI_SUM, CommInfo.depth);
  }
  else{
    MPI_Allreduce(matrixEnginePtr, matrix.getRawData(), numElems, mpi_type<T>::type, MPI_SUM, CommInfo.depth);
  }
}

template<typename MatrixAType, typename MatrixBType, typename CommType>
void summa::distribute_allgather(MatrixAType& matrixA, MatrixBType& matrixB, CommType&& CommInfo,
                      std::vector<typename MatrixAType::ScalarType>& matrixAEngineVector, std::vector<typename MatrixBType::ScalarType>& matrixBEngineVector,
	              bool& serializeKeyA, bool& serializeKeyB){

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  int rowCommSize,columnCommSize,depthCommSize;
  MPI_Comm_size(CommInfo.row, &rowCommSize);
  MPI_Comm_size(CommInfo.column, &columnCommSize);
  MPI_Comm_size(CommInfo.depth, &depthCommSize);

/* Debugging notes
  How do I deal with serialization? Like, what if matrixA or B is a UT or LT? I do not want to communicate more than the packed data
    in that case obviously, but I also need to make sure that things are still in order, which I think is harder now than it was in distribute_bcast( method )
*/

  std::vector<T>& dataA = matrixA.getVectorData(); 
  U sizeA = matrixA.getNumElems();
  U sizeB = matrixB.getNumElems();

  // Allgathering matrixA is no problem because we store matrices column-wise
  // Note: using rowCommSize here instead of rowCommSize shouldn't matter for a 3D grid with uniform 2D slices
  U localNumRowsA = matrixA.getNumRowsLocal();
  U localNumColumnsA = matrixA.getNumColumnsLocal();
  int modA = localNumColumnsA%rowCommSize;
  int divA = localNumColumnsA/rowCommSize;
  U gatherSizeA = (modA == 0 ? sizeA : (divA+1)*rowCommSize*localNumRowsA);
  std::vector<T> collectMatrixA(gatherSizeA);
  int shift = (CommInfo.z + CommInfo.x) % rowCommSize;
  U dataAOffset = localNumRowsA*divA*shift;
  dataAOffset += std::min(shift,modA)*localNumRowsA;
  // matrixAEngineVector can stay with sizeA elements, because when we move data into it, we will get rid of the zeros.
  matrixAEngineVector.resize(sizeA);			// will need to change upon invoke changes
  int messageSizeA = gatherSizeA/rowCommSize;

  // Some processors will need to serialize
  if (modA && (shift >= modA)){
    std::vector<T> partitionMatrixA(messageSizeA,0);
    memcpy(&partitionMatrixA[0], &dataA[dataAOffset], (messageSizeA - localNumRowsA)*sizeof(T));  // truncation should be fine here. Rest is zeros
    MPI_Allgather(&partitionMatrixA[0], messageSizeA, mpi_type<T>::type, &collectMatrixA[0], messageSizeA, mpi_type<T>::type, CommInfo.row);
  }
  else{
    MPI_Allgather(&dataA[dataAOffset], messageSizeA, mpi_type<T>::type, &collectMatrixA[0], messageSizeA, mpi_type<T>::type, CommInfo.row);
  }


  // If pGridCoordZ != 0, then we need to re-shuffle the data. AllGather did not put into optimal order.
  if (CommInfo.z == 0){
    if (gatherSizeA == sizeA){
      matrixAEngineVector = std::move(collectMatrixA);
    }
    else{
      // first serialize into collectMatrixA itself by removing excess zeros
      // then move into matrixAEngineVector
      // Later optimizaion: avoid copying unless writeIndex < readInde
      U readIndex = 0;
      U writeIndex = 0;
      for (int i=0; i<rowCommSize; i++){
        U writeSize = (((modA == 0) || (i < modA)) ? messageSizeA : divA*localNumRowsA);
        memcpy(&collectMatrixA[writeIndex], &collectMatrixA[readIndex], writeSize*sizeof(T));
        writeIndex += writeSize;
        readIndex += messageSizeA;
      }
      collectMatrixA.resize(sizeA);
      matrixAEngineVector = std::move(collectMatrixA);
    }
  }
  else{
    matrixAEngineVector.resize(sizeA);
    U shuffleAoffset = messageSizeA*((rowCommSize - CommInfo.z)%rowCommSize);
    U stepA = 0;
    for (int i=0; i<rowCommSize; i++){
      // Don't really need the 2nd if statement condition like the one above. Actually, neither do
      U writeSize = (((i % rowCommSize) < modA) ? messageSizeA : divA*localNumRowsA);
      memcpy(&matrixAEngineVector[stepA], &collectMatrixA[shuffleAoffset], writeSize*sizeof(T));
      stepA += writeSize;
      shuffleAoffset += messageSizeA;
      shuffleAoffset %= gatherSizeA;
    }
  }
  
  // Now we Allgather partitions of matrix B
  U localNumRowsB = matrixB.getNumRowsLocal();
  U localNumColumnsB = matrixB.getNumColumnsLocal();
  int modB = localNumRowsB%columnCommSize;
  int divB = localNumRowsB/columnCommSize;
  int blockLengthB = (modB == 0 ? divB : divB +1);
  shift = (CommInfo.x + CommInfo.y) % columnCommSize;
  U dataBOffset = divB*shift;
  dataBOffset += std::min(shift, modB);       // WATCH: could be wrong
  U gatherSizeB = blockLengthB*columnCommSize*localNumColumnsB;
  int messageSizeB = gatherSizeB/columnCommSize;
  std::vector<T> collectMatrixB(gatherSizeB);			// will need to change upon invoke changes
  std::vector<T> partitionMatrixB(messageSizeB,0);			// Important to fill with zeros first
  // Special serialize. Can't use my Matrixserialize here.
  U writeSize = (((modB == 0)) || (shift < modB) ? blockLengthB : blockLengthB-1);
  for (U i=0; i<localNumColumnsB; i++){
    memcpy(&partitionMatrixB[i*blockLengthB], &matrixB.getRawData()[dataBOffset + i*localNumRowsB], writeSize*sizeof(T));
  }
  MPI_Allgather(&partitionMatrixB[0], partitionMatrixB.size(), mpi_type<T>::type, &collectMatrixB[0], partitionMatrixB.size(), mpi_type<T>::type, CommInfo.column);
/*
  // Allgathering matrixB is a problem for AMPI when using derived datatypes
  MPI_Datatype matrixBcolumnData;
  MPI_Type_vector(localNumColumnsB,blockLengthB,localNumRowsB,mpi_type<T>::type,&matrixBcolumnData);
  MPI_Type_commit(&matrixBcolumnData);
  U messageSizeB = sizeB/columnCommSize;
  MPI_Allgather(&dataB[dataBOffset], 1, matrixBcolumnData, &collectMatrixB[0], messageSizeB, mpi_type<T>::type, columnComm);
*/
  // Then need to re-shuffle the data in collectMatrixB because of the format Allgather puts the received data in 
  // Note: there is a particular order to it beyond the AllGather order. Depends what z coordinate we are on (that determines the shift)

  if ((rowCommSize == 1) && (columnCommSize == 1) && (depthCommSize == 1)){
    matrixBEngineVector = std::move(collectMatrixB);
  }
  else{
    matrixBEngineVector.resize(sizeB);			// will need to change upon invoke changes
    // Open question: Is this the most cache-efficient way to reshuffle the data?
    for (U i=0; i<localNumColumnsB; i++){
      // We always start in the same offset in the gatherBuffer
      U shuffleBoffset = messageSizeB*((columnCommSize - CommInfo.z)%columnCommSize);
      U saveStepB = i*localNumRowsB;
      for (int j=0; j<columnCommSize; j++){
        U writeSize = (((modB == 0) || ((j % columnCommSize) < modB)) ? blockLengthB : blockLengthB-1);
        U saveOffsetB = shuffleBoffset + i*blockLengthB;
/*
        if (saveStepB+writeSize > matrixBEngineVector.size())
        {
          std::cout << "saveStepB - " << saveStepB << ", writeSize - " << writeSize << ", matrixBEngineVector size - " << matrixBEngineVector.size() << ", j - " << j << ", columnCommSize - " << columnCommSize << ", i - " << i << ", localNumColumnsB - " << localNumColumnsB << ", localNumRowsB - " << localNumRowsB << ", blockSize - " << blockLengthB << ", sizeB - " << sizeB << ", size of collectMatrixB - " << collectMatrixB.size() << ", matrixB things - " << matrixB.getNumRowsLocal() << " " << matrixB.getNumColumnsLocal() << " " << matrixB.getNumElems() << std::endl;
          assert(saveStepB+writeSize <= matrixBEngineVector.size());
        }
*/
        memcpy(&matrixBEngineVector[saveStepB], &collectMatrixB[saveOffsetB], writeSize*sizeof(T));
        saveStepB += writeSize;
        shuffleBoffset += messageSizeB;
        shuffleBoffset %= gatherSizeB;
      }
    }
  }

/*
  if ((!std::is_same<StructureArg1<T,U,Distribution>,MatrixStructureRectangular<T,U,Distribution>>::value)
    && (!std::is_same<StructureArg1<T,U,Distribution>,MatrixStructuresquare<T,U,Distribution>>::value))		// compile time if statement. Branch prediction should be correct.
  {
    matrix<T,U,MatrixStructureRectangular,Distribution> helperA(std::vector<T>(), localDimensionK, localDimensionM, localDimensionK, localDimensionM);
    getEnginePtr(matrixA, helperA, (isRootRow ? dataA : foreignA), isRootRow);
    matrixAEngineVector = std::move(helperA.getVectorData());
  }
  if ((!std::is_same<StructureArg2<T,U,Distribution>,MatrixStructureRectangular<T,U,Distribution>>::value)
    && (!std::is_same<StructureArg2<T,U,Distribution>,MatrixStructuresquare<T,U,Distribution>>::value))		// compile time if statement. Branch prediction should be correct.
  {
    matrix<T,U,MatrixStructureRectangular,Distribution> helperB(std::vector<T>(), localDimensionN, localDimensionK, localDimensionN, localDimensionK);
    getEnginePtr(matrixB, helperB, (isRootColumn ? dataB : foreignB), isRootColumn);
    matrixBEngineVector = std::move(helperB.getVectorData());
  }
*/
}

/*
template<typename T, typename U>
void summa::BroadcastPanels(T*& data, U size, bool isRoot, size_t pGridCoordZ, MPI_Comm panel){

  if (isRoot){
    MPI_Bcast(data, size, mpi_type<T>::type, pGridCoordZ, panel);
  }
  else{
    // TODO: Is this causing a memory leak? Usually I would be overwriting vector allocated memory. Not sure if this will cause issues or if
    //         the vector will still delete itself.
    data = new T[size];
    MPI_Bcast(data, size, mpi_type<T>::type, pGridCoordZ, panel);
  }
}
*/

template<typename MatrixSrcType, typename MatrixDestType>
void summa::getEnginePtr(MatrixSrcType& matrixSrc, MatrixDestType& matrixDest, std::vector<typename MatrixSrcType::ScalarType>& data, bool isRoot){

  using StructureSrc = typename MatrixSrcType::StructureType;
  using StructureDest = typename MatrixDestType::StructureType;

  // Need to separate the below out into its own function that will not get instantied into object code
  //   unless it passes the test above. This avoids template-enduced template compiler errors
  if (!isRoot){
    MatrixSrcType matrixToinvoke(std::move(data), matrixSrc.getNumColumnsLocal(), matrixSrc.getNumRowsLocal(), matrixSrc.getNumColumnsGlobal(), matrixSrc.getNumRowsGlobal(), true);
    serialize<StructureSrc,StructureDest>::invoke(matrixToinvoke, matrixDest);
  }
  else{
    // If code path gets here, StructureArg must be a LT or UT, so we need to serialize into a Square, not a Rectangular
    serialize<StructureSrc,StructureDest>::invoke(matrixSrc, matrixDest);
  }
}


template<typename MatrixType>
MatrixType summa::getSubMatrix(MatrixType& srcMatrix, typename MatrixType::DimensionType matrixArgColumnStart, typename MatrixType::DimensionType matrixArgColumnEnd,
                              typename MatrixType::DimensionType matrixArgRowStart, typename MatrixType::DimensionType matrixArgRowEnd,
		              size_t sliceDim, bool getSub){

  using T = typename MatrixType::ScalarType;
  using U = typename MatrixType::DimensionType;
  using Structure = typename MatrixType::StructureType;

  if (getSub){
    U numColumns = matrixArgColumnEnd - matrixArgColumnStart;
    U numRows = matrixArgRowEnd - matrixArgRowStart;
    MatrixType fillMatrix(std::vector<T>(), numColumns, numRows, numColumns*sliceDim, numRows*sliceDim);
    serialize<Structure,Structure>::invoke(srcMatrix, fillMatrix, matrixArgColumnStart, matrixArgColumnEnd, matrixArgRowStart, matrixArgRowEnd);
    return fillMatrix;			// I am returning an rvalue
  }
  else{
    // return cheap garbage.
    return MatrixType(0,0,1,1);
  }
}
}
