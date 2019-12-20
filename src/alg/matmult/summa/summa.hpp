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
                     const blas::ArgPack_gemm<typename MatrixAType::ScalarType>& srcPackage){

  // Use tuples so we don't have to pass multiple things by reference.
  // Also this way, we can take advantage of the new pass-by-value move semantics that are efficient
  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  T* matrixAEnginePtr;
  T* matrixBEnginePtr;
  std::vector<T> matrixAEngineVector;
  std::vector<T> matrixBEngineVector;
  bool serializeKeyA = false;
  bool serializeKeyB = false;
  U localDimensionM = (srcPackage.transposeA == blas::Transpose::AblasNoTrans ? matrixA.getNumRowsLocal() : matrixA.getNumColumnsLocal());
  U localDimensionN = (srcPackage.transposeB == blas::Transpose::AblasNoTrans ? matrixB.getNumColumnsLocal() : matrixB.getNumRowsLocal());
  U localDimensionK = (srcPackage.transposeA == blas::Transpose::AblasNoTrans ? matrixA.getNumColumnsLocal() : matrixA.getNumRowsLocal());

  distribute_bcast(matrixA,matrixB,std::forward<CommType>(CommInfo),matrixAEnginePtr,matrixBEnginePtr,matrixAEngineVector,matrixBEngineVector,serializeKeyA,serializeKeyB);

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
                     const blas::ArgPack_trmm<typename MatrixAType::ScalarType>& srcPackage){

  // Use tuples so we don't have to pass multiple things by reference.
  // Also this way, we can take advantage of the new pass-by-value move semantics that are efficient
  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  T* matrixAEnginePtr;
  T* matrixBEnginePtr;
  std::vector<T> matrixAEngineVector;
  std::vector<T> matrixBEngineVector;
  bool serializeKeyA = false;
  bool serializeKeyB = false;
  U localDimensionM = matrixB.getNumRowsLocal();
  U localDimensionN = matrixB.getNumColumnsLocal();

  if (srcPackage.side == blas::Side::AblasLeft){
    distribute_bcast(matrixA, matrixB, std::forward<CommType>(CommInfo), matrixAEnginePtr, matrixBEnginePtr, matrixAEngineVector, matrixBEngineVector,
      serializeKeyA, serializeKeyB);
    blas::engine::_trmm((serializeKeyA ? &matrixAEngineVector[0] : matrixAEnginePtr), (serializeKeyB ? &matrixBEngineVector[0] : matrixBEnginePtr),
      localDimensionM, localDimensionN, localDimensionM, (srcPackage.order == blas::Order::AblasColumnMajor ? localDimensionM : localDimensionN),
      srcPackage);
  }
  else{
    distribute_bcast(matrixB, matrixA, std::forward<CommType>(CommInfo), matrixBEnginePtr, matrixAEnginePtr, matrixBEngineVector, matrixAEngineVector, serializeKeyB, serializeKeyA);
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
                     const blas::ArgPack_syrk<typename MatrixAType::ScalarType>& srcPackage){
  // Note: Internally, this routine uses gemm, not syrk, as its not possible for each processor to perform local MM with symmetric matrices
  //         given the data layout over the processor grid.

  using T = typename MatrixAType::ScalarType;
  using U = typename MatrixAType::DimensionType;

  // Note: The routine will be C <- BA or AB, depending on the order in the srcPackage. B will always be the transposed matrix
  T* matrixAEnginePtr;
  T* matrixBEnginePtr;
  std::vector<T> matrixAEngineVector;
  std::vector<T> matrixBEngineVector;
  bool serializeKeyA = false;
  bool serializeKeyB = false;
  U localDimensionN = matrixC.getNumColumnsLocal();  // rows or columns, doesn't matter. They should be the same. matrixC is meant to be square
  U localDimensionK = (srcPackage.transposeA == blas::Transpose::AblasNoTrans ? matrixA.getNumColumnsLocal() : matrixA.getNumRowsLocal());

  MatrixAType matrixB = matrixA;
  util::transposeSwap(matrixB, std::forward<CommType>(CommInfo));

  if (srcPackage.transposeA == blas::Transpose::AblasNoTrans){
    distribute_bcast(matrixA,matrixB,std::forward<CommType>(CommInfo),matrixAEnginePtr,matrixBEnginePtr,
      matrixAEngineVector,matrixBEngineVector,serializeKeyA,serializeKeyB);
  }
  else{
    distribute_bcast(matrixB,matrixA,std::forward<CommType>(CommInfo),matrixBEnginePtr,matrixAEnginePtr,
      matrixBEngineVector,matrixAEngineVector,serializeKeyB,serializeKeyA);
  }

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
                    const blas::ArgPack_gemm<typename MatrixAType::ScalarType>& srcPackage, bool cutA, bool cutB, bool cutC){
  // We will set up 3 matrices and call the method above.

  using StructureC = typename MatrixCType::StructureType;

  // I cannot use a fast-pass-by-value via move constructor because I don't want to corrupt the true matrices A,B,C. Other reasons as well.
  MatrixAType matA = getSubMatrix(matrixA, matrixAcutXstart, matrixAcutXend, matrixAcutYstart, matrixAcutYend, CommInfo.d, cutA);
  MatrixBType matB = getSubMatrix(matrixB, matrixBcutZstart, matrixBcutZend, matrixBcutXstart, matrixBcutXend, CommInfo.d, cutB);
  MatrixCType matC = getSubMatrix(matrixC, matrixCcutZstart, matrixCcutZend, matrixCcutYstart, matrixCcutYend, CommInfo.d, cutC);

  invoke((cutA ? matA : matrixA), (cutB ? matB : matrixB), (cutC ? matC : matrixC), std::forward<CommType>(CommInfo), srcPackage);

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
                    const blas::ArgPack_trmm<typename MatrixAType::ScalarType>& srcPackage, bool cutA, bool cutB){
  // We will set up 2 matrices and call the method above.

  using StructureB = typename MatrixBType::StructureType;

  // I cannot use a fast-pass-by-value via move constructor because I don't want to corrupt the true matrices A,B,C. Other reasons as well.
  MatrixAType matA = getSubMatrix(matrixA, matrixAcutXstart, matrixAcutXend, matrixAcutYstart, matrixAcutYend, CommInfo.d, cutA);
  MatrixBType matB = getSubMatrix(matrixB, matrixBcutZstart, matrixBcutZend, matrixBcutXstart, matrixBcutXend, CommInfo.d, cutB);
  invoke((cutA ? matA : matrixA), (cutB ? matB : matrixB), std::forward<CommType>(CommInfo), srcPackage);

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
                    const blas::ArgPack_syrk<typename MatrixAType::ScalarType>& srcPackage, bool cutA, bool cutC){
  // We will set up 2 matrices and call the method above.

  using StructureC = typename MatrixCType::StructureType;

  // I cannot use a fast-pass-by-value via move constructor because I don't want to corrupt the true matrices A,B,C. Other reasons as well.
  MatrixAType matA = getSubMatrix(matrixA, matrixAcutXstart, matrixAcutXend, matrixAcutYstart, matrixAcutYend, CommInfo.d, cutA);
  MatrixAType matC = getSubMatrix(matrixC, matrixCcutZstart, matrixCcutZend, matrixCcutXstart, matrixCcutXend, CommInfo.d, cutC);

  invoke((cutA ? matA : matrixA), (cutC ? matC : matrixC), std::forward<CommType>(CommInfo), srcPackage);

  // reverse serialize, to put the solved piece of matrixC into where it should go.
  if (cutC){
    serialize<StructureC,StructureC>::invoke(matrixC, matC, matrixCcutZstart, matrixCcutZend, matrixCcutXstart, matrixCcutXend, true);
  }
}

template<typename MatrixAType, typename MatrixBType, typename CommType>
void summa::distribute_bcast(MatrixAType& matrixA, MatrixBType& matrixB, CommType&& CommInfo, typename MatrixAType::ScalarType*& matrixAEnginePtr,
                   typename MatrixBType::ScalarType*& matrixBEnginePtr, std::vector<typename MatrixAType::ScalarType>& matrixAEngineVector,
                   std::vector<typename MatrixBType::ScalarType>& matrixBEngineVector, bool& serializeKeyA, bool& serializeKeyB){

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
  std::vector<T> foreignA;
  std::vector<T> foreignB;
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
