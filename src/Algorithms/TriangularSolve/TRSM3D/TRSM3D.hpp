/* Author: Edward Hutter */

template<typename T, typename U, typename OffloadType>
template<
  template<typename,typename, template<typename,typename,int> class> class StructureArg,
  template<typename,typename, template<typename,typename,int> class> class StructureTriangularArg,
  template<typename,typename,int> class Distribution
>
void TRSM3D<T,U,OffloadType>::iSolveLowerLeft(
  Matrix<T,U,StructureArg,Distribution>& matrixA,
  Matrix<T,U,StructureTriangularArg,Distribution>& matrixL,
  Matrix<T,U,StructureTriangularArg,Distribution>& matrixLI,
  std::vector<U>& baseCaseDimList,
  blasEngineArgumentPackage_gemm<T>& gemmPackage,
  blasEngineArgumentPackage_trmm<T>& trmmPackage,
  MPI_Comm commWorld,
  std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D){
}


// For solving AU=B for A. But note that B is A, and we are modifying B in place to solve for A
template<typename T, typename U, typename OffloadType>
template<
  template<typename,typename, template<typename,typename,int> class> class StructureArg,
  template<typename,typename, template<typename,typename,int> class> class StructureTriangularArg,
  template<typename,typename,int> class Distribution
>
void TRSM3D<T,U,OffloadType>::iSolveUpperLeft(
                       Matrix<T,U,StructureArg,Distribution>& matrixA,
                       Matrix<T,U,StructureTriangularArg,Distribution>& matrixU,
                       Matrix<T,U,StructureTriangularArg,Distribution>& matrixUI,
                       std::vector<U>& baseCaseDimList,
                       blasEngineArgumentPackage_gemm<T>& gemmPackage,
                       blasEngineArgumentPackage_trmm<T>& trmmPackage,
                       MPI_Comm commWorld,
                       std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D
                     ){
  TAU_FSTART(TRSM3D::iSolveUpperLeft);
  int pGridDimensionSize;
  MPI_Comm_size(std::get<0>(commInfo3D), &pGridDimensionSize);
/*
  blasEngineArgumentPackage_trmm<T> trmmPackage;
  trmmPackage.order = blasEngineOrder::AblasColumnMajor;
  trmmPackage.side = blasEngineSide::AblasRight;
  trmmPackage.uplo = blasEngineUpLo::AblasLower;
  trmmPackage.diag = blasEngineDiag::AblasNonUnit;
  trmmPackage.transposeA = blasEngineTranspose::AblasTrans;
  trmmPackage.alpha = 1.;
*/
  // to catch debugging issues, assert that this has at least one size
  assert(baseCaseDimList.size());

  // Lets operate on individual columns at a time
  // Potential optimization 1): Don't use MM3D if the columns are too skinny in relation to the block size!
     // Or this could just be taken care of when we tune block sizes?
  // Potential optimization 2) Lots of serializing going on with each MM3D, this needs to be reduced.

  // Communicate matrixA and matrixU and matrixUI immediately.
    // These 3 matrices should never need to be communicated again.
  // matrixB however will need to be AllReduced at each iteration so that final results can be summed and updated before next iteration

  U matAendX = matrixA.getNumColumnsLocal();
  U matAendY = matrixA.getNumRowsLocal();
  U matUendX = matrixU.getNumColumnsLocal();
  U matUendY = matrixU.getNumRowsLocal();

  U offset1 = 0;
  U offset2 = (baseCaseDimList.size() < 1 ? matAendX : baseCaseDimList[0]);
  U offset3 = 0;
  for (U i=0; i<baseCaseDimList.size()/*numBlockColumns*/; i++){
    // Update the current column by accumulating the updates via MM
    gemmPackage.alpha = -1;
    gemmPackage.beta = 1.;

    // Only update once first panel is solved
    if (i>0){
      // As i increases, the size of these updates gets smaller.
      // Special handling. This might only work since the triangular matrix is square, which should be ok
      U arg1 = (gemmPackage.transposeB == blasEngineTranspose::AblasNoTrans ? offset1 : offset3);
      U arg2 = (gemmPackage.transposeB == blasEngineTranspose::AblasNoTrans ? matUendX : offset1);
      U arg3 = (gemmPackage.transposeB == blasEngineTranspose::AblasNoTrans ? offset3 : offset1);
      U arg4 = (gemmPackage.transposeB == blasEngineTranspose::AblasNoTrans ? offset1 : matUendX);

      Matrix<T,U,MatrixStructureRectangle,Distribution> matrixUpartition(std::vector<T>(), arg2-arg1, arg4-arg3, (arg2-arg1)*pGridDimensionSize, (arg4-arg3)*pGridDimensionSize);
      Serializer<T,U,StructureTriangularArg,MatrixStructureRectangle>::Serialize(matrixU, matrixUpartition,
        arg1, arg2, arg3, arg4);
      MM3D<T,U,OffloadType>::Multiply(matrixA.getRawData()+(offset3*matAendY), matrixUpartition, matrixA.getRawData()+(offset1*matAendY),
        offset1-offset3, matAendY, arg2-arg1, arg4-arg3, matAendX-offset1, matAendY, commWorld, commInfo3D, gemmPackage);
    }

    // Solve via TRMM
    U save1 = offset2-offset1;
    // New optimization: prevent this copy if we are doing TRSM only at the top level
    // Note: this change might be rendered useless now that I modified CFR3D.hpp with a similar optimization for that top level of TRSM
    if (baseCaseDimList.size() <= 1){
      MM3D<T,U,OffloadType>::Multiply(matrixUI, matrixA.getRawData()+(offset1*matAendY), save1, save1, save1, matAendY, commWorld, commInfo3D, trmmPackage);
    }
    else{
      Matrix<T,U,StructureTriangularArg,Distribution> matrixUIpartition(std::vector<T>(), save1, save1, save1*pGridDimensionSize, save1*pGridDimensionSize);
      Serializer<T,U,StructureTriangularArg,StructureTriangularArg>::Serialize(matrixUI, matrixUIpartition, offset1, offset2, offset1, offset2);
      MM3D<T,U,OffloadType>::Multiply(matrixUIpartition, matrixA.getRawData()+(offset1*matAendY), save1, save1, save1, matAendY, commWorld, commInfo3D, trmmPackage);
    }
    if ((i+1) < baseCaseDimList.size()){
      // Update the offsets
      offset3 = offset1;
      offset1 = offset2;
      offset2 += baseCaseDimList[i+1];
    }
  }
  TAU_FSTOP(TRSM3D::iSolveUpperLeft);
}


// For solving RA=A for A
template<typename T, typename U, typename OffloadType>
template<
  template<typename,typename, template<typename,typename,int> class> class StructureArg,
  template<typename,typename, template<typename,typename,int> class> class StructureTriangularArg,
  template<typename,typename,int> class Distribution
>
void TRSM3D<T,U,OffloadType>::iSolveLowerRight(
  Matrix<T,U,StructureTriangularArg,Distribution>& matrixR,
  Matrix<T,U,StructureTriangularArg,Distribution>& matrixRI,
  Matrix<T,U,StructureArg,Distribution>& matrixA,
  std::vector<U>& baseCaseDimList,
  blasEngineArgumentPackage_gemm<T>& gemmPackage,
  blasEngineArgumentPackage_trmm<T>& trmmPackage,
  MPI_Comm commWorld,
  std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D
  ){
  TAU_FSTART(TRSM3D::iSolveLowerRight);
  int pGridDimensionSize;
  MPI_Comm_size(std::get<0>(commInfo3D), &pGridDimensionSize);

  // to catch debugging issues, assert that this has at least one size
  assert(baseCaseDimList.size());

  // Lets operate on individual columns at a time
  // Potential optimization 1): Don't use MM3D if the columns are too skinny in relation to the block size!
     // Or this could just be taken care of when we tune block sizes?
  // Potential optimization 2) Lots of serializing going on with each MM3D, this needs to be reduced.

  U matAendX = matrixA.getNumColumnsLocal();
  U matAendY = matrixA.getNumRowsLocal();
  U matRendX = matrixR.getNumColumnsLocal();
  U matRendY = matrixR.getNumRowsLocal();

  U offset1 = 0;
  U offset2 = (baseCaseDimList.size() < 1 ? matAendX : baseCaseDimList[0]);
  U offset3 = 0;
  for (U i=0; i<baseCaseDimList.size()/*numBlockColumns*/; i++){

    // Update the current column by accumulating the updates via MM
    gemmPackage.alpha = -1;
    gemmPackage.beta = 1.;

    // Only update once first panel is solved
    if (i>0){
      // As i increases, the size of these updates gets smaller.
      // Special handling. This might only work since the triangular matrix is square, which should be ok

      // Note that the beginning cases might not be correct. They are not currently used for anything though.
      U arg1 = (gemmPackage.transposeA == blasEngineTranspose::AblasNoTrans ? offset3 : offset1);
      U arg2 = (gemmPackage.transposeA == blasEngineTranspose::AblasNoTrans ? offset1 : matRendX);
      U arg3 = (gemmPackage.transposeA == blasEngineTranspose::AblasNoTrans ? offset3 : offset3);
      U arg4 = (gemmPackage.transposeA == blasEngineTranspose::AblasNoTrans ? offset1 : offset1);

      Matrix<T,U,MatrixStructureRectangle,Distribution> matrixRpartition(std::vector<T>(), arg2-arg1, arg4-arg3, (arg2-arg1)*pGridDimensionSize, (arg4-arg3)*pGridDimensionSize);
      Serializer<T,U,StructureTriangularArg,MatrixStructureRectangle>::Serialize(matrixR, matrixRpartition, arg1, arg2, arg3, arg4);
      MM3D<T,U,OffloadType>::Multiply(matrixRpartition, matrixA, matrixA, 0, arg2-arg1, 0, arg4-arg3, 0, matAendX, offset3, offset1,
        0, matAendX, offset1, matAendY, commWorld, commInfo3D, gemmPackage, false, true, true);
    }

    // Solve via MM
    MM3D<T,U,OffloadType>::Multiply(matrixRI, matrixA, offset1, offset2, offset1, offset2, 0, matAendX, offset1, offset2, commWorld, commInfo3D, trmmPackage, true, true);

    if ((i+1) < baseCaseDimList.size()){
      // Update the offsets
      offset3 = offset1;
      offset1 = offset2;
      offset2 += baseCaseDimList[i+1];
    }
  }
  TAU_FSTOP(TRSM3D::iSolveLowerRight);
}


template<typename T, typename U, typename OffloadType>
template<
  template<typename,typename, template<typename,typename,int> class> class StructureArg,
  template<typename,typename, template<typename,typename,int> class> class StructureTriangularArg,
  template<typename,typename,int> class Distribution
>
void TRSM3D<T,U,OffloadType>::iSolveUpperRight(
  Matrix<T,U,StructureTriangularArg,Distribution>& matrixU,
  Matrix<T,U,StructureTriangularArg,Distribution>& matrixUI,
  Matrix<T,U,StructureArg,Distribution>& matrixA,
  std::vector<U>& baseCaseDimList,
  blasEngineArgumentPackage_gemm<T>& gemmPackage,
  blasEngineArgumentPackage_trmm<T>& trmmPackage,
  MPI_Comm commWorld,
  std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D
  ){
}
