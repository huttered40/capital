/* Author: Edward Hutter */

template<typename T, typename U, template<typename, typename> class blasEngine>
template<
  template<typename,typename, template<typename,typename,int> class> class StructureArg,
  template<typename,typename,int> class Distribution
>
void TRSM3D<T,U,blasEngine>::iSolveLowerLeft(
  Matrix<T,U,StructureArg,Distribution>& matrixA,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixL,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixLI,
  Matrix<T,U,StructureArg,Distribution>& matrixB,
  U matAstartX,
  U matAendX,
  U matAstartY,
  U matAendY,
  U matLstartX,
  U matLendX,
  U matLstartY,
  U matLendY,
  U matBstartX,
  U matBendX,
  U matBstartY,
  U matBendY,
  std::vector<U>& baseCaseDimList,
  blasEngineArgumentPackage_gemm<T>& srcPackage,
  MPI_Comm commWorld,
  int MM_id,
  int TR_id)
{
}


// For solving AU=B for A
template<typename T, typename U, template<typename, typename> class blasEngine>
template<
  template<typename,typename, template<typename,typename,int> class> class StructureArg,
  template<typename,typename,int> class Distribution
>
void TRSM3D<T,U,blasEngine>::iSolveUpperLeft(
                       Matrix<T,U,StructureArg,Distribution>& matrixA,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixU,
                       Matrix<T,U,MatrixStructureSquare,Distribution>& matrixUI,
                       Matrix<T,U,StructureArg,Distribution>& matrixB,
                       U matAstartX,
                       U matAendX,
                       U matAstartY,
                       U matAendY,
                       U matUstartX,
                       U matUendX,
                       U matUstartY,
                       U matUendY,
                       U matBstartX,
                       U matBendX,
                       U matBstartY,
                       U matBendY,
                       std::vector<U>& baseCaseDimList,
                       blasEngineArgumentPackage_gemm<T>& srcPackage,
                       MPI_Comm commWorld,
                       int MM_id,
                       int TR_id         // allows for benchmarking to see which version is faster 
                     )
{
  int rank,size;
  MPI_Comm_rank(commWorld, &rank);
  MPI_Comm_size(commWorld, &size);
  int pGridDimensionSize = std::nearbyint(std::pow(size,1./3.));
  int helper = pGridDimensionSize;
  helper *= helper;

  // to catch debugging issues, assert that this has at least one size
  assert(baseCaseDimList.size());

  // Lets operate on individual columns at a time
  // Potential optimization 1): Don't use MM3D if the columns are too skinny in relation to the block size!
     // Or this could just be taken care of when we tune block sizes?
  // Potential optimization 2) Lots of serializing going on with each MM3D, this needs to be reduced.

  // Communicate matrixA and matrixU and matrixUI immediately.
    // These 3 matrices should never need to be communicated again.
  // matrixB however will need to be AllReduced at each iteration so that final results can be summed and updated before next iteration


  U offset1 = 0;
  U offset2 = (baseCaseDimList.size() < 1 ? matAendX : baseCaseDimList[0]);
  U offset3 = 0;
  for (U i=0; i<baseCaseDimList.size()/*numBlockColumns*/; i++)
  {
    // Update the current column by accumulating the updates via MM
    srcPackage.alpha = -1;
    srcPackage.beta = 1.;

    // Only update once first panel is solved
    if (i>0)
    {
      // As i increases, the size of these updates gets smaller.
      // Special handling. This might only work since the triangular matrix is square, which should be ok
      U arg1 = (srcPackage.transposeB == blasEngineTranspose::AblasNoTrans ? (matUstartX + offset1) : (matUstartY + offset3));
      U arg2 = (srcPackage.transposeB == blasEngineTranspose::AblasNoTrans ? matUendX : (matUstartY+offset1));
      U arg3 = (srcPackage.transposeB == blasEngineTranspose::AblasNoTrans ? (matUstartY + offset3) : (matUstartX + offset1));
      U arg4 = (srcPackage.transposeB == blasEngineTranspose::AblasNoTrans ? (matUstartY+offset1) : matUendX);

      MM3D<T,U,blasEngine>::Multiply(matrixA, matrixU, matrixB, matAstartX+offset3, matAstartX+offset1, matAstartY, matAendY,
        arg1, arg2, arg3, arg4, matBstartX+offset1, matBendX, matBstartY, matBendY, commWorld, srcPackage, true, true, true, MM_id);
    }

    // Solve via MM
    // Future optimization: We are doing the same serialization over and over again between the updates and the MM. Try to reduce this!
    srcPackage.alpha = 1;
    srcPackage.beta = 0;
    // Future optimization: for 1 processor, we don't want to serialize, so change true to false
    // Future optimization: to reduce flops, can't we do a TRSM here instead of a MM? Or no?
    MM3D<T,U,blasEngine>::Multiply(matrixB, matrixUI, matrixA, matBstartX+offset1, matBstartX+offset2, matBstartY, matBendY,
      matUstartX+offset1, matUstartX+offset2, matUstartY+offset1, matUstartY+offset2, matAstartX+offset1, matAstartX+offset2,
      matAstartY, matAendY, commWorld, srcPackage, true, true, true, MM_id);

    if ((i+1) < baseCaseDimList.size())
    {
      // Update the offsets
      offset3 = offset1;
      offset1 = offset2;
      offset2 += baseCaseDimList[i+1];
    }
  }
}


// For solving RA=B for A
template<typename T, typename U, template<typename, typename> class blasEngine>
template<
  template<typename,typename, template<typename,typename,int> class> class StructureArg,
  template<typename,typename,int> class Distribution
>
void TRSM3D<T,U,blasEngine>::iSolveLowerRight(
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixR,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixRI,
  Matrix<T,U,StructureArg,Distribution>& matrixA,
  Matrix<T,U,StructureArg,Distribution>& matrixB,
  U matRstartX,
  U matRendX,
  U matRstartY,
  U matRendY,
  U matAstartX,
  U matAendX,
  U matAstartY,
  U matAendY,
  U matBstartX,
  U matBendX,
  U matBstartY,
  U matBendY,
  std::vector<U>& baseCaseDimList,
  blasEngineArgumentPackage_gemm<T>& srcPackage,
  MPI_Comm commWorld,
  int MM_id,
  int TR_id)         // allows for benchmarking to see which version is faster 
{
  int rank,size;
  MPI_Comm_rank(commWorld, &rank);
  MPI_Comm_size(commWorld, &size);
  int pGridDimensionSize = std::nearbyint(std::pow(size,1./3.));
  int helper = pGridDimensionSize;
  helper *= helper;

  // to catch debugging issues, assert that this has at least one size
  assert(baseCaseDimList.size());

  // Lets operate on individual columns at a time
  // Potential optimization 1): Don't use MM3D if the columns are too skinny in relation to the block size!
     // Or this could just be taken care of when we tune block sizes?
  // Potential optimization 2) Lots of serializing going on with each MM3D, this needs to be reduced.

  U offset1 = 0;
  U offset2 = (baseCaseDimList.size() < 1 ? matAendX : baseCaseDimList[0]);
  U offset3 = 0;
  for (U i=0; i<baseCaseDimList.size()/*numBlockColumns*/; i++)
  {

    // Update the current column by accumulating the updates via MM
    srcPackage.alpha = -1;
    srcPackage.beta = 1.;

    // Only update once first panel is solved
    if (i>0)
    {
      // As i increases, the size of these updates gets smaller.
      // Special handling. This might only work since the triangular matrix is square, which should be ok

      // Note that the beginning cases might not be correct. They are not currently used for anything though.
      U arg1 = (srcPackage.transposeA == blasEngineTranspose::AblasNoTrans ? (matRstartY + offset3) : (matRstartX + offset1));
      U arg2 = (srcPackage.transposeA == blasEngineTranspose::AblasNoTrans ? (matRstartY+offset1) : matRendX);
      U arg3 = (srcPackage.transposeA == blasEngineTranspose::AblasNoTrans ? (matRstartY + offset3) : (matRstartY + offset3));
      U arg4 = (srcPackage.transposeA == blasEngineTranspose::AblasNoTrans ? (matRstartY+offset1) : matRstartY+offset1);

      MM3D<T,U,blasEngine>::Multiply(matrixR, matrixA, matrixB, arg1, arg2, arg3, arg4, matAstartX, matAendX, matAstartY+offset3, matAstartY+offset1,
        matBstartX, matBendX, matBstartY+offset1, matBendY, commWorld, srcPackage, true, true, true, MM_id);
    }

    // Solve via MM
    // Future optimization: We are doing the same serialization over and over again between the updates and the MM. Try to reduce this!
    srcPackage.alpha = 1;
    srcPackage.beta = 0;
    // Future optimization: for 1 processor, we don't want to serialize, so change true to false
    // Future optimization: to reduce flops, can't we do a TRSM here instead of a MM? Or no?
    MM3D<T,U,blasEngine>::Multiply(matrixRI, matrixB, matrixA, matRstartX+offset1, matRstartX+offset2, matRstartY+offset1, matRstartY+offset2,
      matBstartX, matBendX, matBstartY+offset1, matBstartY+offset2, matAstartX, matAendX,
        matAstartY+offset1, matAstartY+offset2, commWorld, srcPackage, true, true, true, MM_id);

    if ((i+1) < baseCaseDimList.size())
    {
      // Update the offsets
      offset3 = offset1;
      offset1 = offset2;
      offset2 += baseCaseDimList[i+1];
    }
  }
}


template<typename T, typename U, template<typename, typename> class blasEngine>
template<
  template<typename,typename, template<typename,typename,int> class> class StructureArg,
  template<typename,typename,int> class Distribution
>
void TRSM3D<T,U,blasEngine>::iSolveUpperRight(
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixU,
  Matrix<T,U,MatrixStructureSquare,Distribution>& matrixUI,
  Matrix<T,U,StructureArg,Distribution>& matrixA,
  Matrix<T,U,StructureArg,Distribution>& matrixB,
  U matUstartX,
  U matUendX,
  U matUstartY,
  U matUendY,
  U matAstartX,
  U matAendX,
  U matAstartY,
  U matAendY,
  U matBstartX,
  U matBendX,
  U matBstartY,
  U matBendY,
  std::vector<U>& baseCaseDimList,
  blasEngineArgumentPackage_gemm<T>& srcPackage,
  MPI_Comm commWorld,
  int MM_id,
  int TR_id)         // allows for benchmarking to see which version is faster 
{
}
