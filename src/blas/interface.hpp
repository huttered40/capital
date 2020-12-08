/* Author: Edward Hutter */

namespace blas{
template<typename T>
void helper::setInfoParameters_gemm(const ArgPack_gemm<T>& srcPackage,
                                    CBLAS_ORDER& destArg1,
                                    CBLAS_TRANSPOSE& destArg2,
                                    CBLAS_TRANSPOSE& destArg3
                                   ){
  // Lots of branches :( --> I can use tertiary operator ?, which is much cheaper than an if/else statements

  destArg1 = (srcPackage.order == Order::AblasRowMajor ? CblasRowMajor : CblasColMajor);
  destArg2 = (srcPackage.transposeA == Transpose::AblasTrans ? CblasTrans : CblasNoTrans);
  destArg3 = (srcPackage.transposeB == Transpose::AblasTrans ? CblasTrans : CblasNoTrans);
}

template<typename T>
void helper::setInfoParameters_trmm(const ArgPack_trmm<T>& srcPackage,
                                    CBLAS_ORDER& destArg1,
                                    CBLAS_SIDE& destArg2,
                                    CBLAS_UPLO& destArg3,
                                    CBLAS_TRANSPOSE& destArg4,
                                    CBLAS_DIAG& destArg5
                                   ){
  destArg1 = (srcPackage.order == Order::AblasRowMajor ? CblasRowMajor : CblasColMajor);
  destArg2 = (srcPackage.side == Side::AblasLeft ? CblasLeft : CblasRight);
  destArg3 = (srcPackage.uplo == UpLo::AblasLower ? CblasLower : CblasUpper);
  destArg4 = (srcPackage.transposeA == Transpose::AblasTrans ? CblasTrans : CblasNoTrans);
  destArg5 = (srcPackage.diag == Diag::AblasUnit ? CblasUnit : CblasNonUnit);
}

template<typename T>
void helper::setInfoParameters_syrk(const ArgPack_syrk<T>& srcPackage,
                                    CBLAS_ORDER& destArg1,
                                    CBLAS_UPLO& destArg2,
                                    CBLAS_TRANSPOSE& destArg3
                                   ){
  destArg1 = (srcPackage.order == Order::AblasRowMajor ? CblasRowMajor : CblasColMajor);
  destArg2 = (srcPackage.uplo == UpLo::AblasLower ? CblasLower : CblasUpper);
  destArg3 = (srcPackage.transposeA == Transpose::AblasTrans ? CblasTrans : CblasNoTrans);
}

template<>
void engine::_gemm(double* matrixA, double* matrixB, double* matrixC, int64_t m, int64_t n, int64_t k, int64_t lda, int64_t ldb, int64_t ldc, const ArgPack_gemm<double>& srcPackage){
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_TRANSPOSE arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_gemm(srcPackage, arg1, arg2, arg3);

#ifdef FUNCTION_SYMBOLS
CRITTER_START(gemm);
#endif
  cblas_dgemm(arg1, arg2, arg3, m, n, k, srcPackage.alpha,
    matrixA, lda, matrixB, ldb, srcPackage.beta, matrixC, ldc);
#ifdef FUNCTION_SYMBOLS
CRITTER_STOP(gemm);
#endif
}

template<>
void engine::_trmm(double* matrixA, double* matrixB, int64_t m, int64_t n, int64_t lda, int64_t ldb, const ArgPack_trmm<double>& srcPackage){
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_SIDE arg2;
  CBLAS_UPLO arg3;
  CBLAS_TRANSPOSE arg4;
  CBLAS_DIAG arg5;
  setInfoParameters_trmm(srcPackage, arg1, arg2, arg3, arg4, arg5);

#ifdef FUNCTION_SYMBOLS
CRITTER_START(trmm);
#endif
  cblas_dtrmm(arg1, arg2, arg3, arg4, arg5, m, n, srcPackage.alpha, matrixA,
    lda, matrixB, ldb);
#ifdef FUNCTION_SYMBOLS
CRITTER_STOP(trmm);
#endif
}

template<>
void engine::_syrk(double* matrixA, double* matrixC, int64_t n, int64_t k, int64_t lda, int64_t ldc, const ArgPack_syrk<double>& srcPackage){
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_UPLO arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_syrk(srcPackage, arg1, arg2, arg3);

#ifdef FUNCTION_SYMBOLS
CRITTER_START(syrk);
#endif
  cblas_dsyrk(arg1, arg2, arg3, n, k, srcPackage.alpha, matrixA,
    lda, srcPackage.beta, matrixC, ldc);
#ifdef FUNCTION_SYMBOLS
CRITTER_STOP(syrk);
#endif
}
}
