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

template<typename T, typename U>
void engine::_gemm(
            T* matrixA,
            T* matrixB,
            T* matrixC,
            U m,
            U n,
            U k,
            U lda,
            U ldb,
            U ldc,
            const ArgPack_gemm<T>& srcPackage
         ){
  TAU_FSTART(gemm);
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_TRANSPOSE arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_gemm(srcPackage, arg1, arg2, arg3);

  auto _gemm_ = GetGEMMroutine(BType<T>());
  _gemm_(arg1, arg2, arg3, m, n, k, srcPackage.alpha,
    matrixA, lda, matrixB, ldb, srcPackage.beta, matrixC, ldc);
  TAU_FSTOP(gemm);
}

template<typename T, typename U>
void engine::_trmm(
            T* matrixA,
            T* matrixB,
            U m,
            U n,
            U lda,
            U ldb,
            const ArgPack_trmm<T>& srcPackage
         ){
  TAU_FSTART(trmm);
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_SIDE arg2;
  CBLAS_UPLO arg3;
  CBLAS_TRANSPOSE arg4;
  CBLAS_DIAG arg5;
  setInfoParameters_trmm(srcPackage, arg1, arg2, arg3, arg4, arg5);

  auto _trmm_ = GetTRMMroutine(BType<T>());
  _trmm_(arg1, arg2, arg3, arg4, arg5, m, n, srcPackage.alpha, matrixA,
    lda, matrixB, ldb);
  TAU_FSTOP(trmm);
}

template<typename T, typename U>
void engine::_syrk(
            T* matrixA,
            T* matrixC,
            U n,
            U k,
            U lda,
            U ldc,
            const ArgPack_syrk<T>& srcPackage
          ){
  TAU_FSTART(syrk);
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_UPLO arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_syrk(srcPackage, arg1, arg2, arg3);

  auto _syrk_ = GetSYRKroutine(BType<T>());
  _syrk_(arg1, arg2, arg3, n, k, srcPackage.alpha, matrixA,
    lda, srcPackage.beta, matrixC, ldc);
  TAU_FSTOP(syrk);
}
}
