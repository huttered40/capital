/* Author: Edward Hutter */


template<typename T>
void cblasHelper::setInfoParameters_gemm(
                                          const blasEngineArgumentPackage_gemm<T>& srcPackage,
                                          CBLAS_ORDER& destArg1,
                                          CBLAS_TRANSPOSE& destArg2,
                                          CBLAS_TRANSPOSE& destArg3
                                        )
{
  // Lots of branches :( --> I can use tertiary operator ?, which is much cheaper than an if/else statements

  destArg1 = (srcPackage.order == blasEngineOrder::AblasRowMajor ? CblasRowMajor : CblasColMajor);
  destArg2 = (srcPackage.transposeA == blasEngineTranspose::AblasTrans ? CblasTrans : CblasNoTrans);
  destArg3 = (srcPackage.transposeB == blasEngineTranspose::AblasTrans ? CblasTrans : CblasNoTrans);
}

template<typename T>
void cblasHelper::setInfoParameters_trmm(
                                          const blasEngineArgumentPackage_trmm<T>& srcPackage,
                                          CBLAS_ORDER& destArg1,
                                          CBLAS_SIDE& destArg2,
                                          CBLAS_UPLO& destArg3,
                                          CBLAS_TRANSPOSE& destArg4,
                                          CBLAS_DIAG& destArg5
                                        )
{
  destArg1 = (srcPackage.order == blasEngineOrder::AblasRowMajor ? CblasRowMajor : CblasColMajor);
  destArg2 = (srcPackage.side == blasEngineSide::AblasLeft ? CblasLeft : CblasRight);
  destArg3 = (srcPackage.uplo == blasEngineUpLo::AblasLower ? CblasLower : CblasUpper);
  destArg4 = (srcPackage.transposeA == blasEngineTranspose::AblasTrans ? CblasTrans : CblasNoTrans);
  destArg5 = (srcPackage.diag == blasEngineDiag::AblasUnit ? CblasUnit : CblasNonUnit);
}

template<typename T>
void cblasHelper::setInfoParameters_syrk(
                                          const blasEngineArgumentPackage_syrk<T>& srcPackage,
                                          CBLAS_ORDER& destArg1,
                                          CBLAS_UPLO& destArg2,
                                          CBLAS_TRANSPOSE& destArg3
                                        )
{
  destArg1 = (srcPackage.order == blasEngineOrder::AblasRowMajor ? CblasRowMajor : CblasColMajor);
  destArg2 = (srcPackage.uplo == blasEngineUpLo::AblasLower ? CblasLower : CblasUpper);
  destArg3 = (srcPackage.transposeA == blasEngineTranspose::AblasTrans ? CblasTrans : CblasNoTrans);
}

template<typename U>
void cblasEngine<float,U>::_gemm(
            float* matrixA,
            float* matrixB,
            float* matrixC,
            U m,
            U n,
            U k,
            U lda,
            U ldb,
            U ldc,
            const blasEngineArgumentPackage_gemm<float>& srcPackage
         )
{
  TAU_FSTART(gemm);
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_TRANSPOSE arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_gemm(srcPackage, arg1, arg2, arg3);

  cblas_sgemm(arg1, arg2, arg3, m, n, k, srcPackage.alpha,
    matrixA, lda, matrixB, ldb, srcPackage.beta, matrixC, ldc);
  TAU_FSTOP(gemm);
}

template<typename U>
void cblasEngine<float,U>::_trmm(
            float* matrixA,
            float* matrixB,
            U m,
            U n,
            U lda,
            U ldb,
            const blasEngineArgumentPackage_trmm<float>& srcPackage
         )
{
  TAU_FSTART(trmm);
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_SIDE arg2;
  CBLAS_UPLO arg3;
  CBLAS_TRANSPOSE arg4;
  CBLAS_DIAG arg5;
  setInfoParameters_trmm(srcPackage, arg1, arg2, arg3, arg4, arg5);

  cblas_strmm(arg1, arg2, arg3, arg4, arg5, m, n, srcPackage.alpha, matrixA,
    lda, matrixB, ldb);
  TAU_FSTOP(trmm);
}

template<typename U>
void cblasEngine<float,U>::_syrk(
            float* matrixA,
            float* matrixC,
            U n,
            U k,
            U lda,
            U ldc,
            const blasEngineArgumentPackage_syrk<float>& srcPackage
          )
{
  TAU_FSTART(syrk);
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_UPLO arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_syrk(srcPackage, arg1, arg2, arg3);

  cblas_ssyrk(arg1, arg2, arg3, n, k, srcPackage.alpha, matrixA,
    lda, srcPackage.beta, matrixC, ldc);
  TAU_FSTOP(syrk);
}

template<typename U>
void cblasEngine<double,U>::_gemm(
            double* matrixA,
            double* matrixB,
            double* matrixC,
            U m,
            U n,
            U k,
            U lda,
            U ldb,
            U ldc,
            const blasEngineArgumentPackage_gemm<double>& srcPackage
         )
{
  TAU_FSTART(gemm);
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_TRANSPOSE arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_gemm(srcPackage, arg1, arg2, arg3);

  cblas_dgemm(arg1, arg2, arg3, m, n, k, srcPackage.alpha,
    matrixA, lda, matrixB, ldb, srcPackage.beta, matrixC, ldc);
  TAU_FSTOP(gemm);
}

template<typename U>
void cblasEngine<double,U>::_trmm(
            double* matrixA,
            double* matrixB,
            U m,
            U n,
            U lda,
            U ldb,
            const blasEngineArgumentPackage_trmm<double>& srcPackage
         )
{
  TAU_FSTART(trmm);
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_SIDE arg2;
  CBLAS_UPLO arg3;
  CBLAS_TRANSPOSE arg4;
  CBLAS_DIAG arg5;
  setInfoParameters_trmm(srcPackage, arg1, arg2, arg3, arg4, arg5);

  cblas_dtrmm(arg1, arg2, arg3, arg4, arg5, m, n, srcPackage.alpha, matrixA,
    lda, matrixB, ldb);
  TAU_FSTOP(trmm);
}

template<typename U>
void cblasEngine<double,U>::_syrk(
            double* matrixA,
            double* matrixC,
            U n,
            U k,
            U lda,
            U ldc,
            const blasEngineArgumentPackage_syrk<double>& srcPackage
          )
{
  TAU_FSTART(syrk);
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_UPLO arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_syrk(srcPackage, arg1, arg2, arg3);

  cblas_dsyrk(arg1, arg2, arg3, n, k, srcPackage.alpha, matrixA,
    lda, srcPackage.beta, matrixC, ldc);
  TAU_FSTOP(syrk);
}

template<typename U>
void cblasEngine<std::complex<float>,U>::_gemm(
            std::complex<float>* matrixA,
            std::complex<float>* matrixB,
            std::complex<float>* matrixC,
            U m,
            U n,
            U k,
            U lda,
            U ldb,
            U ldc,
            const blasEngineArgumentPackage_gemm<std::complex<float>>& srcPackage
         )
{
  TAU_FSTART(gemm);
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_TRANSPOSE arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_gemm(srcPackage, arg1, arg2, arg3);

  cblas_cgemm(arg1, arg2, arg3, m, n, k, srcPackage.alpha,
    matrixA, lda, matrixB, ldb, srcPackage.beta, matrixC, ldc);
  TAU_FSTOP(gemm);
}

template<typename U>
void cblasEngine<std::complex<float>,U>::_trmm(
            std::complex<float>* matrixA,
            std::complex<float>* matrixB,
            U m,
            U n,
            U lda,
            U ldb,
            const blasEngineArgumentPackage_trmm<std::complex<float>>& srcPackage
         )
{
  TAU_FSTART(trmm);
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_SIDE arg2;
  CBLAS_UPLO arg3;
  CBLAS_TRANSPOSE arg4;
  CBLAS_DIAG arg5;
  setInfoParameters_trmm(srcPackage, arg1, arg2, arg3, arg4, arg5);

  cblas_ctrmm(arg1, arg2, arg3, arg4, arg5, m, n, srcPackage.alpha, matrixA,
    lda, matrixB, ldb);
  TAU_FSTOP(trmm);
}

template<typename U>
void cblasEngine<std::complex<float>,U>::_syrk(
            std::complex<float>* matrixA,
            std::complex<float>* matrixC,
            U n,
            U k,
            U lda,
            U ldc,
            const blasEngineArgumentPackage_syrk<std::complex<float>>& srcPackage
          )
{
  TAU_FSTART(syrk);
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_UPLO arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_syrk(srcPackage, arg1, arg2, arg3);

  cblas_csyrk(arg1, arg2, arg3, n, k, srcPackage.alpha, matrixA,
    lda, srcPackage.beta, matrixC, ldc);
  TAU_FSTOP(syrk);
}

template<typename U>
void cblasEngine<std::complex<double>,U>::_gemm(
            std::complex<double>* matrixA,
            std::complex<double>* matrixB,
            std::complex<double>* matrixC,
            U m,
            U n,
            U k,
            U lda,
            U ldb,
            U ldc,
            const blasEngineArgumentPackage_gemm<std::complex<double>>& srcPackage
         )
{
  TAU_FSTART(gemm);
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_TRANSPOSE arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_gemm(srcPackage, arg1, arg2, arg3);

  cblas_zgemm(arg1, arg2, arg3, m, n, k, srcPackage.alpha,
    matrixA, lda, matrixB, ldb, srcPackage.beta, matrixC, ldc);
  TAU_FSTOP(gemm);
}

template<typename U>
void cblasEngine<std::complex<double>,U>::_trmm(
            std::complex<double>* matrixA,
            std::complex<double>* matrixB,
            U m,
            U n,
            U lda,
            U ldb,
            const blasEngineArgumentPackage_trmm<std::complex<double>>& srcPackage
         )
{
  TAU_FSTART(trmm);
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_SIDE arg2;
  CBLAS_UPLO arg3;
  CBLAS_TRANSPOSE arg4;
  CBLAS_DIAG arg5;
  setInfoParameters_trmm(srcPackage, arg1, arg2, arg3, arg4, arg5);

  cblas_ztrmm(arg1, arg2, arg3, arg4, arg5, m, n, srcPackage.alpha, matrixA,
    lda, matrixB, ldb);
  TAU_FSTOP(trmm);
}

template<typename U>
void cblasEngine<std::complex<double>,U>::_syrk(
            std::complex<double>* matrixA,
            std::complex<double>* matrixC,
            U n,
            U k,
            U lda,
            U ldc,
            const blasEngineArgumentPackage_syrk<std::complex<double>>& srcPackage
          )
{
  TAU_FSTART(syrk);
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_UPLO arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_syrk(srcPackage, arg1, arg2, arg3);

  cblas_zsyrk(arg1, arg2, arg3, n, k, srcPackage.alpha, matrixA,
    lda, srcPackage.beta, matrixC, ldc);
  TAU_FSTOP(syrk);
}
