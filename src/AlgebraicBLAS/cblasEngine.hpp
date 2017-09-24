/* Author: Edward Hutter */


template<typename T>
void cblasHelper::setInfoParameters_gemm(
                                          const blasEngineArgumentPackage<T>& srcPackage,
                                          CBLAS_ORDER& destArg1,
                                          CBLAS_TRANSPOSE& destArg2,
                                          CBLAS_TRANSPOSE& destArg3
                                        )
{
  // Lots of branches :( --> I can use tertiary operator ?, which is much cheaper than an if/else statements

  const blasEngineArgumentPackage_gemm<T>& srcArgs = static_cast<const blasEngineArgumentPackage_gemm<T>&>(srcPackage);
  destArg1 = (srcArgs.order == blasEngineOrder::AblasRowMajor ? CblasRowMajor : CblasColMajor);
  destArg2 = (srcArgs.transposeA == blasEngineTranspose::AblasTrans ? CblasTrans : CblasNoTrans);
  destArg3 = (srcArgs.transposeB == blasEngineTranspose::AblasTrans ? CblasTrans : CblasNoTrans);
}

template<typename T>
void cblasHelper::setInfoParameters_trmm(
                                          const blasEngineArgumentPackage<T>& srcPackage,
                                          CBLAS_ORDER& destArg1,
                                          CBLAS_SIDE& destArg2,
                                          CBLAS_UPLO& destArg3,
                                          CBLAS_TRANSPOSE& destArg4,
                                          CBLAS_DIAG& destArg5
                                        )
{
  const blasEngineArgumentPackage_trmm<T>& srcArgs = static_cast<const blasEngineArgumentPackage_trmm<T>&>(srcPackage);
  destArg1 = (srcArgs.order == blasEngineOrder::AblasRowMajor ? CblasRowMajor : CblasColMajor);
  destArg2 = (srcArgs.side == blasEngineSide::AblasLeft ? CblasLeft : CblasRight);
  destArg3 = (srcArgs.uplo == blasEngineUpLo::AblasLower ? CblasLower : CblasUpper);
  destArg4 = (srcArgs.transposeA == blasEngineTranspose::AblasTrans ? CblasTrans : CblasNoTrans);
  destArg5 = (srcArgs.diag == blasEngineDiag::AblasUnit ? CblasUnit : CblasNonUnit);
}

template<typename T>
void cblasHelper::setInfoParameters_syrk(
                                          const blasEngineArgumentPackage<T>& srcPackage,
                                          CBLAS_ORDER& destArg1,
                                          CBLAS_UPLO& destArg2,
                                          CBLAS_TRANSPOSE& destArg3
                                        )
{
  const blasEngineArgumentPackage_syrk<T>& srcArgs = static_cast<const blasEngineArgumentPackage_syrk<T>&>(srcPackage);
  destArg1 = (srcArgs.order == blasEngineOrder::AblasRowMajor ? CblasRowMajor : CblasColMajor);
  destArg2 = (srcArgs.uplo == blasEngineUpLo::AblasLower ? CblasLower : CblasUpper);
  destArg2 = (srcArgs.transposeA == blasEngineTranspose::AblasTrans ? CblasTrans : CblasNoTrans);
}

template<typename U>
void cblasEngine<float,U>::_gemm(
            float* matrixA,
            float* matrixB,
            float* matrixC,
            U matrixAdimX,
            U matrixAdimY,
            U matrixBdimZ,
            U matrixBdimX,
            U matrixCdimZ,
            U matrixCdimY,
            float alpha,
            float beta,
            U lda,
            U ldb,
            U ldc,
            const blasEngineArgumentPackage<float>& srcPackage
         )
{
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_TRANSPOSE arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_gemm(srcPackage, arg1, arg2, arg3);

  cblas_sgemm(arg1, arg2, arg3, matrixAdimY, matrixCdimZ, matrixAdimX, static_cast<const blasEngineArgumentPackage_gemm<float>&>(srcPackage).alpha,
    matrixA, lda, matrixB, ldb, static_cast<const blasEngineArgumentPackage_gemm<float>&>(srcPackage).beta, matrixC, ldc);
}

template<typename U>
void cblasEngine<float,U>::_trmm(
            float* matrixA,
            float* matrixB,
            U matrixBnumRows,
            U matrixBnumCols,
            float alpha,
            U lda,
            U ldb,
            const blasEngineArgumentPackage<float>& srcPackage
         )
{
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_SIDE arg2;
  CBLAS_UPLO arg3;
  CBLAS_TRANSPOSE arg4;
  CBLAS_DIAG arg5;
  setInfoParameters_trmm(srcPackage, arg1, arg2, arg3, arg4, arg5);

  cblas_strmm(arg1, arg2, arg3, arg4, arg5, matrixBnumRows, matrixBnumCols, static_cast<const blasEngineArgumentPackage_trmm<float>&>(srcPackage).alpha, matrixA,
    lda, matrixB, ldb);
}

template<typename U>
void cblasEngine<float,U>::_syrk(
            float* matrixA,
            float* matrixC,
            U matrixAnumColumns,
            U matrixCnumRows,
            U lda,
            U ldc,
            const blasEngineArgumentPackage<float>& srcPackage
          )
{
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_UPLO arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_syrk(srcPackage, arg1, arg2, arg3);

  cblas_ssyrk(arg1, arg2, arg3, matrixCnumRows, matrixAnumColumns, static_cast<const blasEngineArgumentPackage_syrk<float>&>(srcPackage).alpha, matrixA,
    lda, static_cast<const blasEngineArgumentPackage_syrk<float>&>(srcPackage).beta, matrixC, ldc);
}

template<typename U>
void cblasEngine<double,U>::_gemm(
            double* matrixA,
            double* matrixB,
            double* matrixC,
            U matrixAdimX,
            U matrixAdimY,
            U matrixBdimZ,
            U matrixBdimX,
            U matrixCdimZ,
            U matrixCdimY,
            double alpha,
            double beta,
            U lda,
            U ldb,
            U ldc,
            const blasEngineArgumentPackage<double>& srcPackage
         )
{
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_TRANSPOSE arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_gemm(srcPackage, arg1, arg2, arg3);

  cblas_dgemm(arg1, arg2, arg3, matrixAdimY, matrixCdimZ, matrixAdimX, static_cast<const blasEngineArgumentPackage_gemm<double>&>(srcPackage).alpha,
    matrixA, lda, matrixB, ldb, static_cast<const blasEngineArgumentPackage_gemm<double>&>(srcPackage).beta, matrixC, ldc);
}

template<typename U>
void cblasEngine<double,U>::_trmm(
            double* matrixA,
            double* matrixB,
            U matrixBnumRows,
            U matrixBnumCols,
            double alpha,
            U lda,
            U ldb,
            const blasEngineArgumentPackage<double>& srcPackage
         )
{
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_SIDE arg2;
  CBLAS_UPLO arg3;
  CBLAS_TRANSPOSE arg4;
  CBLAS_DIAG arg5;
  setInfoParameters_trmm(srcPackage, arg1, arg2, arg3, arg4, arg5);

  cblas_dtrmm(arg1, arg2, arg3, arg4, arg5, matrixBnumRows, matrixBnumCols, static_cast<const blasEngineArgumentPackage_trmm<double>&>(srcPackage).alpha, matrixA,
    lda, matrixB, ldb);
}

template<typename U>
void cblasEngine<double,U>::_syrk(
            double* matrixA,
            double* matrixC,
            U matrixAnumColumns,
            U matrixCnumRows,
            U lda,
            U ldc,
            const blasEngineArgumentPackage<double>& srcPackage
          )
{
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_UPLO arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_syrk(srcPackage, arg1, arg2, arg3);

  cblas_dsyrk(arg1, arg2, arg3, matrixCnumRows, matrixAnumColumns, static_cast<const blasEngineArgumentPackage_syrk<double>&>(srcPackage).alpha, matrixA,
    lda, static_cast<const blasEngineArgumentPackage_syrk<double>&>(srcPackage).beta, matrixC, ldc);
}

template<typename U>
void cblasEngine<std::complex<float>,U>::_gemm(
            std::complex<float>* matrixA,
            std::complex<float>* matrixB,
            std::complex<float>* matrixC,
            U matrixAdimX,
            U matrixAdimY,
            U matrixBdimZ,
            U matrixBdimX,
            U matrixCdimZ,
            U matrixCdimY,
            std::complex<float> alpha,
            std::complex<float> beta,
            U lda,
            U ldb,
            U ldc,
            const blasEngineArgumentPackage<std::complex<float>>& srcPackage
         )
{
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_TRANSPOSE arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_gemm(srcPackage, arg1, arg2, arg3);

  cblas_cgemm(arg1, arg2, arg3, matrixAdimY, matrixCdimZ, matrixAdimX, static_cast<const blasEngineArgumentPackage_gemm<std::complex<float>>&>(srcPackage).alpha,
    matrixA, lda, matrixB, ldb, static_cast<const blasEngineArgumentPackage_gemm<std::complex<float>>&>(srcPackage).beta, matrixC, ldc);
}

template<typename U>
void cblasEngine<std::complex<float>,U>::_trmm(
            std::complex<float>* matrixA,
            std::complex<float>* matrixB,
            U matrixBnumRows,
            U matrixBnumCols,
            std::complex<float> alpha,
            U lda,
            U ldb,
            const blasEngineArgumentPackage<std::complex<float>>& srcPackage
         )
{
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_SIDE arg2;
  CBLAS_UPLO arg3;
  CBLAS_TRANSPOSE arg4;
  CBLAS_DIAG arg5;
  setInfoParameters_trmm(srcPackage, arg1, arg2, arg3, arg4, arg5);

  cblas_ctrmm(arg1, arg2, arg3, arg4, arg5, matrixBnumRows, matrixBnumCols, static_cast<const blasEngineArgumentPackage_trmm<std::complex<float>>&>(srcPackage).alpha, matrixA,
    lda, matrixB, ldb);
}

template<typename U>
void cblasEngine<std::complex<float>,U>::_syrk(
            std::complex<float>* matrixA,
            std::complex<float>* matrixC,
            U matrixAnumColumns,
            U matrixCnumRows,
            U lda,
            U ldc,
            const blasEngineArgumentPackage<std::complex<float>>& srcPackage
          )
{
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_UPLO arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_syrk(srcPackage, arg1, arg2, arg3);

  cblas_csyrk(arg1, arg2, arg3, matrixCnumRows, matrixAnumColumns, static_cast<const blasEngineArgumentPackage_syrk<std::complex<float>>&>(srcPackage).alpha, matrixA,
    lda, static_cast<const blasEngineArgumentPackage_syrk<std::complex<float>>&>(srcPackage).beta, matrixC, ldc);
}

template<typename U>
void cblasEngine<std::complex<double>,U>::_gemm(
            std::complex<double>* matrixA,
            std::complex<double>* matrixB,
            std::complex<double>* matrixC,
            U matrixAdimX,
            U matrixAdimY,
            U matrixBdimZ,
            U matrixBdimX,
            U matrixCdimZ,
            U matrixCdimY,
            std::complex<double> alpha,
            std::complex<double> beta,
            U lda,
            U ldb,
            U ldc,
            const blasEngineArgumentPackage<std::complex<double>>& srcPackage
         )
{
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_TRANSPOSE arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_gemm(srcPackage, arg1, arg2, arg3);

  cblas_zgemm(arg1, arg2, arg3, matrixAdimY, matrixCdimZ, matrixAdimX, static_cast<const blasEngineArgumentPackage_gemm<std::complex<double>>&>(srcPackage).alpha,
    matrixA, lda, matrixB, ldb, static_cast<const blasEngineArgumentPackage_gemm<std::complex<double>>&>(srcPackage).beta, matrixC, ldc);
}

template<typename U>
void cblasEngine<std::complex<double>,U>::_trmm(
            std::complex<double>* matrixA,
            std::complex<double>* matrixB,
            U matrixBnumRows,
            U matrixBnumCols,
            std::complex<double> alpha,
            U lda,
            U ldb,
            const blasEngineArgumentPackage<std::complex<double>>& srcPackage
         )
{
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_SIDE arg2;
  CBLAS_UPLO arg3;
  CBLAS_TRANSPOSE arg4;
  CBLAS_DIAG arg5;
  setInfoParameters_trmm(srcPackage, arg1, arg2, arg3, arg4, arg5);

  cblas_ztrmm(arg1, arg2, arg3, arg4, arg5, matrixBnumRows, matrixBnumCols, static_cast<const blasEngineArgumentPackage_trmm<std::complex<double>>&>(srcPackage).alpha, matrixA,
    lda, matrixB, ldb);
}

template<typename U>
void cblasEngine<std::complex<double>,U>::_syrk(
            std::complex<double>* matrixA,
            std::complex<double>* matrixC,
            U matrixAnumColumns,
            U matrixCnumRows,
            U lda,
            U ldc,
            const blasEngineArgumentPackage<std::complex<double>>& srcPackage
          )
{
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_UPLO arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_syrk(srcPackage, arg1, arg2, arg3);

  cblas_zsyrk(arg1, arg2, arg3, matrixCnumRows, matrixAnumColumns, static_cast<const blasEngineArgumentPackage_syrk<std::complex<double>>&>(srcPackage).alpha, matrixA,
    lda, static_cast<const blasEngineArgumentPackage_syrk<std::complex<double>>&>(srcPackage).beta, matrixC, ldc);
}
