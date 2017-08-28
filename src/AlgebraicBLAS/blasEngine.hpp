/* Author: Edward Hutter */


void cblasHelper::setInfoParameters_gemm(
                                           int info,
                                           CBLAS_ORDER& arg1,
                                           CBLAS_TRANSPOSE& arg2,
                                           CBLAS_TRANSPOSE& arg3
                                         )
{
  if ((info&0x1))
  {
    arg1 = CblasRowMajor;
  }
  else
  {
    arg1 = CblasColMajor;
  }
  info >>= 1;
  if ((info&0x1))
  {
    arg2 = CblasTrans;
  }
  else
  {
    arg2 = CblasNoTrans;
  }
  info >>= 1;
  if ((info&0x1))
  {
    arg2 = CblasTrans;
  }
  else
  {
    arg2 = CblasNoTrans;
  }
}

void cblasHelper::setInfoParameters_trmm(
                                          int info,
                                          CBLAS_ORDER& arg1,
                                          CBLAS_SIDE& arg2,
                                          CBLAS_UPLO& arg3,
                                          CBLAS_TRANSPOSE& arg4,
                                          CBLAS_DIAG& arg5
                                        )
{
  if ((info&0x1))
  {
    arg1 = CblasRowMajor;
  }
  else
  {
    arg1 = CblasColMajor;
  }
  info >>= 1;
  if ((info&0x1))
  {
    arg2 = CblasLeft;
  }
  else
  {
    arg2 = CblasRight;
  }
  info >>= 1;
  if ((info&0x1))
  {
    arg3 = CblasUpper;
  }
  else
  {
    arg3 = CblasLower;
  }
  info >>= 1;
  if ((info&0x1))
  {
    arg4 = CblasTrans;
  }
  else
  {
    arg4 = CblasNoTrans;
  }
  info >>= 1;
  if ((info&0x1))
  {
    arg5 = CblasUnit;
  }
  else
  {
    arg5 = CblasNonUnit;
  }
}

template<typename U>
void cblasEngine<float,U>::_gemm(
            float* matrixA,
            float* matrixB,
            float* matrixC,
            U matrixAdimX,
            U matrixAdimY,
            U matrixBdimX,
            U matrixBdimZ,
            U matrixCdimY,
            U matrixCdimZ,
            float alpha,
            float beta,
            U lda,
            U ldb,
            U ldc,
            int info 
         )
{
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_TRANSPOSE arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_gemm(info, arg1, arg2, arg3);

  cblas_sgemm(arg1, arg2, arg3, matrixAdimY, matrixCdimZ, matrixAdimX, alpha,
    matrixA, lda, matrixB, ldb, beta, matrixC, ldc);
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
            int info
         )
{
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_SIDE arg2;
  CBLAS_UPLO arg3;
  CBLAS_TRANSPOSE arg4;
  CBLAS_DIAG arg5;
  setInfoParameters_trmm(info, arg1, arg2, arg3, arg4, arg5);

  cblas_strmm(arg1, arg2, arg3, arg4, arg5, matrixBnumRows, matrixBnumCols, alpha, matrixA,
    lda, matrixB, ldb);
}

template<typename U>
void cblasEngine<double,U>::_gemm(
            double* matrixA,
            double* matrixB,
            double* matrixC,
            U matrixAdimX,
            U matrixAdimY,
            U matrixBdimX,
            U matrixBdimZ,
            U matrixCdimY,
            U matrixCdimZ,
            double alpha,
            double beta,
            U lda,
            U ldb,
            U ldc,
            int info 
         )
{
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_TRANSPOSE arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_gemm(info, arg1, arg2, arg3);

  cblas_dgemm(arg1, arg2, arg3, matrixAdimY, matrixCdimZ, matrixAdimX, alpha,
    matrixA, lda, matrixB, ldb, beta, matrixC, ldc);
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
            int info
         )
{
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_SIDE arg2;
  CBLAS_UPLO arg3;
  CBLAS_TRANSPOSE arg4;
  CBLAS_DIAG arg5;
  setInfoParameters_trmm(info, arg1, arg2, arg3, arg4, arg5);

  cblas_dtrmm(arg1, arg2, arg3, arg4, arg5, matrixBnumRows, matrixBnumCols, alpha, matrixA,
    lda, matrixB, ldb);
}

template<typename U>
void cblasEngine<std::complex<float>,U>::_gemm(
            std::complex<float>* matrixA,
            std::complex<float>* matrixB,
            std::complex<float>* matrixC,
            U matrixAdimX,
            U matrixAdimY,
            U matrixBdimX,
            U matrixBdimZ,
            U matrixCdimY,
            U matrixCdimZ,
            std::complex<float> alpha,
            std::complex<float> beta,
            U lda,
            U ldb,
            U ldc,
            int info 
         )
{
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_TRANSPOSE arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_gemm(info, arg1, arg2, arg3);

  cblas_cgemm(arg1, arg2, arg3, matrixAdimY, matrixCdimZ, matrixAdimX, alpha,
    matrixA, lda, matrixB, ldb, beta, matrixC, ldc);
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
            int info
         )
{
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_SIDE arg2;
  CBLAS_UPLO arg3;
  CBLAS_TRANSPOSE arg4;
  CBLAS_DIAG arg5;
  setInfoParameters_trmm(info, arg1, arg2, arg3, arg4, arg5);

  cblas_ctrmm(arg1, arg2, arg3, arg4, arg5, matrixBnumRows, matrixBnumCols, alpha, matrixA,
    lda, matrixB, ldb);
}

template<typename U>
void cblasEngine<std::complex<double>,U>::_gemm(
            std::complex<double>* matrixA,
            std::complex<double>* matrixB,
            std::complex<double>* matrixC,
            U matrixAdimX,
            U matrixAdimY,
            U matrixBdimX,
            U matrixBdimZ,
            U matrixCdimY,
            U matrixCdimZ,
            std::complex<double> alpha,
            std::complex<double> beta,
            U lda,
            U ldb,
            U ldc,
            int info 
         )
{
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_TRANSPOSE arg2;
  CBLAS_TRANSPOSE arg3;
  setInfoParameters_gemm(info, arg1, arg2, arg3);

  cblas_zgemm(arg1, arg2, arg3, matrixAdimY, matrixCdimZ, matrixAdimX, alpha,
    matrixA, lda, matrixB, ldb, beta, matrixC, ldc);
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
            int info
         )
{
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_SIDE arg2;
  CBLAS_UPLO arg3;
  CBLAS_TRANSPOSE arg4;
  CBLAS_DIAG arg5;
  setInfoParameters_trmm(info, arg1, arg2, arg3, arg4, arg5);

  cblas_ztrmm(arg1, arg2, arg3, arg4, arg5, matrixBnumRows, matrixBnumCols, alpha, matrixA,
    lda, matrixB, ldb);
}
