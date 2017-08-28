/* Author: Edward Hutter */

#ifndef BLASENGINE_H_
#define BLASENGINE_H_


// System includes
#include <complex>
#include <cblas.h>

// Local includes

// Goal: Have a BLAS Policy with the particular BLAS implementation as one of the Policy classes
//       This will allow for easier switching when needing alternate BLAS implementations

// BLAS libraries typically work on 4 different datatypes
//   single precision real(float)
//   double precision real (double)
//   single precision complex (std::complex<float>)
//   double precision complex (std::complex<double>)
// So it'd be best if we used partial template specialization to specialize the blasEngine Policy classes
//   so as to only define the routines with T = one of the 4 types above


class cblasHelper
{
public:
  cblasHelper() = delete;
  cblasHelper(const cblasHelper& rhs) = delete;
  cblasHelper(cblasHelper&& rhs) = delete;
  cblasHelper& operator=(const cblasHelper& rhs) = delete;
  cblasHelper& operator=(cblasHelper&& rhs) = delete;

// Make these methods protected so that only the derived classes can access them.
protected:
  static void setInfoParameters_gemm(int info, CBLAS_ORDER& arg1, CBLAS_TRANSPOSE& arg2, CBLAS_TRANSPOSE& arg3);
  static void setInfoParameters_trmm(int info, CBLAS_ORDER& arg1, CBLAS_SIDE& arg2, CBLAS_UPLO& arg3, CBLAS_TRANSPOSE& arg4, CBLAS_DIAG& arg5);
};


// Declare this fully templated "base" class but do not define it. This prevents users from using this class, but
//   allows partially specialized template classes to specialize it.
template<typename T, typename U>
class cblasEngine; 

template<typename U>
class cblasEngine<float,U> : public cblasHelper
{
  // Lets prevent any instances of this class from being created.
public:
  cblasEngine() = delete;
  cblasEngine(const cblasEngine& rhs) = delete;
  cblasEngine(cblasEngine&& rhs) = delete;
  cblasEngine<float,U>& operator=(const cblasEngine& rhs) = delete;
  cblasEngine<float,U>& operator=(cblasEngine&& rhs) = delete;
  ~cblasEngine() = delete;

  // Engine methods
  static void _gemm(float* matrixA, float* matrixB, float* matrixC, U matrixAdimX, U matrixAdimY, U matrixBdimX, U matrixBdimZ, U matrixCdimY, U matrixCdimZ,
                      float alpha, float beta, U lda, U ldb, U ldc, int info);
  static void _trmm(float* matrixA, float* matrixB, U matrixBnumRows, U matrixBnumCols, float alpha, U lda, U ldb, int info);

};

template<typename U>
class cblasEngine<double,U> : public cblasHelper
{
  // Lets prevent any instances of this class from being created.
public:
  cblasEngine() = delete;
  cblasEngine(const cblasEngine& rhs) = delete;
  cblasEngine(cblasEngine&& rhs) = delete;
  cblasEngine<double,U>& operator=(const cblasEngine& rhs) = delete;
  cblasEngine<double,U>& operator=(cblasEngine&& rhs) = delete;
  ~cblasEngine() = delete;

  // Engine methods
  static void _gemm(double* matrixA, double* matrixB, double* matrixC, U matrixAdimX, U matrixAdimY, U matrixBdimX, U matrixBdimZ, U matrixCdimY, U matrixCdimZ,
                      double alpha, double beta, U lda, U ldb, U ldc, int info);
  static void _trmm(double* matrixA, double* matrixB, U matrixBnumRows, U matrixBnumCols, double alpha, U lda, U ldb, int info);

};

template<typename U>
class cblasEngine<std::complex<float>,U> : public cblasHelper
{
  // Lets prevent any instances of this class from being created.
public:
  cblasEngine() = delete;
  cblasEngine(const cblasEngine& rhs) = delete;
  cblasEngine(cblasEngine&& rhs) = delete;
  cblasEngine<std::complex<float>,U>& operator=(const cblasEngine& rhs) = delete;
  cblasEngine<std::complex<float>,U>& operator=(cblasEngine&& rhs) = delete;
  ~cblasEngine() = delete;

  // Engine methods
  static void _gemm(std::complex<float>* matrixA, std::complex<float>* matrixB, std::complex<float>* matrixC, U matrixAdimX, U matrixAdimY, U matrixBdimX,
                     U matrixBdimZ, U matrixCdimY, U matrixCdimZ, std::complex<float> alpha, std::complex<float> beta, U lda, U ldb, U ldc, int info);
  static void _trmm(std::complex<float>* matrixA, std::complex<float>* matrixB, U matrixBnumRows, U matrixBnumCols, std::complex<float> alpha, U lda, U ldb, int info);

};

template<typename U>
class cblasEngine<std::complex<double>,U> : public cblasHelper
{
  // Lets prevent any instances of this class from being created.
public:
  cblasEngine() = delete;
  cblasEngine(const cblasEngine& rhs) = delete;
  cblasEngine(cblasEngine&& rhs) = delete;
  cblasEngine<std::complex<double>,U>& operator=(const cblasEngine& rhs) = delete;
  cblasEngine<std::complex<double>,U>& operator=(cblasEngine&& rhs) = delete;
  ~cblasEngine() = delete;

  // Engine methods
  static void _gemm(std::complex<double>* matrixA, std::complex<double>* matrixB, std::complex<double>* matrixC, U matrixAdimX, U matrixAdimY, U matrixBdimX,
                     U matrixBdimZ, U matrixCdimY, U matrixCdimZ, std::complex<double> alpha, std::complex<double> beta, U lda, U ldb, U ldc, int info);
  static void _trmm(std::complex<double>* matrixA, std::complex<double>* matrixB, U matrixBnumRows, U matrixBnumCols, std::complex<double> alpha, U lda, U ldb, int info);

};

#include "blasEngine.hpp"

#endif /* BLASENGINE_H_ */
