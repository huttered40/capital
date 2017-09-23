/* Author: Edward Hutter */

#ifndef CBLASENGINE_H_
#define CBLASENGINE_H_


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
  template<typename T>
  static void setInfoParameters_gemm(const blasEngineArgumentPackage<T>& srcPackage, CBLAS_ORDER& destArg1, CBLAS_TRANSPOSE& destArg2, CBLAS_TRANSPOSE& destArg3);

  template<typename T>
  static void setInfoParameters_trmm(const blasEngineArgumentPackage<T>& srcPackage, CBLAS_ORDER& destArg1, CBLAS_SIDE& destArg2, CBLAS_UPLO& destArg3, CBLAS_TRANSPOSE& destArg4, CBLAS_DIAG& destArg5);
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
  static void _gemm(float* matrixA, float* matrixB, float* matrixC, U matrixAdimX, U matrixAdimY, U matrixBdimZ, U matrixBdimX, U matrixCdimZ, U matrixCdimY,
                      float alpha, float beta, U lda, U ldb, U ldc, const blasEngineArgumentPackage<float>& srcPackage);
  static void _trmm(float* matrixA, float* matrixB, U matrixBnumRows, U matrixBnumCols, float alpha, U lda, U ldb, const blasEngineArgumentPackage<float>& srcPackage);

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
  static void _gemm(double* matrixA, double* matrixB, double* matrixC, U matrixAdimX, U matrixAdimY, U matrixBdimZ, U matrixBdimX, U matrixCdimZ, U matrixCdimY,
                      double alpha, double beta, U lda, U ldb, U ldc, const blasEngineArgumentPackage<double>& srcPackage);
  static void _trmm(double* matrixA, double* matrixB, U matrixBnumRows, U matrixBnumCols, double alpha, U lda, U ldb, const blasEngineArgumentPackage<double>& srcPackage);

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
  static void _gemm(std::complex<float>* matrixA, std::complex<float>* matrixB, std::complex<float>* matrixC, U matrixAdimX, U matrixAdimY, U matrixBdimZ,
                     U matrixBdimX, U matrixCdimZ, U matrixCdimY, std::complex<float> alpha, std::complex<float> beta, U lda, U ldb, U ldc,
                     const blasEngineArgumentPackage<std::complex<float>>& srcPackage);
  static void _trmm(std::complex<float>* matrixA, std::complex<float>* matrixB, U matrixBnumRows, U matrixBnumCols, std::complex<float> alpha, U lda, U ldb,
                      const blasEngineArgumentPackage<std::complex<float>>& srcPackage);

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
  static void _gemm(std::complex<double>* matrixA, std::complex<double>* matrixB, std::complex<double>* matrixC, U matrixAdimX, U matrixAdimY, U matrixBdimZ,
                     U matrixBdimX, U matrixCdimZ, U matrixCdimY, std::complex<double> alpha, std::complex<double> beta, U lda, U ldb, U ldc,
                     const blasEngineArgumentPackage<std::complex<double>>& srcPackage);
  static void _trmm(std::complex<double>* matrixA, std::complex<double>* matrixB, U matrixBnumRows, U matrixBnumCols, std::complex<double> alpha, U lda, U ldb,
                      const blasEngineArgumentPackage<std::complex<double>>& srcPackage);

};

#include "cblasEngine.hpp"

#endif /* CBLASENGINE_H_ */