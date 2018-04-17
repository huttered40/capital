/* Author: Edward Hutter */

#ifndef CBLASENGINE_H_
#define CBLASENGINE_H_


// System includes
#include <complex>

#ifdef PORTER
#include <cblas.h>
#endif

#ifdef THETA
#include "mkl.h"
#endif

#ifdef STAMPEDE2
#include "mkl.h"
#endif

#ifdef BGQ
#include "/soft/libraries/alcf/current/xl/CBLAS/include/cblas.h"
#endif

// Local includes
#include "./../Util/shared.h"
#include "./../Timer/CTFtimer.h"

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
  static void setInfoParameters_gemm(const blasEngineArgumentPackage_gemm<T>& srcPackage, CBLAS_ORDER& destArg1, CBLAS_TRANSPOSE& destArg2, CBLAS_TRANSPOSE& destArg3);

  template<typename T>
  static void setInfoParameters_trmm(const blasEngineArgumentPackage_trmm<T>& srcPackage, CBLAS_ORDER& destArg1, CBLAS_SIDE& destArg2, CBLAS_UPLO& destArg3, CBLAS_TRANSPOSE& destArg4, CBLAS_DIAG& destArg5);

  template<typename T>
  static void setInfoParameters_syrk(const blasEngineArgumentPackage_syrk<T>& srcPackage, CBLAS_ORDER& destArg1, CBLAS_UPLO& destArg2, CBLAS_TRANSPOSE& destArg3);
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
  static void _gemm(float* matrixA, float* matrixB, float* matrixC, U m, U n, U k,
                      U lda, U ldb, U ldc, const blasEngineArgumentPackage_gemm<float>& srcPackage);
  static void _trmm(float* matrixA, float* matrixB, U m, U n, U lda, U ldb, const blasEngineArgumentPackage_trmm<float>& srcPackage);
  static void _syrk(float* matrixA, float* matrixC, U n, U k, U lda, U ldc, const blasEngineArgumentPackage_syrk<float>& srcPackage);
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
  static void _gemm(double* matrixA, double* matrixB, double* matrixC, U m, U n, U k,
                      U lda, U ldb, U ldc, const blasEngineArgumentPackage_gemm<double>& srcPackage);
  static void _trmm(double* matrixA, double* matrixB, U m, U n, U lda, U ldb, const blasEngineArgumentPackage_trmm<double>& srcPackage);
  static void _syrk(double* matrixA, double* matrixC, U n, U k, U lda, U ldc, const blasEngineArgumentPackage_syrk<double>& srcPackage);
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
  static void _gemm(std::complex<float>* matrixA, std::complex<float>* matrixB, std::complex<float>* matrixC, U m, U n, U k, U lda, U ldb, U ldc,
                     const blasEngineArgumentPackage_gemm<std::complex<float>>& srcPackage);
  static void _trmm(std::complex<float>* matrixA, std::complex<float>* matrixB, U m, U n, U lda, U ldb,
                      const blasEngineArgumentPackage_trmm<std::complex<float>>& srcPackage);
  static void _syrk(std::complex<float>* matrixA, std::complex<float>* matrixC, U n, U k, U lda, U ldc,
    const blasEngineArgumentPackage_syrk<std::complex<float>>& srcPackage);
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
  static void _gemm(std::complex<double>* matrixA, std::complex<double>* matrixB, std::complex<double>* matrixC, U m, U n, U k, U lda, U ldb, U ldc,
                     const blasEngineArgumentPackage_gemm<std::complex<double>>& srcPackage);
  static void _trmm(std::complex<double>* matrixA, std::complex<double>* matrixB, U m, U n, U lda, U ldb,
                      const blasEngineArgumentPackage_trmm<std::complex<double>>& srcPackage);
  static void _syrk(std::complex<double>* matrixA, std::complex<double>* matrixC, U n, U k, U lda, U ldc,
    const blasEngineArgumentPackage_syrk<std::complex<double>>& srcPackage);
};

#include "cblasEngine.hpp"

#endif /* CBLASENGINE_H_ */
