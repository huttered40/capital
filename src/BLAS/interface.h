/* Author: Edward Hutter */

#ifndef BLAS_INTERFACE_H_
#define BLAS_INTERFACE_H_

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

// Note: Blue Waters does not include libraries that support the CBLAS interface. Must use the Fortran interface
#ifdef BLUEWATERS
#include "/sw/xe/cblas/2003.02.23/cnl5.2_gnu4.8.2/include/cblas.h"
/*
  #define CBLAS_ORDER char
  #define CBLAS_SIDE char
  #define CBLAS_TRANSPOSE char
  #define CBLAS_UPLO char
  #define CBLAS_DIAG char
  #define CblasRowMajor 'R'
  #define CblasColMajor 'c'
  #define CblasLower 'L'
  #define CblasUpper 'U'
  #define CblasTrans 'T'
  #define CblasNoTrans 'N'
  #define CblasLeft 'L'
  #define CblasRight 'R'
  #define CblasUnit 'U'
  #define CblasNonUnit 'N'
*/
#endif

// ************************************************************************************************************************************************************
#ifdef FLOAT_TYPE
extern void cblas_sgemm(const enum CBLAS_ORDER, const enum CBLAS_TRANSPOSE, const enum CBLAS_TRANSPOSE, const int, const int, const int, const float, const float *, const int, const float *, const int, const float, float *, const int);
extern void cblas_strmm(const enum CBLAS_ORDER, const enum CBLAS_SIDE, const enum CBLAS_UPLO, const enum CBLAS_TRANSPOSE, const enum CBLAS_DIAG, const int, const int, const float, const float *, const int, float *, const int);
extern void cblas_ssyrk(const enum CBLAS_ORDER, const enum CBLAS_UPLO, const enum CBLAS_TRANSPOSE, const int, const int, const float, const float *, const int, const float, float *, const int);
#endif
#ifdef DOUBLE_TYPE
extern void cblas_dgemm(const enum CBLAS_ORDER, const enum CBLAS_TRANSPOSE, const enum CBLAS_TRANSPOSE, const int, const int, const int, const double, const double *, const int, const double *, const int, const double, double *, const int);
extern void cblas_dtrmm(const enum CBLAS_ORDER, const enum CBLAS_SIDE, const enum CBLAS_UPLO, const enum CBLAS_TRANSPOSE, const enum CBLAS_DIAG, const int, const int, const double, const double *, const int, double *, const int);
extern void cblas_dsyrk(const enum CBLAS_ORDER, const enum CBLAS_UPLO, const enum CBLAS_TRANSPOSE, const int, const int, const double, const double *, const int, const double, double *, const int);
#endif
#ifdef COMPLEX_FLOAT_TYPE
extern void cblas_cgemm(const enum CBLAS_ORDER, const enum CBLAS_TRANSPOSE, const enum CBLAS_TRANSPOSE, const int, const int, const int, const std::complex<float>, const std::complex<float> *, const int, const std::complex<float> *, const int, const std::complex<float>, std::complex<float> *, const int);
extern void cblas_ctrmm(const enum CBLAS_ORDER, const enum CBLAS_SIDE, const enum CBLAS_UPLO, const enum CBLAS_TRANSPOSE, const enum CBLAS_DIAG, const int, const int, const std::complex<float>, const std::complex<float> *, const int, std::complex<float> *, const int);
extern void cblas_csyrk(const enum CBLAS_ORDER, const enum CBLAS_UPLO, const enum CBLAS_TRANSPOSE, const int, const int, const std::complex<float>, const std::complex<float> *, const int, const std::complex<float>, std::complex<float> *, const int);
#endif
#ifdef COMPLEX_DOUBLE_TYPE
extern void cblas_zgemm(const enum CBLAS_ORDER, const enum CBLAS_TRANSPOSE, const enum CBLAS_TRANSPOSE, const int, const int, const int, const std::complex<double>, const std::complex<double> *, const int, const std::complex<double> *, const int, const std::complex<double>, std::complex<double> *, const int);
extern void cblas_ztrmm(const enum CBLAS_ORDER, const enum CBLAS_SIDE, const enum CBLAS_UPLO, const enum CBLAS_TRANSPOSE, const enum CBLAS_DIAG, const int, const int, const std::complex<double>, const std::complex<double> *, const int, std::complex<double> *, const int);
extern void cblas_zsyrk(const enum CBLAS_ORDER, const enum CBLAS_UPLO, const enum CBLAS_TRANSPOSE, const int, const int, const std::complex<double>, const std::complex<double> *, const int, const std::complex<double>, std::complex<double> *, const int);
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

// ************************************************************************************************************************************************************
template<typename T>
class BType{};

// ************************************************************************************************************************************************************
auto GetGEMMroutine(BType<float>) -> decltype(&cblas_sgemm){
  return &cblas_sgemm;
}
auto GetGEMMroutine(BType<double>) -> decltype(&cblas_dgemm){
  return &cblas_dgemm;
}
auto GetGEMMroutine(BType<std::complex<float>>) -> decltype(&cblas_cgemm){
  return &cblas_cgemm;
}
auto GetGEMMroutine(BType<std::complex<double>>) -> decltype(&cblas_zgemm){
  return &cblas_zgemm;
}

auto GetTRMMroutine(BType<float>) -> decltype(&cblas_strmm){
  return &cblas_strmm;
}
auto GetTRMMroutine(BType<double>) -> decltype(&cblas_dtrmm){
  return &cblas_dtrmm;
}
auto GetTRMMroutine(BType<std::complex<float>>) -> decltype(&cblas_ctrmm){
  return &cblas_ctrmm;
}
auto GetTRMMroutine(BType<std::complex<double>>) -> decltype(&cblas_ztrmm){
  return &cblas_ztrmm;
}

auto GetSYRKroutine(BType<float>) -> decltype(&cblas_ssyrk){
  return &cblas_ssyrk;
}
auto GetSYRKroutine(BType<double>) -> decltype(&cblas_dsyrk){
  return &cblas_dsyrk;
}
auto GetSYRKroutine(BType<std::complex<float>>) -> decltype(&cblas_csyrk){
  return &cblas_csyrk;
}
auto GetSYRKroutine(BType<std::complex<double>>) -> decltype(&cblas_zsyrk){
  return &cblas_zsyrk;
}

// ************************************************************************************************************************************************************
class blasHelper{
public:
  blasHelper() = delete;
  blasHelper(const blasHelper& rhs) = delete;
  blasHelper(blasHelper&& rhs) = delete;
  blasHelper& operator=(const blasHelper& rhs) = delete;
  blasHelper& operator=(blasHelper&& rhs) = delete;

// Make these methods protected so that only the derived classes can access them.
protected:
  template<typename T>
  static void setInfoParameters_gemm(const blasEngineArgumentPackage_gemm<T>& srcPackage, CBLAS_ORDER& destArg1, CBLAS_TRANSPOSE& destArg2, CBLAS_TRANSPOSE& destArg3);

  template<typename T>
  static void setInfoParameters_trmm(const blasEngineArgumentPackage_trmm<T>& srcPackage, CBLAS_ORDER& destArg1, CBLAS_SIDE& destArg2, CBLAS_UPLO& destArg3, CBLAS_TRANSPOSE& destArg4, CBLAS_DIAG& destArg5);

  template<typename T>
  static void setInfoParameters_syrk(const blasEngineArgumentPackage_syrk<T>& srcPackage, CBLAS_ORDER& destArg1, CBLAS_UPLO& destArg2, CBLAS_TRANSPOSE& destArg3);
};


// ************************************************************************************************************************************************************
// Declare this fully templated "base" class but do not define it. This prevents users from using this class, but
//   allows partially specialized template classes to specialize it.
template<typename T, typename U>
class blasEngine : blasHelper{
  // Lets prevent any instances of this class from being created.
public:
  blasEngine() = delete;
  blasEngine(const blasEngine& rhs) = delete;
  blasEngine(blasEngine&& rhs) = delete;
  blasEngine& operator=(const blasEngine& rhs) = delete;
  blasEngine& operator=(blasEngine&& rhs) = delete;
  ~blasEngine() = delete;

  static auto _gemm_ = GetGEMMroutine(BType<T>());
  static auto _trmm_ = GetTRMMroutine(BType<T>());
  static auto _syrk_ = GetSYRKroutine(BType<T>());

  // Engine methods
  static void _gemm(T* matrixA, T* matrixB, T* matrixC, U m, U n, U k,
                      U lda, U ldb, U ldc, const blasEngineArgumentPackage_gemm<T>& srcPackage);
  static void _trmm(T* matrixA, T* matrixB, U m, U n, U lda, U ldb, const blasEngineArgumentPackage_trmm<T>& srcPackage);
  static void _syrk(T* matrixA, T* matrixC, U n, U k, U lda, U ldc, const blasEngineArgumentPackage_syrk<T>& srcPackage);
};

#include "interface.hpp"

#endif /* BLAS_INTERFACE_H_ */
