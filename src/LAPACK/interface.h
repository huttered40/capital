/* Author: Edward Hutter */

#ifndef LAPACK_INTERFACE_H_
#define LAPACK_INTERFACE_H_

// ************************************************************************************************************************************************************
// System includes
#include <complex>

#ifdef PORTER
#include "/home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/lapack-netlib/LAPACKE/include/lapacke.h"
#endif

#ifdef THETA
#include "mkl.h"
#endif

#ifdef STAMPEDE2
#include "mkl.h"
#endif

#if defined(BGQ) || defined(BLUEWATERS)
// Note: LAPACK Fortran routines must be externed so that linker knows where to look
#ifdef FLOAT_TYPE
extern "C" void spotrf_(char*, int*, double*, int*, int*);
extern "C" void strtri_(char*, char*, int*, double*, int*, int*);
extern "C" void sgeqrf_(int*, int*, double*, int*, double*, int*);
extern "C" void sorgqr_(int*, int*, int*, double*, int*, double*, int*);
#endif
#ifdef DOUBLE_TYPE
extern "C" void dpotrf_(char*, int*, double*, int*, int*);
extern "C" void dtrtri_(char*, char*, int*, double*, int*, int*);
extern "C" void dgeqrf_(int*, int*, double*, int*, double*, int*);
extern "C" void dorgqr_(int*, int*, int*, double*, int*, double*, int*);
#endif
#ifdef COMPLEX_FLOAT_TYPE
extern "C" void cpotrf_(char*, int*, std::complex<float>*, int*, int*);
extern "C" void ctrtri_(char*, char*, int*, std::complex<float>*, int*, int*);
extern "C" void cgeqrf_(int*, int*, std::complex<float>*, int*, std::complex<float>*, int*);
extern "C" void corgqr_(int*, int*, int*, std::complex<float>*, int*, std::complex<float>*, int*);
#endif
#ifdef COMPLEX_DOUBLE_TYPE
extern "C" void zpotrf_(char*, int*, double*, int*, int*);
extern "C" void ztrtri_(char*, char*, int*, double*, int*, int*);
extern "C" void zgeqrf_(int*, int*, std::complex<double>*, int*, std::complex<double>*, int*);
extern "C" void zorgqr_(int*, int*, int*, std::complex<double>*, int*, std::complex<double>*, int*);
#endif
#else // Try this for LAPACKE interface
#ifdef FLOAT_TYPE
extern "C" void LAPACKE_spotrf(char, char, int, float*, int);
extern "C" void LAPACKE_strtri(char, char, char, int, float*, int);
extern "C" void LAPACKE_sgeqrf_(char, int, int, float*, int, float*);
extern "C" void LAPACKE_sorgqr_(char, int, int, int, float*, int, float*);
#endif
#ifdef DOUBLE_TYPE
extern "C" void LAPACKE_dpotrf(char, char, int, double*, int);
extern "C" void LAPACKE_dtrtri(char, char, char, int, double*, int);
extern "C" void LAPACKE_sgeqrf_(char, int, int, double*, int, float*);
extern "C" void LAPACKE_sorgqr_(char, int, int, int, double*, int, float*);
#endif
#ifdef COMPLEX_FLOAT_TYPE
extern "C" void LAPACKE_cpotrf(char, char, int, std::complex<float>*, int);
extern "C" void LAPACKE_ctrtri(char, char, char, int, std::complex<float>*, int);
extern "C" void LAPACKE_sgeqrf_(char, int, int, std::complex<float>*, int, std::complex<float>*);
extern "C" void LAPACKE_sorgqr_(char, int, int, int, std::complex<float>*, int, std::complex<float>*);
#endif
#ifdef COMPLEX_DOUBLE_TYPE
extern "C" void LAPACKE_zpotrf(char, char, int, std::complex<double>*, int);
extern "C" void LAPACKE_ztrtri(char, char, char, int, std::complex<double>*, int);
extern "C" void LAPACKE_sgeqrf_(char, int, int, std::complex<double>*, int, std::complex<double>*);
extern "C" void LAPACKE_sorgqr_(char, int, int, int, std::complex<double>*, int, std::complex<double>*);
#endif
#endif

// Local includes
#include "./../Util/shared.h"
#include "./../Timer/CTFtimer.h"

// ************************************************************************************************************************************************************
template<typename T>
void* GetPOTRFroutine();

template<>
void* GetPOTRFroutine<float>(){
#if defined(BGQ) || defined(BLUEWATERS)
  return &spotrf_;
#else
  return &LAPACKE_spotrf;
#endif
}
template<>
void* GetPOTRFroutine<double>(){
#if defined(BGQ) || defined(BLUEWATERS)
  return &dpotrf_;
#else
  return &LAPACKE_dpotrf;
#endif
}
template<>
void* GetPOTRFroutine<std::complex<float>>(){
#if defined(BGQ) || defined(BLUEWATERS)
  return &cpotrf_;
#else
  return &LAPACKE_cpotrf;
#endif
}
template<>
void* GetPOTRFroutine<std::complex<double>>(){
#if defined(BGQ) || defined(BLUEWATERS)
  return &zpotrf_;
#else
  return &LAPACKE_zpotrf;
#endif
}

template<typename T>
void* GetTRTRIroutine();

template<>
void* GetTRTRIroutine<float>(){
#if defined(BGQ) || defined(BLUEWATERS)
  return &strtri_;
#else
  return &LAPACKE_strtri;
#endif
}
template<>
void* GetTRTRIroutine<double>(){
#if defined(BGQ) || defined(BLUEWATERS)
  return &dtrtri_;
#else
  return &LAPACKE_dtrtri;
#endif
}
template<>
void* GetTRTRIroutine<std::complex<float>>(){
#if defined(BGQ) || defined(BLUEWATERS)
  return &ctrtri_;
#else
  return &LAPACKE_ctrtri;
#endif
}
template<>
void* GetTRTRIroutine<std::complex<double>>(){
#if defined(BGQ) || defined(BLUEWATERS)
  return &ztrtri_;
#else
  return &LAPACKE_ztrtri;
#endif
}

template<typename T>
void* GetGEQRFroutine();

template<>
void* GetGEQRFroutine<float>(){
#if defined(BGQ) || defined(BLUEWATERS)
  return &sgeqrf_;
#else
  return &LAPACKE_sgeqrf;
#endif
}
template<>
void* GetGEQRFroutine<double>(){
#if defined(BGQ) || defined(BLUEWATERS)
  return &dgeqrf_;
#else
  return &LAPACKE_dgeqrf;
#endif
}
template<>
void* GetGEQRFroutine<std::complex<float>>(){
#if defined(BGQ) || defined(BLUEWATERS)
  return &cgeqrf_;
#else
  return &LAPACKE_cgeqrf;
#endif
}
template<>
void* GetGEQRFroutine<std::complex<double>>(){
#if defined(BGQ) || defined(BLUEWATERS)
  return &zgeqrf_;
#else
  return &LAPACKE_zgeqrf;
#endif
}

template<typename T>
void* GetORGQRroutine();

template<>
void* GetORGQRroutine<float>(){
#if defined(BGQ) || defined(BLUEWATERS)
  return &sorgqr_;
#else
  return &LAPACKE_sorgqr;
#endif
}
template<>
void* GetORGQRroutine<double>(){
#if defined(BGQ) || defined(BLUEWATERS)
  return &dorgqr_;
#else
  return &LAPACKE_dorgqr;
#endif
}
template<>
void* GetORGQRroutine<std::complex<float>>(){
#if defined(BGQ) || defined(BLUEWATERS)
  return &corgqr_;
#else
  return &LAPACKE_corgqr;
#endif
}
template<>
void* GetORGQRroutine<std::complex<double>>(){
#if defined(BGQ) || defined(BLUEWATERS)
  return &zorgqr_;
#else
  return &LAPACKE_zorgqr;
#endif
}

// ************************************************************************************************************************************************************
class lapackHelper{
public:
  lapackHelper() = delete;
  lapackHelper(const lapackHelper& rhs) = delete;
  lapackHelper(lapackHelper&& rhs) = delete;
  lapackHelper& operator=(const lapackHelper& rhs) = delete;
  lapackHelper& operator=(lapackHelper&& rhs) = delete;

// Make these methods protected so that only the derived classes can access them.
protected:
  template<typename T>
  static void setInfoParameters_potrf(const lapackEngineArgumentPackage_potrf<T>& srcPackage, int& destArg1, char& destArg2);

  template<typename T>
  static void setInfoParameters_trtri(const lapackEngineArgumentPackage_trtri<T>& srcPackage, int& destArg1, char& destArg2, char& destArg3);

  template<typename T>
  static void setInfoParameters_geqrf(const lapackEngineArgumentPackage_geqrf<T>& srcPackage, int& destArg1);

  template<typename T>
  static void setInfoParameters_orgqr(const lapackEngineArgumentPackage_orgqr<T>& srcPackage, int& destArg1);
};


// ************************************************************************************************************************************************************
// Declare this fully templated "base" class but do not define it. This prevents users from using this class, but
//   allows partially specialized template classes to specialize it.
template<typename T, typename U>
class lapackEngine : public lapackHelper{
  // Lets prevent any instances of this class from being created.
public:
  lapackEngine() = delete;
  lapackEngine(const lapackEngine& rhs) = delete;
  lapackEngine(lapackEngine&& rhs) = delete;
  lapackEngine<T,U>& operator=(const lapackEngine& rhs) = delete;
  lapackEngine<T,U>& operator=(lapackEngine&& rhs) = delete;
  ~lapackEngine() = delete;

  static auto _potrf_ = GetPOTRFroutine<T>();
  static auto _trtri_ = GetTRTRIroutine<T>();
  static auto _geqrf_ = GetGEQRFroutine<T>();
  static auto _orgqr_ = GetORGQRroutine<T>();

  // Engine methods
  static void _potrf(T* matrixA, T* matrixB, T* matrixC, U m, U n, U k,
                      U lda, U ldb, U ldc, const lapackEngineArgumentPackage_potrf<T>& srcPackage);
  static void _trtri(T* matrixA, T* matrixB, U m, U n, U lda, U ldb, const lapackEngineArgumentPackage_trtri<T>& srcPackage);
  static void _geqrf(T* matrixA, T* matrixC, U n, U k, U lda, U ldc, const lapackEngineArgumentPackage_geqrf<T>& srcPackage);
  static void _orgqr(T* matrixA, T* matrixC, U n, U k, U lda, U ldc, const lapackEngineArgumentPackage_orgqr<T>& srcPackage);
};

#include "interface.hpp"

#endif /* LAPACK_INTERFACE_H_ */
