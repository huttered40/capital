/* Author: Edward Hutter */

#ifndef LAPACK_INTERFACE_H_
#define LAPACK_INTERFACE_H_

// ************************************************************************************************************************************************************
// System includes
#include <complex>

#ifdef PORTER
#include "/home/hutter2/hutter2/external/BLAS/OpenBLAS/lapack-netlib/LAPACKE/include/lapacke.h"
#endif

#ifdef THETA
#include "mkl.h"
#endif

#ifdef STAMPEDE2
#include "mkl.h"
#endif

#if defined(BGQ) || defined(BLUEWATERS)
// Note: LAPACK Fortran routines must be externed so that linker knows where to look
extern "C" void spotrf_(char*, int*, double*, int*, int*);
extern "C" void strtri_(char*, char*, int*, double*, int*, int*);
extern "C" void sgeqrf_(int*, int*, double*, int*, double*, int*);
extern "C" void sorgqr_(int*, int*, int*, double*, int*, double*, int*);
extern "C" void dpotrf_(char*, int*, double*, int*, int*);
extern "C" void dtrtri_(char*, char*, int*, double*, int*, int*);
extern "C" void dgeqrf_(int*, int*, double*, int*, double*, int*);
extern "C" void dorgqr_(int*, int*, int*, double*, int*, double*, int*);
extern "C" void cpotrf_(char*, int*, std::complex<float>*, int*, int*);
extern "C" void ctrtri_(char*, char*, int*, std::complex<float>*, int*, int*);
extern "C" void cgeqrf_(int*, int*, std::complex<float>*, int*, std::complex<float>*, int*);
extern "C" void zpotrf_(char*, int*, double*, int*, int*);
extern "C" void ztrtri_(char*, char*, int*, double*, int*, int*);
extern "C" void zgeqrf_(int*, int*, std::complex<double>*, int*, std::complex<double>*, int*);

// These constants below are placeholders. They mean nothing on these machines.
#define LAPACK_ROW_MAJOR 0
#define LAPACK_COL_MAJOR 0
#endif

#if 0// Try this for LAPACKE interface
extern "C" void LAPACKE_spotrf(char, char, int, float*, int);
extern "C" void LAPACKE_strtri(char, char, char, int, float*, int);
extern "C" void LAPACKE_sgeqrf(char, int, int, float*, int, float*);
extern "C" void LAPACKE_sorgqr(char, int, int, int, float*, int, float*);
extern "C" void LAPACKE_dpotrf(char, char, int, double*, int);
extern "C" void LAPACKE_dtrtri(char, char, char, int, double*, int);
extern "C" void LAPACKE_dgeqrf(char, int, int, double*, int, float*);
extern "C" void LAPACKE_dorgqr(char, int, int, int, double*, int, float*);
extern "C" void LAPACKE_cpotrf(char, char, int, std::complex<float>*, int);
extern "C" void LAPACKE_ctrtri(char, char, char, int, std::complex<float>*, int);
extern "C" void LAPACKE_cgeqrf(char, int, int, std::complex<float>*, int, std::complex<float>*);
extern "C" void LAPACKE_zpotrf(char, char, int, std::complex<double>*, int);
extern "C" void LAPACKE_ztrtri(char, char, char, int, std::complex<double>*, int);
extern "C" void LAPACKE_zgeqrf(char, int, int, std::complex<double>*, int, std::complex<double>*);
#endif

// Local includes
#include "./../util/shared.h"

namespace lapack{

// ************************************************************************************************************************************************************
template<typename T>
class LType{};

// ************************************************************************************************************************************************************
auto GetPOTRFroutine(LType<float>){
#if defined(BGQ) || defined(BLUEWATERS)
  return &spotrf_;
#else
  return &LAPACKE_spotrf;
#endif
}
auto GetPOTRFroutine(LType<double>){
#if defined(BGQ) || defined(BLUEWATERS)
  return &dpotrf_;
#else
  return &LAPACKE_dpotrf;
#endif
}
auto GetPOTRFroutine(LType<std::complex<float>>){
#if defined(BGQ) || defined(BLUEWATERS)
  return &cpotrf_;
#else
  return &LAPACKE_cpotrf;
#endif
}
auto GetPOTRFroutine(LType<std::complex<double>>){
#if defined(BGQ) || defined(BLUEWATERS)
  return &zpotrf_;
#else
  return &LAPACKE_zpotrf;
#endif
}

auto GetTRTRIroutine(LType<float>){
#if defined(BGQ) || defined(BLUEWATERS)
  return &strtri_;
#else
  return &LAPACKE_strtri;
#endif
}
auto GetTRTRIroutine(LType<double>){
#if defined(BGQ) || defined(BLUEWATERS)
  return &dtrtri_;
#else
  return &LAPACKE_dtrtri;
#endif
}
auto GetTRTRIroutine(LType<std::complex<float>>){
#if defined(BGQ) || defined(BLUEWATERS)
  return &ctrtri_;
#else
  return &LAPACKE_ctrtri;
#endif
}
auto GetTRTRIroutine(LType<std::complex<double>>){
#if defined(BGQ) || defined(BLUEWATERS)
  return &ztrtri_;
#else
  return &LAPACKE_ztrtri;
#endif
}

auto GetGEQRFroutine(LType<float>){
#if defined(BGQ) || defined(BLUEWATERS)
  return &sgeqrf_;
#else
  return &LAPACKE_sgeqrf;
#endif
}
auto GetGEQRFroutine(LType<double>){
#if defined(BGQ) || defined(BLUEWATERS)
  return &dgeqrf_;
#else
  return &LAPACKE_dgeqrf;
#endif
}
auto GetGEQRFroutine(LType<std::complex<float>>){
#if defined(BGQ) || defined(BLUEWATERS)
  return &cgeqrf_;
#else
  return &LAPACKE_cgeqrf;
#endif
}
auto GetGEQRFroutine(LType<std::complex<double>>){
#if defined(BGQ) || defined(BLUEWATERS)
  return &zgeqrf_;
#else
  return &LAPACKE_zgeqrf;
#endif
}

auto GetORGQRroutine(LType<float>){
#if defined(BGQ) || defined(BLUEWATERS)
  return &sorgqr_;
#else
  return &LAPACKE_sorgqr;
#endif
}
auto GetORGQRroutine(LType<double>){
#if defined(BGQ) || defined(BLUEWATERS)
  return &dorgqr_;
#else
  return &LAPACKE_dorgqr;
#endif
}

// ************************************************************************************************************************************************************
class helper{
public:
  helper() = delete;
  helper(const helper& rhs) = delete;
  helper(helper&& rhs) = delete;
  helper& operator=(const helper& rhs) = delete;
  helper& operator=(helper&& rhs) = delete;

// Make these methods protected so that only the derived classes can access them.
protected:
  static void setInfoParameters_potrf(const ArgPack_potrf& srcPackage, int& destArg1, char& destArg2);
  static void setInfoParameters_trtri(const ArgPack_trtri& srcPackage, int& destArg1, char& destArg2, char& destArg3);
  static void setInfoParameters_geqrf(const ArgPack_geqrf& srcPackage, int& destArg1);
  static void setInfoParameters_orgqr(const ArgPack_orgqr& srcPackage, int& destArg1);
};


// ************************************************************************************************************************************************************
// Declare this fully templated "base" class but do not define it. This prevents users from using this class, but
//   allows partially specialized template classes to specialize it.

// Note: each method below could also parameterize itself on the int-type (32/64 bit), but current FORTRAN libraries only have 32-bit
class engine : public helper{
  // Lets prevent any instances of this class from being created.
public:
  engine() = delete;
  engine(const engine& rhs) = delete;
  engine(engine&& rhs) = delete;
  engine& operator=(const engine& rhs) = delete;
  engine& operator=(engine&& rhs) = delete;
  ~engine() = delete;

  // Engine methods
  template<typename T>
  static void _potrf(T* matrixA, int n, int lda, const ArgPack_potrf& srcPackage);

  template<typename T>
  static void _trtri(T* matrixA, int n, int lda, const ArgPack_trtri& srcPackage);

  template<typename T>
  static void _geqrf(T* matrixA, T* tau, int m, int n, int lda, const ArgPack_geqrf& srcPackage);

  template<typename T>
  static void _orgqr(T* matrixA, T* tau, int m, int n, int k, int lda, const ArgPack_orgqr& srcPackage);
};
}

#include "interface.hpp"

#endif /* LAPACK_INTERFACE_H_ */
