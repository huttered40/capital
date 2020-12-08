/* Author: Edward Hutter */

#ifndef BLAS_INTERFACE_H_
#define BLAS_INTERFACE_H_

// Local includes
#include "./../util/shared.h"

// Goal: Have a BLAS Policy with the particular BLAS implementation as one of the Policy classes
//       This will allow for easier switching when needing alternate BLAS implementations

namespace blas{

// BLAS libraries typically work on 4 different datatypes
//   single precision real(float)
//   double precision real (double)
//   single precision complex (std::complex<float>)
//   double precision complex (std::complex<double>)
// So it'd be best if we used partial template specialization to specialize the blasEngine Policy classes
//   so as to only define the routines with T = one of the 4 types above

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
  template<typename T>
  static void setInfoParameters_gemm(const ArgPack_gemm<T>& srcPackage, CBLAS_ORDER& destArg1, CBLAS_TRANSPOSE& destArg2, CBLAS_TRANSPOSE& destArg3);

  template<typename T>
  static void setInfoParameters_trmm(const ArgPack_trmm<T>& srcPackage, CBLAS_ORDER& destArg1, CBLAS_SIDE& destArg2, CBLAS_UPLO& destArg3, CBLAS_TRANSPOSE& destArg4, CBLAS_DIAG& destArg5);

  template<typename T>
  static void setInfoParameters_syrk(const ArgPack_syrk<T>& srcPackage, CBLAS_ORDER& destArg1, CBLAS_UPLO& destArg2, CBLAS_TRANSPOSE& destArg3);
};


// ************************************************************************************************************************************************************
// Declare this fully templated "base" class but do not define it. This prevents users from using this class, but
//   allows partially specialized template classes to specialize it.
class engine : helper{
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
  static void _gemm(T* matrixA, T* matrixB, T* matrixC, int64_t m, int64_t n, int64_t k,
                      int64_t lda, int64_t ldb, int64_t ldc, const ArgPack_gemm<T>& srcPackage);

  template<typename T>
  static void _trmm(T* matrixA, T* matrixB, int64_t m, int64_t n, int64_t lda, int64_t ldb, const ArgPack_trmm<T>& srcPackage);

  template<typename T>
  static void _syrk(T* matrixA, T* matrixC, int64_t n, int64_t k, int64_t lda, int64_t ldc, const ArgPack_syrk<T>& srcPackage);
};
}

#include "interface.hpp"

#endif /* BLAS_INTERFACE_H_ */
