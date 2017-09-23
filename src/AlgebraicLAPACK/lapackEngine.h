/* Author: Edward Hutter */

#ifndef LAPACK_ENGINE_H_
#define LAPACK_ENGINE_H_

// Goal: Have a LAPACK Policy with the particular LAPACK implementation as one of the Policy classes
//       This will allow for easier switching when needing alternate LAPACK implementations and APIs such as LAPACKE and FLAME

// LAPACK libraries typically work on 4 different datatypes
//   single precision real(float)
//   double precision real (double)
//   single precision complex (std::complex<float>)
//   double precision complex (std::complex<double>)
// So it'd be best if we used partial template specialization to specialize the lapackEngine Policy classes
//   so as to only define the routines with T = one of the 4 types above


// Enum definitions for the user

enum class lapackEngineOrder : unsigned char
{
  AlapackRowMajor = 0x0,
  AlapackColumnMajor = 0x1
};

enum class lapackEngineUpLo : unsigned char
{
  AblasLower = 0x0,
  AblasUpper = 0x1
};

/*
// Empty Base class for generic BLAS method arguments -- this is to prevent another template overload that is unnecessary

// We need to template this because we use the base class as the "type" of derived class memory in places like MatrixMultiplication
  // so that we don't need multiple functions for using gemm or dtrmm, etc.
template<typename T>
class blasEngineArgumentPackage
{
public:
  // Base class contains a single member variable that can be used by its derived classes without explicitely casting
  blasEngineMethod method;
};

// Now we have the derived classes that inherit from blasEngineArgumentPackage and contain the necessary arguments for the BLAS method
//   specified by the user


// See comment below
template<typename T>
class blasEngineArgumentPackage_gemm : public blasEngineArgumentPackage<T>
{
public:
  blasEngineArgumentPackage_gemm() { this->method = blasEngineMethod::AblasGemm; }

  blasEngineOrder order;
  blasEngineTranspose transposeA;
  blasEngineTranspose transposeB;
  T alpha;					// Added these two constants, alpha and beta
  T beta;
};

// As above, we assume that the user will use T = float, double, complex<float>, or complex<double>
	// We could declare this template class, and then use full template specialization to just implement
	// those 4 cases, but the BLAS compiler will catch it anyways, but the first option is always on the table
template<typename T>
class blasEngineArgumentPackage_trmm : public blasEngineArgumentPackage<T>
{
public:
  blasEngineArgumentPackage_trmm() { this->method = blasEngineMethod::AblasTrmm; }

  blasEngineOrder order;
  blasEngineSide side;
  blasEngineUpLo uplo;
  blasEngineTranspose transposeA;
  blasEngineDiag diag;
  T alpha;					// Added this constant
};

// Local includes -- include the possible BLAS libraries
#include "lapackeEngine.h"

*/

#endif /*LAPACK_ENGINE_H_*/