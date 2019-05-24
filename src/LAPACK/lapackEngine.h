/* Author: Edward Hutter */

#ifndef LAPACKENGINE_H_
#define LAPACKENGINE_H_

#include "./../Util/shared.h"
// Goal: Have a LAPACK Policy with the particular LAPACK implementation as one of the Policy classes
//       This will allow for easier switching when needing alternate LAPACK implementations

// LAPACK libraries typically work on 4 different datatypes
//   single precision real(float)
//   double precision real (double)
//   single precision complex (std::complex<float>)
//   double precision complex (std::complex<double>)
// So it'd be best if we used partial template specialization to specialize the lapack Engine Policy classes
//   so as to only define the routines with T = one of the 4 types above


// Enum definitions for the user

enum class lapackEngineOrder : unsigned char{
  AblasRowMajor = 0x0,
  AblasColumnMajor = 0x1
};

enum class lapackEngineUpLo : unsigned char{
  AblasLower = 0x0,
  AblasUpper = 0x1
};

enum class lapackEngineDiag : unsigned char{
  AblasNonUnit = 0x0,
  AblasUnit = 0x1
};

enum class lapackEngineMethod : unsigned char{
  AlapackPotrf = 0x0,
  AlapackTrtri = 0x1,
  AlapackGeqrf = 0x10,
  AlapackOrgqr = 0x11
};

// Empty Base class for generic LAPACK method arguments -- this is to prevent another template overload that is unnecessary

// We need to template this because we use the base class as the "type" of derived class memory in places like MatrixMultiplication
  // so that we don't need multiple functions for using gemm or dtrmm, etc.
template<typename T>
class lapackEngineArgumentPackage{
public:
  // Base class contains a single member variable that can be used by its derived classes without explicitely casting
  lapackEngineMethod method;
};

// Now we have the derived classes that inherit from blasEngineArgumentPackage and contain the necessary arguments for the BLAS method
//   specified by the user

// We assume that the user will use T = float, double, complex<float>, or complex<double>
	// We could declare this class template, and then use full template specialization to just implement
	// those 4 cases, but the BLAS compiler will catch it anyways. The first option is always on the table
template<typename T>
class lapackEngineArgumentPackage_potrf : public lapackEngineArgumentPackage<T>{
public:
  lapackEngineArgumentPackage_potrf(lapackEngineOrder orderArg, lapackEngineUpLo uploArg){
    this->method = lapackEngineMethod::AlapackPotrf;
    this->order = orderArg;
    this->uplo = uploArg;
  }

  lapackEngineOrder order;
  lapackEngineUpLo uplo;
};

template<typename T>
class lapackEngineArgumentPackage_trtri : public lapackEngineArgumentPackage<T>{
public:
  lapackEngineArgumentPackage_trtri(lapackEngineOrder orderArg, lapackEngineUpLo uploArg, lapackEngineDiag diagArg){
    this->method = lapackEngineMethod::AlapackTrtri;
    this->order = orderArg;
    this->uplo = uploArg;
    this->diag = diagArg;
  }

  lapackEngineOrder order;
  lapackEngineUpLo uplo;
  lapackEngineDiag diag;
};

template<typename T>
class lapackEngineArgumentPackage_geqrf : public lapackEngineArgumentPackage<T>{
public:
  lapackEngineArgumentPackage_geqrf(lapackEngineOrder orderArg){
    this->method = lapackEngineMethod::AlapackGeqrf;
    this->order = orderArg;
  }

  lapackEngineOrder order;
};

template<typename T>
class lapackEngineArgumentPackage_orgqr : public lapackEngineArgumentPackage<T>{
public:
  lapackEngineArgumentPackage_orgqr(lapackEngineOrder orderArg){
    this->method = lapackEngineMethod::AlapackOrgqr;
    this->order = orderArg;
  }

  lapackEngineOrder order;
};

// Local includes -- include the possible LAPACK libraries
#include "interface.h"


#endif /* LAPACKENGINE_H_ */
