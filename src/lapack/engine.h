/* Author: Edward Hutter */

#ifndef LAPACK__ENGINE_H_
#define LAPACK__ENGINE_H_

#include "./../util/shared.h"
// Goal: Have a LAPACK Policy with the particular LAPACK implementation as one of the Policy classes
//       This will allow for easier switching when needing alternate LAPACK implementations

namespace lapack{

// LAPACK libraries typically work on 4 different datatypes
//   single precision real(float)
//   double precision real (double)
//   single precision complex (std::complex<float>)
//   double precision complex (std::complex<double>)
// So it'd be best if we used partial template specialization to specialize the lapack Engine Policy classes
//   so as to only define the routines with T = one of the 4 types above


// Enum definitions for the user

enum class lapackEngineOrder : unsigned char{
  AlapackRowMajor = 0x0,
  AlapackColumnMajor = 0x1
};

enum class lapackEngineUpLo : unsigned char{
  AlapackLower = 0x0,
  AlapackUpper = 0x1
};

enum class lapackEngineDiag : unsigned char{
  AlapackNonUnit = 0x0,
  AlapackUnit = 0x1
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
class lapackEngineArgumentPackage{
public:
  // Base class contains a single member variable that can be used by its derived classes without explicitely casting
  lapackEngineMethod method;
};

// Now we have the derived classes that inherit from lapackEngineArgumentPackage and contain the necessary arguments for the LAPACK method
//   specified by the user

class lapackEngineArgumentPackage_potrf : public lapackEngineArgumentPackage{
public:
  lapackEngineArgumentPackage_potrf(lapackEngineOrder orderArg, lapackEngineUpLo uploArg){
    this->method = lapackEngineMethod::AlapackPotrf;
    this->order = orderArg;
    this->uplo = uploArg;
  }

  lapackEngineOrder order;
  lapackEngineUpLo uplo;
};

class lapackEngineArgumentPackage_trtri : public lapackEngineArgumentPackage{
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

class lapackEngineArgumentPackage_geqrf : public lapackEngineArgumentPackage{
public:
  lapackEngineArgumentPackage_geqrf(lapackEngineOrder orderArg){
    this->method = lapackEngineMethod::AlapackGeqrf;
    this->order = orderArg;
  }

  lapackEngineOrder order;
};

class lapackEngineArgumentPackage_orgqr : public lapackEngineArgumentPackage{
public:
  lapackEngineArgumentPackage_orgqr(lapackEngineOrder orderArg){
    this->method = lapackEngineMethod::AlapackOrgqr;
    this->order = orderArg;
  }

  lapackEngineOrder order;
};
}

// Local includes -- include the possible LAPACK libraries
#include "interface.h"


#endif /* LAPACKENGINE_H_ */
