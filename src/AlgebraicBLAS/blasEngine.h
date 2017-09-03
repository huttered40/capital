/* Author: Edward Hutter */

#ifndef BLASENGINE_H_
#define BLASENGINE_H_

// Goal: Have a BLAS Policy with the particular BLAS implementation as one of the Policy classes
//       This will allow for easier switching when needing alternate BLAS implementations

// BLAS libraries typically work on 4 different datatypes
//   single precision real(float)
//   double precision real (double)
//   single precision complex (std::complex<float>)
//   double precision complex (std::complex<double>)
// So it'd be best if we used partial template specialization to specialize the blasEngine Policy classes
//   so as to only define the routines with T = one of the 4 types above


// Enum definitions for the user

enum class blasEngineOrder : unsigned char
{
  AblasRowMajor = 0x0,
  AblasColumnMajor = 0x1
};

enum class blasEngineTranspose : unsigned char
{
  AblasNoTrans = 0x0,
  AblasTrans = 0x1
};

enum class blasEngineSide : unsigned char
{
  AblasLeft = 0x0,
  AblasRight = 0x1
};

enum class blasEngineUpLo : unsigned char
{
  AblasLower = 0x0,
  AblasUpper = 0x1
};

enum class blasEngineDiag : unsigned char
{
  AblasNonUnit = 0x0,
  AblasUnit = 0x1
};

enum class blasEngineMethod : unsigned char
{
  AblasGemm = 0x0,
  AblasTrmm = 0x1
};

// Empty Base class for generic BLAS method arguments -- this is to prevent another template overload that is unnecessary
class blasEngineArgumentPackage
{
public:
  // Base class contains a single member variable that can be used by its derived classes without explicitely casting
  blasEngineMethod method;
};

// Now we have the derived classes that inherit from blasEngineArgumentPackage and contain the necessary arguments for the BLAS method
//   specified by the user

class blasEngineArgumentPackage_gemm : public blasEngineArgumentPackage
{
public:
  blasEngineArgumentPackage_gemm() { method = blasEngineMethod::AblasGemm; }

  blasEngineOrder order;
  blasEngineTranspose transposeA;
  blasEngineTranspose transposeB;
};

class blasEngineArgumentPackage_trmm : public blasEngineArgumentPackage
{
public:
  blasEngineArgumentPackage_trmm() { method = blasEngineMethod::AblasTrmm; }

  blasEngineOrder order;
  blasEngineSide side;
  blasEngineUpLo uplo;
  blasEngineTranspose transposeA;
  blasEngineDiag diag;
};

// Local includes -- include the possible BLAS libraries
#include "cblasEngine.h"


#endif /* BLASENGINE_H_ */
