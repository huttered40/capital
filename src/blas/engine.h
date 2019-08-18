/* Author: Edward Hutter */

#ifndef BLAS__ENGINE_H_
#define BLAS__ENGINE_H_

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


// Enum definitions for the user

enum class Order : unsigned char{
  AblasRowMajor = 0x0,
  AblasColumnMajor = 0x1
};

enum class Transpose : unsigned char{
  AblasNoTrans = 0x0,
  AblasTrans = 0x1
};

enum class Side : unsigned char{
  AblasLeft = 0x0,
  AblasRight = 0x1
};

enum class UpLo : unsigned char{
  AblasLower = 0x0,
  AblasUpper = 0x1
};

enum class Diag : unsigned char{
  AblasNonUnit = 0x0,
  AblasUnit = 0x1
};

enum class Method : unsigned char{
  AblasGemm = 0x0,
  AblasTrmm = 0x1,
  AblasSyrk = 0x10
};

// Empty Base class for generic BLAS method arguments -- this is to prevent another template overload that is unnecessary

// We need to template this because we use the base class as the "type" of derived class memory in places like MatrixMultiplication
  // so that we don't need multiple functions for using gemm or dtrmm, etc.
template<typename T>
class ArgPack{
public:
  // Base class contains a single member variable that can be used by its derived classes without explicitely casting
  Method method;
};

// Now we have the derived classes that inherit from blasEngineArgumentPackage and contain the necessary arguments for the BLAS method
//   specified by the user


// We assume that the user will use T = float, double, complex<float>, or complex<double>
	// We could declare this template class, and then use full template specialization to just implement
	// those 4 cases, but the BLAS compiler will catch it anyways, but the first option is always on the table
template<typename T>
class ArgPack_gemm : public ArgPack<T>{
public:
  ArgPack_gemm(Order orderArg, Transpose transposeAArg, Transpose transposeBArg, T alphaArg, T betaArg){
    this->method = Method::AblasGemm;
    this->order = orderArg;
    this->transposeA = transposeAArg;
    this->transposeB = transposeBArg;
    this->alpha = alphaArg;
    this->beta = betaArg;
  }

  Order order;
  Transpose transposeA;
  Transpose transposeB;
  T alpha;					// Added these two constants, alpha and beta
  T beta;
};

template<typename T>
class ArgPack_trmm : public ArgPack<T>{
public:
  ArgPack_trmm(Order orderArg, Side sideArg, UpLo uploArg, Transpose transposeAArg,
    Diag diagArg, T alphaArg){
    this->method = Method::AblasTrmm;
    this->order = orderArg;
    this->side = sideArg;
    this->uplo = uploArg;
    this->transposeA = transposeAArg;
    this->diag = diagArg;
    this->alpha = alphaArg;
  }

  Order order;
  Side side;
  UpLo uplo;
  Transpose transposeA;
  Diag diag;
  T alpha;					// Added this constant
};

template<typename T>
class ArgPack_syrk : public ArgPack<T>{
public:
  ArgPack_syrk(Order orderArg, UpLo uploArg, Transpose transposeAArg, T alphaArg, T betaArg){
    this->method = Method::AblasSyrk;
    this->order = orderArg;
    this->uplo = uploArg;
    this->transposeA = transposeAArg;
    this->alpha = alphaArg;
    this->beta = betaArg;
  }

  Order order;
  UpLo uplo;
  Transpose transposeA;
  T alpha;					// Added this constant
  T beta;					// Added this constant
};
}

// Local includes -- include the possible BLAS libraries
#include "interface.h"


#endif /* BLASENGINE_H_ */
