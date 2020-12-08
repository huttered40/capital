/* Author: Edward Hutter */

#ifndef LAPACK_INTERFACE_H_
#define LAPACK_INTERFACE_H_

// Local includes
#include "./../util/shared.h"

namespace lapack{

// ************************************************************************************************************************************************************
template<typename T>
class LType{};

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
