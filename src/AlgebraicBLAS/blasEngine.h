/* Author: Edward Hutter */

#ifndef BLASENGINE_H_
#define BLASENGINE_H_


// System includes
#include <cblas.h>

// Local includes
#include "./../AlgebraicStructures/Matrix/MatrixStructure.h"

// We should use a host class and implement/define partial specialization classes

template<typename T, typename U,
  template<typename,typename,template<typename,typename,int> class> class MatrixStructureA,
  template<typename,typename,template<typename,typename,int> class> class MatrixStructureB,
  template<typename,typename,template<typename,typename,int> class> class MatrixStructureC>
class blasEngine;

template<typename T, typename U>
class blasEngine<T,U,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare>
{
  // Lets prevent any instances of this class from being created.
public:
  blasEngine() = delete;
  blasEngine(const blasEngine& rhs) = delete;
  blasEngine(blasEngine&& rhs) = delete;
  blasEngine<T,U,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare>& operator=(const blasEngine& rhs) = delete;
  blasEngine<T,U,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare>& operator=(blasEngine&& rhs) = delete;
  ~blasEngine() = delete;

  // Engine methods
  static void multiply(T* matrixA, T* matrixB, T* matrixC, U matrixAdimX, U matrixAdimY, U matrixBdimX, U matrixBdimZ, U matrixCdimY, U matrixCdimZ);

};

#include "blasEngine.hpp"

#endif /* BLASENGINE_H_ */
