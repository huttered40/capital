/* Author: Edward Hutter */

#ifndef SUMMA3DENGINE_H_
#define SUMMA3DENGINE_H_


// System includes

// Local includes
#include "./../../../../../AlgebraicStructures/Matrix/MatrixStructure.h"

// We should use a host class and implement/define partial specialization classes

template<typename T, typename U,
  template<typename,typename,template<typename,typename,int> class> class MatrixStructureA,
  template<typename,typename,template<typename,typename,int> class> class MatrixStructureB,
  template<typename,typename,template<typename,typename,int> class> class MatrixStructureC>
class Summa3DEngine;

template<typename T, typename U>
class Summa3DEngine<T,U,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare>
{
  // Lets prevent any instances of this class from being created.
public:
  Summa3DEngine() = delete;
  Summa3DEngine(const Summa3DEngine& rhs) = delete;
  Summa3DEngine(Summa3DEngine&& rhs) = delete;
  Summa3DEngine<T,U,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare>& operator=(const Summa3DEngine& rhs) = delete;
  Summa3DEngine<T,U,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare>& operator=(Summa3DEngine&& rhs) = delete;
  ~Summa3DEngine() = delete;

  // Engine methods
  //static void ...

private:
};

#include "Summa3DEngine.hpp"

#endif /* SUMMA3DENGINE_H_ */
