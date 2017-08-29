/* Author: Edward Hutter */

#ifndef MMVALIDATE_H_
#define MMVALIDATE_H_

// System includes
#include <iostream>

// Local includes
#include "./../../../../../AlgebraicStructures/Matrix/Matrix.h"
#include "./../../../../../AlgebraicStructures/Matrix/MatrixSerializer.h"
#include "./../../../../../AlgebraicBLAS/blasEngine.h"

template<typename T, typename U, template<typename,typename> class blasEngine>
class MMvalidate
{
public:
  MMvalidate() = delete;
  ~MMvalidate() = delete;
  MMvalidate(const MMvalidate& rhs) = delete;
  MMvalidate(MMvalidate&& rhs) = delete;
  MMvalidate& operator=(const MMvalidate& rhs) = delete;
  MMvalidate& operator=(MMvalidate&& rhs) = delete;

  // This method does not depend on the structures. There are situations where have square structured matrices,
  //   but want to use trmm, for example, so disregard the structures in this entire class.
  
  // However, we still need to template this method because the method needs to be aware of the Matrix's template parameters
  //   in order to use it as a method argument.

  template<
            template<typename,typename, template<typename,typename,int> class> class StructureA,
            template<typename,typename, template<typename,typename,int> class> class StructureB,
            template<typename,typename, template<typename,typename,int> class> class StructureC,
            template<typename,typename,int> class Distribution
          >
  static void multiply(
                        Matrix<T,U,StructureA,Distribution>& matrixA,
                        Matrix<T,U,StructureB,Distribution>& matrixB,
                        Matrix<T,U,StructureC,Distribution>& matrixC,
                        U dimensionX,
                        U dimensionY,
                        U dimensionZ,
                        int blasEngineInfo
                      );

  template<
            template<typename,typename, template<typename,typename,int> class> class StructureA,
            template<typename,typename, template<typename,typename,int> class> class StructureB,
            template<typename,typename, template<typename,typename,int> class> class StructureC,
            template<typename,typename,int> class Distribution
          >
  static void multiply(
                        Matrix<T,U,StructureA,Distribution>& matrixA,
                        Matrix<T,U,StructureB,Distribution>& matrixB,
                        Matrix<T,U,StructureC,Distribution>& matrixC,
                        U matrixAcutXstart,
                        U matrixAcutXend,
                        U matrixAcutYstart,
                        U matrixAcutYend,
                        U matrixBcutXstart,
                        U matrixBcutXend,
                        U matrixBcutZstart,
                        U matrixBcutZend,
                        U matrixCcutYstart,
                        U matrixCcutYend,
                        U matrixCcutZstart,
                        U matrixCcutZend,
                        int blasEngineInfo
                      );

};

// Templated classes require method definition within the same unit as method declarations (correct wording?)
#include "MMvalidate.hpp"

#endif /* MMVALIDATE_H_ */
