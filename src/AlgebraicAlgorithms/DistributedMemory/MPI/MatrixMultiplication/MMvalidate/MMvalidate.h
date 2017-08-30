/* Author: Edward Hutter */

#ifndef MMVALIDATE_H_
#define MMVALIDATE_H_

// System includes
#include <iostream>

// Local includes
#include "./../../../../../AlgebraicStructures/Matrix/Matrix.h"
#include "./../../../../../AlgebraicStructures/Matrix/MatrixSerializer.h"
#include "./../../../../../AlgebraicBLAS/blasEngine.h"


// These static methods will take the matrix in question, distributed in some fashion across the processors
//   and use them to calculate the residual or error.

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
            template<typename,typename, template<typename,typename,int> class> class Structure,
            template<typename,typename,int> class Distribution
          >
  static T validateLocal(
                        Matrix<T,U,Structure,Distribution>& matrixSol,
                        U dimensionX,
                        U dimensionY,
                        U dimensionZ,
                        int blasEngineInfo
                      );

  template<
            template<typename,typename, template<typename,typename,int> class> class Structure,
            template<typename,typename,int> class Distribution
          >
  static T validateLocal(
                        Matrix<T,U,Structure,Distribution>& matrixSol,
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
