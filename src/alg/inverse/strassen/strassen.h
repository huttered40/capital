/* Author: Edward Hutter */

#ifndef INVERSE__STRASSEN_H_
#define INVERSE__STRASSEN_H_

#include "./../../alg.h"
#include "./../../matmult/summa/summa.h"
#include "./../../trsm/diaginvert/diaginvert.h"

namespace inverse{
class strassen{
public:
  template<typename MatrixType, typename CommType>
  static void invoke(MatrixType& matrix, CommType&& CommInfo, typename MatrixType::DimensionType NewtonDimension=1);

private:
  template<typename MatrixType, typename CommType>
  static void invert(MatrixType& matrixA, typename MatrixType::DimensionType localDimension, typename MatrixType::DimensionType trueLocalDimenion,
                           typename MatrixType::DimensionType bcDimension, typename MatrixType::DimensionType globalDimension, typename MatrixType::DimensionType trueGlobalDimension,
                           typename MatrixType::DimensionType matAstartX, typename MatrixType::DimensionType matAendX, typename MatrixType::DimensionType matAstartY,
                           typename MatrixType::DimensionType matAendY, CommType&& CommInfo);

  template<typename MatrixType, typename CommType>
  static void baseCase(MatrixType& matrixA, typename MatrixType::DimensionType localDimension, typename MatrixType::DimensionType trueLocalDimension,
                       typename MatrixType::DimensionType bcDimension, typename MatrixType::DimensionType NewtonDimension, typename MatrixType::DimensionType globalDimension,
                       typename MatrixType::DimensionType trueGlobalDimension,
                       typename MatrixType::DimensionType matAstartX, typename MatrixType::DimensionType matAendX, typename MatrixType::DimensionType matAstartY,
                       typename MatrixType::DimensionType matAendY, CommType&& CommInfo);

  template<typename MatrixType>
  static std::vector<typename MatrixType::ScalarType>
  blockedToCyclicTransformation(MatrixType& matA, typename MatrixType::DimensionType localDimension, typename MatrixType::DimensionType globalDimension,
                                typename MatrixType::DimensionType bcDimension, typename MatrixType::DimensionType matAstartX, typename MatrixType::DimensionType matAendX,
                                typename MatrixType::DimensionType matAstartY, typename MatrixType::DimensionType matAendY, size_t sliceDim, MPI_Comm slice2Dcomm, char dir);

  template<typename T, typename U>
  static void cyclicToLocalTransformation(std::vector<T>& storeT, std::vector<T>& storeTI, U localDimension, U globalDimension, U bcDimension, size_t sliceDim, size_t rankSlice, char dir);
};
}

#include "strassen.hpp"

#endif /* INVERSE__STRASSEN_H_ */
