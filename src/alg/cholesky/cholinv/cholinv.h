/* Author: Edward Hutter */

#ifndef CHOLESKY__CHOLINV_H_
#define CHOLESKY__CHOLINV_H_

#include "./../../alg.h"
#include "./../../matmult/summa/summa.h"
#include "./../../trsm/diaginvert/diaginvert.h"
#include "./../policies/cholinv/policy.h"

// Lets use partial template specialization
// So only declare the fully templated class
// Why not just use square? Because later on, I may want to experiment with LowerTriangular Structure.
// Also note, we do not need an extra template parameter for L-inverse. Presumably if the user wants L to be LowerTriangular, then he wants L-inverse
//   to be LowerTriangular as well

namespace cholesky{
template<class TrailingMatrixUpdateLocalCompPolicy = policy::cholinv::TrmmUpdate>
class cholinv{
public:
  template<typename MatrixAType, typename MatrixTIType, typename CommType>
  static std::pair<bool,std::vector<typename MatrixAType::DimensionType>>
         invoke(MatrixAType& matrixA, MatrixTIType& matrixTI, CommType&& CommInfo, typename MatrixAType::DimensionType inverseCutOffGlobalDimension, char dir);

private:
  template<typename MatrixAType, typename MatrixLIType, typename CommType>
  static void rFactorLower(MatrixAType& MatrixA, MatrixLIType& MatrixLI, typename MatrixAType::DimensionType localDimension, typename MatrixAType::DimensionType trueLocalDimenion,
                           typename MatrixAType::DimensionType bcDimension, typename MatrixAType::DimensionType globalDimension, typename MatrixAType::DimensionType trueGlobalDimension,
                           typename MatrixAType::DimensionType matAstartX, typename MatrixAType::DimensionType matAendX, typename MatrixAType::DimensionType matAstartY,
                           typename MatrixAType::DimensionType matAendY, typename MatrixAType::DimensionType matLIstartX, typename MatrixAType::DimensionType matLIendX,
                           typename MatrixAType::DimensionType matLIstartY, typename MatrixAType::DimensionType matLIendY,
                           CommType&& CommInfo, bool& isInversePath, std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                           typename MatrixAType::DimensionType inverseCutoffGlobalDimension);

  template<typename MatrixAType, typename MatrixRIType, typename CommType>
  static void rFactorUpper(MatrixAType& MatrixA, MatrixRIType& MatrixRI, typename MatrixAType::DimensionType localDimension, typename MatrixAType::DimensionType trueLocalDimension,
                           typename MatrixAType::DimensionType bcDimension, typename MatrixAType::DimensionType globalDimension, typename MatrixAType::DimensionType trueGlobalDimension,
                           typename MatrixAType::DimensionType matAstartX, typename MatrixAType::DimensionType matAendX, typename MatrixAType::DimensionType matAstartY,
                           typename MatrixAType::DimensionType matAendY, typename MatrixAType::DimensionType matRIstartX, typename MatrixAType::DimensionType matRIendX,
                           typename MatrixAType::DimensionType matRIstartY, typename MatrixAType::DimensionType matRIendY,
                           CommType&& CommInfo, bool& isInversePath, std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                           typename MatrixAType::DimensionType inverseCutoffGlobalDimension);

  
  template<typename MatrixAType, typename MatrixIType, typename CommType>
  static void baseCase(MatrixAType& MatrixA, MatrixIType& MatrixLI, typename MatrixAType::DimensionType localDimension, typename MatrixAType::DimensionType trueLocalDimension,
                       typename MatrixAType::DimensionType bcDimension, typename MatrixAType::DimensionType globalDimension, typename MatrixAType::DimensionType trueGlobalDimension,
                       typename MatrixAType::DimensionType matAstartX, typename MatrixAType::DimensionType matAendX, typename MatrixAType::DimensionType matAstartY,
                       typename MatrixAType::DimensionType matAendY, typename MatrixAType::DimensionType matIstartX, typename MatrixAType::DimensionType matIendX,
                       typename MatrixAType::DimensionType matIstartY, typename MatrixAType::DimensionType matIendY, CommType&& CommInfo,
                       bool& isInversePath, std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                       typename MatrixAType::DimensionType inverseCutoffGlobalDimension, char dir);

  template<typename MatrixType>
  static std::vector<typename MatrixType::ScalarType>
  blockedToCyclicTransformation(MatrixType& MatrixA, typename MatrixType::DimensionType localDimension, typename MatrixType::DimensionType globalDimension,
                                typename MatrixType::DimensionType bcDimension, typename MatrixType::DimensionType matAstartX, typename MatrixType::DimensionType matAendX,
                                typename MatrixType::DimensionType matAstartY, typename MatrixType::DimensionType matAendY, size_t sliceDim, MPI_Comm slice2Dcomm, char dir);

  template<typename T, typename U>
  static void cyclicToLocalTransformation(std::vector<T>& storeT, std::vector<T>& storeTI, U localDimension, U globalDimension, U bcDimension, size_t sliceDim, size_t rankSlice, char dir);

  template<typename U>
  static inline void updateInversePath(U inverseCutoffGlobalDimension, U globalDimension, bool& isInversePath, std::vector<U>& baseCaseDimList, U localDimension);
};
}

#include "cholinv.hpp"

#endif /* CHOLESKY__CHOLINV_H_ */
