/* Author: Edward Hutter */

#ifndef CHOLESKY__CHOLINV_H_
#define CHOLESKY__CHOLINV_H_

#include "./../../alg.h"
#include "./../../matmult/summa/summa.h"
#include "./../../trsm/diaginvert/diaginvert.h"

// Lets use partial template specialization
// So only declare the fully templated class
// Why not just use square? Because later on, I may want to experiment with LowerTriangular Structure.
// Also note, we do not need an extra template parameter for L-inverse. Presumably if the user wants L to be LowerTriangular, then he wants L-inverse
//   to be LowerTriangular as well

namespace cholesky{
class cholinv{
public:
  // cholinv is not parameterized as its not dependent on any lower-level algorithmic type
  class pack{
  public:
    pack(const pack& p) : inv_cut_off_dim(p.inv_cut_off_dim), dir(p.dir) {}
    pack(pack&& p) : inv_cut_off_dim(std::move(p.inv_cut_off_dim)), dir(std::move(p.dir)) {}
    pack(int64_t inv_cut_off_dim, char dir) : inv_cut_off_dim(inv_cut_off_dim), dir(dir) {}
    int64_t inv_cut_off_dim;
    char dir;
  };

  template<typename MatrixAType, typename MatrixTIType, typename ArgType, typename CommType>
  static std::pair<bool,std::vector<typename MatrixAType::DimensionType>>
         invoke(MatrixAType& matrixA, MatrixTIType& matrixTI, ArgType&& args, CommType&& CommInfo);

private:
  template<typename MatrixAType, typename MatrixLIType, typename BaseCaseMatrixType, typename CommType>
  static void factor_lower(MatrixAType& matrixA, MatrixLIType& matrixLI, BaseCaseMatrixType& matrix_base_case,
                           std::vector<typename MatrixAType::ScalarType>& blocked_data, std::vector<typename MatrixAType::ScalarType>& cyclic_data,
                           typename MatrixAType::DimensionType localDimension, typename MatrixAType::DimensionType trueLocalDimenion,
                           typename MatrixAType::DimensionType bcDimension, typename MatrixAType::DimensionType globalDimension, typename MatrixAType::DimensionType trueGlobalDimension,
                           typename MatrixAType::DimensionType matAstartX, typename MatrixAType::DimensionType matAendX, typename MatrixAType::DimensionType matAstartY,
                           typename MatrixAType::DimensionType matAendY, typename MatrixAType::DimensionType matLIstartX, typename MatrixAType::DimensionType matLIendX,
                           typename MatrixAType::DimensionType matLIstartY, typename MatrixAType::DimensionType matLIendY,
                           CommType&& CommInfo, bool& isInversePath, std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                           typename MatrixAType::DimensionType inverseCutoffGlobalDimension);

  template<typename MatrixAType, typename MatrixRIType, typename BaseCaseMatrixType, typename CommType>
  static void factor_upper(MatrixAType& matrixA, MatrixRIType& matrixRI, BaseCaseMatrixType& matrix_base_case,
                           std::vector<typename MatrixAType::ScalarType>& blocked_data, std::vector<typename MatrixAType::ScalarType>& cyclic_data,
                           typename MatrixAType::DimensionType localDimension, typename MatrixAType::DimensionType trueLocalDimension,
                           typename MatrixAType::DimensionType bcDimension, typename MatrixAType::DimensionType globalDimension, typename MatrixAType::DimensionType trueGlobalDimension,
                           typename MatrixAType::DimensionType matAstartX, typename MatrixAType::DimensionType matAendX, typename MatrixAType::DimensionType matAstartY,
                           typename MatrixAType::DimensionType matAendY, typename MatrixAType::DimensionType matRIstartX, typename MatrixAType::DimensionType matRIendX,
                           typename MatrixAType::DimensionType matRIstartY, typename MatrixAType::DimensionType matRIendY,
                           CommType&& CommInfo, bool& isInversePath, std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                           typename MatrixAType::DimensionType inverseCutoffGlobalDimension);

  template<typename MatrixAType, typename MatrixIType, typename BaseCaseMatrixType, typename CommType>
  static void baseCase(MatrixAType& matrixA, MatrixIType& matrixLI, BaseCaseMatrixType& matrix_base_case,
                       std::vector<typename MatrixAType::ScalarType>& blocked_data, std::vector<typename MatrixAType::ScalarType>& cyclic_data,
                       typename MatrixAType::DimensionType localDimension, typename MatrixAType::DimensionType trueLocalDimension,
                       typename MatrixAType::DimensionType bcDimension, typename MatrixAType::DimensionType globalDimension, typename MatrixAType::DimensionType trueGlobalDimension,
                       typename MatrixAType::DimensionType matAstartX, typename MatrixAType::DimensionType matAendX, typename MatrixAType::DimensionType matAstartY,
                       typename MatrixAType::DimensionType matAendY, typename MatrixAType::DimensionType matIstartX, typename MatrixAType::DimensionType matIendX,
                       typename MatrixAType::DimensionType matIstartY, typename MatrixAType::DimensionType matIendY, CommType&& CommInfo,
                       bool& isInversePath, std::vector<typename MatrixAType::DimensionType>& baseCaseDimList,
                       typename MatrixAType::DimensionType inverseCutoffGlobalDimension, char dir);

  template<typename MatrixType, typename BaseCaseMatrixType, typename CommType>
  static void aggregate(MatrixType& matA, BaseCaseMatrixType& matrix_base_case, std::vector<typename MatrixType::ScalarType>& blocked_data,
                        std::vector<typename MatrixType::ScalarType>& cyclic_data, typename MatrixType::DimensionType localDimension, typename MatrixType::DimensionType globalDimension,
                        typename MatrixType::DimensionType bcDimension, typename MatrixType::DimensionType matAstartX, typename MatrixType::DimensionType matAendX,
                        typename MatrixType::DimensionType matAstartY, typename MatrixType::DimensionType matAendY, CommType&& CommInfo, char dir);

  template<typename T, typename U>
  static void cyclicToLocalTransformation(std::vector<T>& storeT, std::vector<T>& storeTI, U localDimension, U globalDimension, U bcDimension, int64_t sliceDim, int64_t rankSlice, char dir);

  template<typename U>
  static inline void updateInversePath(U inverseCutoffGlobalDimension, U globalDimension, bool& isInversePath, std::vector<U>& baseCaseDimList, U localDimension);
};
}

#include "cholinv.hpp"

#endif /* CHOLESKY__CHOLINV_H_ */
