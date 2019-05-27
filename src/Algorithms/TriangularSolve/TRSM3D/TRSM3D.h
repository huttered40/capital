/* Author: Edward Hutter */

#ifndef TRSM3D_H_
#define TRSM3D_H_

#include "./../../Algorithms.h"
#include "./../../MatrixMultiplication/MM3D/MM3D.h"

// Lets use partial template specialization
// So only declare the fully templated class
// Why not just use square? Because later on, I may want to experiment with LowerTriangular Structure.
// Also note, we do not need an extra template parameter for L-inverse. Presumably if the user wants L to be LowerTriangular, then he wants L-inverse
//   to be LowerTriangular as well

template<typename T, typename U, typename OffloadType = OffloadEachGemm>
class TRSM3D
{
public:
  // Prevent instantiation of this class
  TRSM3D() = delete;
  TRSM3D(const TRSM3D& rhs) = delete;
  TRSM3D(TRSM3D&& rhs) = delete;
  TRSM3D& operator=(const TRSM3D& rhs) = delete;
  TRSM3D& operator=(TRSM3D&& rhs) = delete;

  template<
    template<typename,typename, template<typename,typename,int> class> class StructureArg,
    template<typename,typename, template<typename,typename,int> class> class StructureTriangularArg,
    template<typename,typename,int> class Distribution
  >
  static void iSolveLowerLeft(
                       Matrix<T,U,StructureArg,Distribution>& matrixA,
                       Matrix<T,U,StructureTriangularArg,Distribution>& matrixL,
                       Matrix<T,U,StructureTriangularArg,Distribution>& matrixLI,
                       std::vector<U>& baseCaseDimList,
                       blasEngineArgumentPackage_gemm<T>& gemmPackage,
                       blasEngineArgumentPackage_trmm<T>& trmmPackage,
                       MPI_Comm commWorld,
                       std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D
                     );

  template<
    template<typename,typename, template<typename,typename,int> class> class StructureArg,
    template<typename,typename, template<typename,typename,int> class> class StructureTriangularArg,
    template<typename,typename,int> class Distribution
  >
  static void iSolveUpperLeft(
                       Matrix<T,U,StructureArg,Distribution>& matrixA,
                       Matrix<T,U,StructureTriangularArg,Distribution>& matrixU,
                       Matrix<T,U,StructureTriangularArg,Distribution>& matrixUI,
                       std::vector<U>& baseCaseDimList,
                       blasEngineArgumentPackage_gemm<T>& gemmPackage,
                       blasEngineArgumentPackage_trmm<T>& trmmPackage,
                       MPI_Comm commWorld,
                       std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D
                     );
  
  template<
    template<typename,typename, template<typename,typename,int> class> class StructureArg,
    template<typename,typename, template<typename,typename,int> class> class StructureTriangularArg,
    template<typename,typename,int> class Distribution
  >
  static void iSolveLowerRight(
                       Matrix<T,U,StructureTriangularArg,Distribution>& matrixL,
                       Matrix<T,U,StructureTriangularArg,Distribution>& matrixLI,
                       Matrix<T,U,StructureArg,Distribution>& matrixA,
                       std::vector<U>& baseCaseDimList,
                       blasEngineArgumentPackage_gemm<T>& gemmPackage,
                       blasEngineArgumentPackage_trmm<T>& trmmPackage,
                       MPI_Comm commWorld,
                       std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D
                     );

  template<
    template<typename,typename, template<typename,typename,int> class> class StructureArg,
    template<typename,typename, template<typename,typename,int> class> class StructureTriangularArg,
    template<typename,typename,int> class Distribution
  >
  static void iSolveUpperRight(
                       Matrix<T,U,StructureTriangularArg,Distribution>& matrixU,
                       Matrix<T,U,StructureTriangularArg,Distribution>& matrixUI,
                       Matrix<T,U,StructureArg,Distribution>& matrixA,
                       std::vector<U>& baseCaseDimList,
                       blasEngineArgumentPackage_gemm<T>& gemmPackage,
                       blasEngineArgumentPackage_trmm<T>& trmmPackage,
                       MPI_Comm commWorld,
                       std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D
                     );
};

#include "TRSM3D.hpp"

#endif /* TRSM3D_H_ */
