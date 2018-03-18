/* Author: Edward Hutter */

#ifndef VALIDATION_H_
#define VALIDATION_H_

#include <vector>
#include <tuple>

#include "shared.h"
#include "util.h"
#include "./../Timer/Timer.h"
#include "../Matrix/Matrix.h"
#include "../AlgebraicAlgorithms/MatrixMultiplication/MM3D/MM3D.h"

template<typename T, typename U>
class validator
{
public:
  validator() = delete;
  ~validator();
  validator(const validator& rhs) = delete;
  validator(validator&& rhs) = delete;
  validator& operator=(const validator& rhs) = delete;
  validator& operator=(validator&& rhs) = delete;

  template< template<typename,typename,template<typename,typename,int> class> class StructureArg1,
    template<typename,typename,template<typename,typename,int> class> class StructureArg2,
    template<typename,typename,template<typename,typename,int> class> class StructureArg3,
    template<typename,typename,int> class Distribution>
  static void validateResidualParallel(
                        Matrix<T,U,StructureArg1,Distribution>& matrixA,
                        Matrix<T,U,StructureArg2,Distribution>& matrixB,
                        Matrix<T,U,StructureArg3,Distribution>& matrixC,
                        char dir,
                        MPI_Comm commWorld,
                        std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D,
                        MPI_Comm columnAltComm = MPI_COMM_WORLD
                      );

  template< template<typename,typename,template<typename,typename,int> class> class StructureArg,
    template<typename,typename,int> class Distribution>
  static void validateOrthogonalityParallel(
                        Matrix<T,U,StructureArg,Distribution>& matrixQ,
                        MPI_Comm commWorld,
                        std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D,
                        MPI_Comm columnAltComm = MPI_COMM_WORLD
                      );
};

#include "validation.hpp"
#endif /*VALIDATION_H_*/
