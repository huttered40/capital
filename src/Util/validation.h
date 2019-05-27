/* Author: Edward Hutter */

#ifndef VALIDATION_H_
#define VALIDATION_H_

// This file is included by the validation headers, which each include the rest of the dependencies
#include "../Algorithms/MatrixMultiplication/MM3D/MM3D.h"

template<typename T, typename U>
class validator{
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
  static T validateResidualParallel(
                        Matrix<T,U,StructureArg1,Distribution>& matrixA,
                        Matrix<T,U,StructureArg2,Distribution>& matrixB,
                        Matrix<T,U,StructureArg3,Distribution>& matrixC,
                        char dir,
                        MPI_Comm commWorld,
                        std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D,
                        MPI_Comm columnAltComm,
			std::string& label
			);

  template< template<typename,typename,template<typename,typename,int> class> class StructureArg,
    template<typename,typename,int> class Distribution>
  static T validateOrthogonalityParallel(
                        Matrix<T,U,StructureArg,Distribution>& matrixQ,
                        MPI_Comm commWorld,
                        std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,int,int,int>& commInfo3D,
                        MPI_Comm columnAltComm,
			std::string& label
                      );
};

#include "validation.hpp"
#endif /*VALIDATION_H_*/
