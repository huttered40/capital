/* Author: Edward Hutter */

#ifndef VALIDATION_H_
#define VALIDATION_H_

// This file is included by the validation headers, which each include the rest of the dependencies
#include "../Algorithms/MatrixMultiplication/MM3D/MM3D.h"

class validator{
public:
  template<typename MatrixAType, typename MatrixBType, typename MatrixCType>
  static typename MatrixAType::ScalarType
         validateResidualParallel(MatrixAType& matrixA, MatrixBType& matrixB, MatrixCType& matrixC,
                                  char dir, MPI_Comm commWorld, std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D, MPI_Comm columnAltComm, std::string& label);

  template<typename MatrixType>
  static typename MatrixType::ScalarType
         validateOrthogonalityParallel(MatrixType& matrixQ, MPI_Comm commWorld,
                                       std::tuple<MPI_Comm,MPI_Comm,MPI_Comm,MPI_Comm,size_t,size_t,size_t>& commInfo3D, MPI_Comm columnAltComm, std::string& label);
};

#include "validation.hpp"
#endif /*VALIDATION_H_*/
