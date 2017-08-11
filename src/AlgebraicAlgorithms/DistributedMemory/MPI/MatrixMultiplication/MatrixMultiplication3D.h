/* Author: Edward Hutter */

#ifndef MATRIX_MULTIPLICATION_3D_H_
#define MATRIX_MULTIPLICATION_3D_H_

// System includes
#include <iostream>

// Local includes
#include "~/hutter2/ParallelAlgebraicAlgorithms/src/AlgebraicStructures/Matrix/Matrix.h"

/*
  We can implement square MM for now, but soon, we will need triangular MM
    and triangular matrices.
*/
template<typename T, typename U, class Allocator, class Distributer>
class MatrixMultiplication3D
{
public:
  MatrixMultuplication3D();
  MatrixMultuplication3D(const MatrixMultiplication3D& rhs);
  MatrixMultuplication3D(MatrixMultiplication3D&& rhs);
  ~MatrixMultiplication3D();

private:
  // Matrices for C=AB
  Matrix<T,U,Allocator,Distributer> _matrixA;
  Matrix<T,U,Allocator,Distributer> _matrixB;
  Matrix<T,U,Allocator,Distributer> _matrixC;

  // A - X x Y
  // B - Y x Z
  // C - X x Z
  U _dimensionX;
  U _dimensionY;
  U _dimensionZ;
};

#endif /*MATRIX_MULTIPLICATION_3D_H_ */
