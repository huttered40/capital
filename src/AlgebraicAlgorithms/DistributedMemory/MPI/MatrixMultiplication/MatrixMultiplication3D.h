/* Author: Edward Hutter */

#ifndef MATRIX_MULTIPLICATION_3D_H_
#define MATRIX_MULTIPLICATION_3D_H_

// System includes
#include <iostream>

// Local includes
#include "~/hutter2/ParallelAlgebraicAlgorithms/src/AlgebraicStructures/Matrix/Matrix.h"

/*
  We can implement square MM for now, but soon, we will need triangular MM
    and triangular matrices, as well as Square-Triangular Multiplication and Triangular-Square Multiplication
*/
template<typename T, typename U, class AllocatorA, class AllocatorB, class AllocatorC, class Distributer>
class MatrixMultiplication3D
{
  using MatrixTypeA = Matrix<T,U,AllocatorA,Distributer>
  using MatrixTypeB = Matrix<T,U,AllocatorB,Distributer>
  using MatrixTypeC = Matrix<T,U,AllocatorC,Distributer>

public:
  MatrixMultuplication3D();
  MatrixMultuplication3D(const MatrixMultiplication3D& rhs);
  MatrixMultuplication3D(MatrixMultiplication3D&& rhs);
  ~MatrixMultiplication3D();

  void Multiply(const MatrixTypeA& matrixA, const MatrixTypeB& matrixB, MatrixTypeC& matrixC);
  void Multiply(MatrixTypeA& matrixA, MatrixTypeB& matrixB, MatrixTypeC& matrixC);
  MatrixTypeC Multiply(const MatrixTypeA& matrixA, const MatrixTypeB& matrixB); // Matrix move constructor should be called in return statement
  MatrixTypeC Multiply(MatrixTypeA& matrixA, MatrixTypeB& matrixB);  // Matrix move constructor should be called in return statement

private:
  // Matrices for C=AB
  Matrix<T,U,AllocatorA,Distributer> _matrixA;
  Matrix<T,U,AllocatorB,Distributer> _matrixB;
  Matrix<T,U,AllocatorC,Distributer> _matrixC;

  // A - X x Y
  // B - Y x Z
  // C - X x Z
  U _dimensionX;
  U _dimensionY;
  U _dimensionZ;
};

#include "MatrixMultiplication3D.hpp"

#endif /*MATRIX_MULTIPLICATION_3D_H_ */
