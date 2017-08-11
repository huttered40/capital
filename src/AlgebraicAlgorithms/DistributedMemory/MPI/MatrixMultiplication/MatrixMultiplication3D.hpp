/* Author: Edward Hutter */


  MatrixMultuplication3D();
  MatrixMultuplication3D(const MatrixMultiplication3D& rhs);
  MatrixMultuplication3D(MatrixMultiplication3D&& rhs);
  ~MatrixMultiplication3D();

  void Multiply(const MatrixTypeA& matrixA, const MatrixTypeB& matrixB, MatrixTypeC& matrixC);
  void Multiply(MatrixTypeA& matrixA, MatrixTypeB& matrixB, MatrixTypeC& matrixC);
  MatrixTypeC Multiply(const MatrixTypeA& matrixA, const MatrixTypeB& matrixB); // Matrix move constructor should be called in return statement
  MatrixTypeC Multiply(MatrixTypeA& matrixA, MatrixTypeB& matrixB);  // Matrix move constructor should be called in return statement
