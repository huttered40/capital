/* Author : Edward Hutter */

// System includes
#include <iostream>

// Local includes
#include "Matrix.h"

using std::cout;

int main(void)
{
  // When we instantiate a template class, we must provide the template parameters so that the compiler can create the corresponding class implementation
  //   from the blueprint we provided.

  // Matrix<double,int> theMatrix();        -> should be prevented by the compiler.
  Matrix<double,int> theMatrix(3.6);
  Matrix<double,int> theMatrix2{theMatrix};
  const Matrix<double,int>& theMatrix3 = theMatrix2;
  Matrix<double,int> theMatrix4 = std::move(theMatrix2);
  return 0;
}
