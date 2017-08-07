/* Author : Edward Hutter */

// System includes
#include <iostream>

// Local includes
#include "Matrix.h"

using std::cout;

int main(void)
{
  Matrix<double,int> theMatrix;
  Matrix<double,int> theMatrix2{theMatrix};
  const Matrix<double,int>& theMatrix3 = theMatrix2;
  Matrix<double,int> theMatrix4 = std::move(theMatrix2);
  return 0;
}
