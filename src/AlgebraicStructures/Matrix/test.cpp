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

  using MatrixType1 = Matrix<double,int,MatrixAllocatorContiguous<double,int>,MatrixDistributerCyclic<double,int,std::vector<double*>>>;
  MatrixType1 theMatrix(5,5,10,10);
  MatrixType1 theMatrix2{theMatrix};
  const MatrixType1& theMatrix3 = theMatrix2;
  theMatrix2.Distribute(0,0,4,4);
  MatrixType1 theMatrix4 = std::move(theMatrix2);
  theMatrix4.print();
  return 0;
}
