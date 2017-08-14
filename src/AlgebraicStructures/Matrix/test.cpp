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

  using MatrixType1 = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;
  using MatrixType2 = Matrix<double,int,MatrixStructureUpperTriangular,MatrixDistributerCyclic>;
  MatrixType1 theMatrix(8,8,16,16);
  MatrixType2 theMatrixS(8,8,16,16);
  theMatrixS.Distribute(0,0,2,2);
  MatrixType1 theMatrix2{theMatrix};
  const MatrixType1& theMatrix3 = theMatrix2;
  theMatrix2.Distribute(1,1,2,2);
  MatrixType1 theMatrix4 = std::move(theMatrix2);
  theMatrix4.print();
  cout << "\n\n";
  theMatrixS.print();

  theMatrix4.Serialize(theMatrixS);
  return 0;
}
