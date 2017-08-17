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

  using MatrixType1 = Matrix<double,int,MatrixStructureLowerTriangular,MatrixDistributerCyclic>;
  using MatrixType2 = Matrix<double,int,MatrixStructureLowerTriangular,MatrixDistributerCyclic>;
  using MatrixType3 = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;

/*
  MatrixType1 theMatrix(8,8,16,16);
  MatrixType2 theMatrixS(4,4,16,16);
  theMatrixS.DistributeRandom(0,0,2,2);
  MatrixType1 theMatrix2{theMatrix};
  const MatrixType1& theMatrix3 = theMatrix2;
  theMatrix2.DistributeRandom(1,1,2,2);
  MatrixType1 theMatrix4 = std::move(theMatrix2);
  theMatrix4.print();
  cout << "\n\n";
  theMatrixS.print();
  cout << "\n\n";
  theMatrix4.Serialize(theMatrixS);
  theMatrixS.print();
*/

  MatrixType3 theMatrix1(4,4,16,16);
  MatrixType3 theMatrix2(8,8,16,16);
  theMatrix2.DistributeSymmetric(1,1,2,2, true);
  theMatrix2.Serialize(theMatrix1,4,8,4,8);	// square to lower triangular
  theMatrix1.print();
  cout << "\n\n";
  theMatrix2.print();
  return 0;
}
