/* Author : Edward Hutter */

// System includes
#include <iostream>

// Local includes
#include "MatrixSerializer.h"
#include "Matrix.h"

using std::cout;
using std::endl;

int main(void)
{
  // When we instantiate a template class, we must provide the template parameters so that the compiler can create the corresponding class implementation
  //   from the blueprint we provided.

  using MatrixType1 = Matrix<double,int,MatrixStructureLowerTriangular,MatrixDistributerCyclic>;
  using MatrixType2 = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;
  using MatrixType3 = Matrix<double,int,MatrixStructureUpperTriangular,MatrixDistributerCyclic>;

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


  MatrixType1 theMatrix1(8,8,16,16);
  MatrixType1 theMatrix2(2,2,16,16);
  theMatrix2.DistributeRandom(0,0,2,2);
  Serializer<double,int,MatrixStructureLowerTriangular,MatrixStructureLowerTriangular>::Serialize(theMatrix1,theMatrix2,0,2,6,8,true);
  theMatrix1.print();
  cout << "\n\n";
  theMatrix2.print();


/*
  MatrixType1 theMatrix1(8,8,16,16);
  MatrixType2 theMatrix2(8,8,16,16);
  MatrixType3 theMatrix3(8,8,16,16);
  theMatrix1.DistributeRandom(1,0,2,2);
  theMatrix2.DistributeRandom(1,0,2,2);
  theMatrix3.DistributeRandom(1,0,2,2);
  theMatrix1.print();
  cout << "\n\n";
  theMatrix2.print();
  cout << "\n\n";
  theMatrix3.print();
*/

  return 0;
}
