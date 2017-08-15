/* Author: Edward Hutter */

// System includes
#include <iostream>

// Local includes
#include "Summa3D.h"

using namespace std;

int main(void)
{
  using MatrixTypeA = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;
  using MatrixTypeB = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;
  using MatrixTypeC = Matrix<double,int,MatrixStructureSquare,MatrixDistributerCyclic>;

  MatrixTypeA matA(4,4,4,4);
  MatrixTypeB matB(4,4,4,4);
  MatrixTypeC matC(4,4,4,4);

  Summa3D<double,int,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare>::Multiply(matA, matB, matC, 4, 4, 4);
  return 0;
}
