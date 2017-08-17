/* Author: Edward Hutter */

#ifndef CFR3D_H_
#define CFR3D_H_

// System includes

// Local includes


// Lets use partial template specialization
// So only declare the fully templated class
template<typename T, typename U,
  template<typename,typename,template<typename,typename,class> class, template<typename,typename,int> class> class MatrixA,
  template<typename,typename,template<typename,typename,class> class, template<typename,typename,int> class> class MatrixU>
class CFR3D;

// Partial specialization of CFR3D algorithm class
template<typename T, typename U>
class CFR3D<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular>
{
public:
  // Prevent instantiation of this class
  CFR3D() = delete;
  CFR3D(const CFR3D& rhs) = delete;
  CFR3D(CFR3D&& rhs) = delete;
  CFR3D<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular>& operator=(const CFR3D& rhs) = delete;
  CFR3D<T,U,MatrixStructureSquare,MatrixStructureLowerTriangular>& operator=(CFR3D&& rhs) = delete;

  static void Factor(MatrixStructureSquare matrixA, MatrixStructureLowerTriangular matrixL, U dimension, MPI_Comm commWorld);

};

#include "CFR3D.hpp"

#endif /* CFR3D_H_ */
