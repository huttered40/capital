/* Author: Edward Hutter */

#ifndef CHOLESKYQR2_H_
#define CHOLESKYQR2_H_

// System includes


// Local includes


// Need template parameters for all 3 matrices (A,Q,R), as well as some other things, right?
template<typename T,typename U,
  template<...> structureA,		// Note: this vould be either rectangular or square. Hmmm. How will this work with CFR3D, which neds to be square?
  template<...> structureQ,
  template<...> structureR,		// Should probably be square, but maybe some idiot will want to use Triangular
class CholeskyQR2
{
public:

  void factor1D();
  void factor3D();
  void factorTunable();

};

#include "CholeskyQR2.hpp"

#endif /* CHOLESKYQR2_H_ */
