/* Author: Edward Hutter */

#ifndef TIMECONTROLLER_H_
#define TIMECONTROLLER_H_

#include <tuple>
#include <map>
#include <string>
#include <vector>

#include "./../Util/shared.h"
#include "./../Timer/Timer.h"
#include "./../Util/util.h"
#include "./../AlgebraicAlgorithms/CholeskyFactorization/CFR3D/CFR3D.h"
#include "./../AlgebraicAlgorithms/MatrixMultiplication/MM3D/MM3D.h"
#include "./../AlgebraicAlgorithms/TriangularSolve/TRSM3D/TRSM3D.h"
#include "./../AlgebraicAlgorithms/QRFactorization/CholeskyQR2/CholeskyQR2.h"
#include "./../AlgebraicBLAS/cblasEngine.h"
#include "./../Matrix/Matrix.h"
#include "./../Matrix/MatrixSerializer.h"
#include "./../Matrix/MatrixDistributer.h"
#include "./../Matrix/MatrixStructure.h"

// This method should and need only be called on a single rank
template<
  typename T,
  typename U,
  template<typename,typename,template<typename,typename,int> class> class Structure,
  template<typename, typename,int> class Distributer,
  template<typename,typename> class blasEngine
>
class TimeController
{
public:
  // Prevent any instantiation of this class.

  void initTimers();
  void displayResults();

private:
  void ConvertTimerInfo(pTimer& timer);

  std::map<std::string,size_t> saveIndices;
  std::vector<std::tuple<std::string,int,double> > timerInfo;
};

#include "TimeController.hpp"

#endif /*TIMECONTROLLER_H_*/
