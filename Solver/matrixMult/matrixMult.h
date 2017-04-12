/*
	Author: Edward Hutter
*/

#ifndef MATRIX_MULT_H_
#define MATRIX_MULT_H_

#include <iostream>
#include <vector>

using namespace std;

class matrixMult
{
public:
  matrixMult(void);		// I should set up the right P-grid that I need to use in here so that there is no confusion
  multiply(void);

private:
  // a ton of different kinds of matrix multiplication goes here, called from multipy method, get these from the cholesky and the one(s) I need for QR
  // For now, these are very specific, maybe later I can generalize these to take care of all different layouts, storage, kinds of MM, rectangular MM, etc

  void multiply1(void);
  void multiply2(void);
  void multiply3(void);
  void multiply4(void);
  void multiply5(void);

  // The methods above do not need arguments because i will "move" the arguments passed into multiply() into my member variables and then will call one
  // of these methods, and then will "move" them back before the multiply() routine ends.
};

#include "matrixMultImp.h"
#include "scalapackMatrixMult.h"

#endif /*MATRIX_MULT_H_*/
