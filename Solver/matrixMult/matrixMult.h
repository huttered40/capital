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
  matrixMult(void);
  multiply(void);

private:

  // a ton of different kinds of matrix multiplication goes here, called from multipy method, get these from the cholesky and the one(s) I need for QR
  // For now, these are very specific, maybe later I can generalize these to take care of all different layouts, storage, kinds of MM, rectangular MM, etc
};

#include "matrixMultImp.h"
#include "scalapackMatrixMult.h"

#endif /*MATRIX_MULT_H_*/
