/* Author: Edward Hutter */

#ifndef TIMER_H_
#define TIMER_H_

// This file contains (for now) helper functions for dealing with timings for parallel algorithms

// System includes
#include <iostream>
#include <string>
#include <mpi.h>
#include <chrono>
#include <climits>
#include <algorithm>
#include <map>
#include <vector>

// Local includes
#include "../Util/shared.h"
#include "CTFtimer.cxx"

class pTimer
{
public:
  pTimer() {/*this->count = 0; this->totalMin = std::numeric_limits<double>::max(); this->totalMax = 0;*/}

  size_t setStartTime(const std::string& funcName)
  {
#ifdef CTFTIMER
    TAU_FSTART(funcName);
    return 0;
#endif /*CTFTIMER*/
  }

  void setEndTime(const std::string& funcName, size_t index)
  {
#ifdef CTFTIMER
    TAU_FSTOP(funcName);
#endif /*CTFTIMER*/
  }
};

#endif /* TIMER_H_ */
