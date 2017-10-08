/* Author: Edward Hutter */

#ifndef TIMER_H_
#define TIMER_H_

// This file contains (for now) helper functions for dealing with timings for parallel algorithms

// System includes
#include <iostream>
#include <string>
#include <mpi.h>
#include <chrono>

class pTimer
{
public:
  // constructors can be left default for now  

  void printParallelTime(double tolerance, MPI_Comm comm, const std::string& str, int iteration = 0)
  {
    int myRank;
    double maxTime,minTime;
    MPI_Comm_rank(comm, &myRank);

    std::chrono::duration<double> elapsed_seconds = end-start;
    double count = elapsed_seconds.count();
    MPI_Allreduce(&count, &maxTime, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&count, &minTime, 1, MPI_DOUBLE, MPI_MIN, comm);

    if (std::abs(count-maxTime) <= tolerance)
    {
      std::cout << "Max time is on processor " << myRank << " for " << str << " on iteration " << iteration << " has a wall-clock time of " << maxTime << std::endl;
    }

    MPI_Barrier(comm);

    if (std::abs(count-minTime) <= tolerance)
    {
      std::cout << "Min time is on processor " << myRank << " for " << str << " on iteration " << iteration << " has a wall-clock time of " << maxTime << std::endl;
    }
  }

  void setStartTime()
  {
    this->start = std::chrono::system_clock::now();
  }

  void setEndTime()
  {
    this->end = std::chrono::system_clock::now();
  }

private:
  std::chrono::time_point<std::chrono::system_clock> start;
  std::chrono::time_point<std::chrono::system_clock> end;
};

#endif /* TIMER_H_ */
