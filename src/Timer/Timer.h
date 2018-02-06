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

class pTimer
{
public:
  pTimer() {this->count = 0; this->totalMin = std::numeric_limits<double>::max(); this->totalMax = 0;}

  void printParallelTime(double tolerance, MPI_Comm comm, const std::string& str, int iteration = 0)
  {
    int myRank,numPEs;
    double maxTime,minTime,avgTime;
    MPI_Comm_rank(comm, &myRank);
    MPI_Comm_size(comm, &numPEs);

    std::chrono::duration<double> elapsed_seconds = this->end - this->start;
    double count = elapsed_seconds.count();
    MPI_Allreduce(&count, &maxTime, 1, MPI_DOUBLE, MPI_MAX, comm);
    MPI_Allreduce(&count, &minTime, 1, MPI_DOUBLE, MPI_MIN, comm);
    MPI_Allreduce(&count, &avgTime, 1, MPI_DOUBLE, MPI_SUM, comm);
    avgTime /= numPEs; 

    this->count++;
    this->totalMin = std::min(this->totalMin, maxTime);
    this->totalMax = std::max(this->totalMax, maxTime);
    this->totalAvg += maxTime;

    /*
    if (std::abs(count-maxTime) <= tolerance)
    {
      std::cout << "Max time is on processor " << myRank << " for " << str << " on iteration " << iteration << " has a wall-clock time of " << maxTime << std::endl;
    }

    MPI_Barrier(comm);

    if (std::abs(count-minTime) <= tolerance)
    {
      std::cout << "Min time is on processor " << myRank << " for " << str << " on iteration " << iteration << " has a wall-clock time of " << maxTime << std::endl;
    }
    */
    if (myRank == 0)
    {
      std::cout << "Max time for " << str << " on iteration " << iteration << " has a wall-clock time of " << maxTime << std::endl;
      std::cout << "Min time for " << str << " on iteration " << iteration << " has a wall-clock time of " << minTime << std::endl;
      std::cout << "Average time for " << str << " on iteration " << iteration << " has a wall-clock time of " << avgTime << std::endl;
    }
    MPI_Barrier(comm);
  }

  void printRunStats(MPI_Comm comm, const std::string& str)
  {
    int myRank,numPEs;
    MPI_Comm_rank(comm, &myRank);

    this->totalAvg /= this->count;
    if (myRank == 0)
    {
      std::cout << "Max time for " << str << " over all runs has a wall-clock time of " << this->totalMax << std::endl;
      std::cout << "Min time for " << str << " over all runs has a wall-clock time of " << this->totalMin << std::endl;
      std::cout << "Average time for " << str << " over all runs has a wall-clock time of " << this->totalAvg << std::endl;
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
  double totalMin, totalAvg, totalMax;                        // These are for keeping track of data from many runs
  int count;
};

#endif /* TIMER_H_ */
