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

static bool compareFunctionInfo(const std::pair<std::string,double>& info1, const std::pair<std::string,double>& info2)
{
  return info1.second < info2.second;
}

class pTimer
{
public:
  pTimer() {/*this->count = 0; this->totalMin = std::numeric_limits<double>::max(); this->totalMax = 0;*/}

  void clear()
  {
    this->table.clear();
  }

  void finalize(MPI_Comm commWorld)
  {
    std::vector<double> functionTimes;
    std::vector<std::string> functionNames;
    // Obtain max time spent in each function
    // Number of such calls should be the same accross all processes
    for (auto mapIter = this->table.begin(); mapIter != this->table.end(); mapIter++)
    {
      functionTimes.push_back(mapIter->second.second);
      functionNames.push_back(mapIter->first);    // Yes, I know I could have used std::move or emplace_back.
    }

    int rank;
    MPI_Comm_rank(commWorld, &rank);
    MPI_Allreduce(MPI_IN_PLACE, &functionTimes[0], functionTimes.size(), MPI_DOUBLE, MPI_MAX, commWorld);

    // Now put into a single buffer that can be sorted
    std::vector<std::pair<std::string,double> > functionInfo(functionNames.size());
    for (int i=0; i<functionInfo.size(); i++)
    {
      functionInfo[i] = std::make_pair(functionNames[i], functionTimes[i]);
    }

    std::sort(functionInfo.begin(), functionInfo.end(), compareFunctionInfo);

    // Now lets display the results
    if (rank == 0)
    {
      std::cout << "\n\n\n\n";
      for (size_t i=0; i<functionInfo.size(); i++)
      {
        std::cout << "      " << functionInfo[i].first << " -- Number of calls: " << this->table[functionInfo[i].first].first.size() << ", total time: " << functionInfo[i].second << std::endl;
        std::cout << "\n";
      }
      std::cout << "\n\n\n\n";
    }
    this->clear();
  }


  void printParallelTime(double tolerance, MPI_Comm comm, const std::string& str, int iteration = 0)
  {
/*
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

    if (myRank == 0)
    {
      std::cout << "Max time for " << str << " on iteration " << iteration << " has a wall-clock time of " << maxTime << std::endl;
      std::cout << "Min time for " << str << " on iteration " << iteration << " has a wall-clock time of " << minTime << std::endl;
      std::cout << "Average time for " << str << " on iteration " << iteration << " has a wall-clock time of " << avgTime << std::endl;
    }
    MPI_Barrier(comm);
*/
  }

  void printRunStats(MPI_Comm comm, const std::string& str)
  {
/*
    int myRank,numPEs;
    MPI_Comm_rank(comm, &myRank);

    this->totalAvg /= this->count;
    if (myRank == 0)
    {
      std::cout << "Max time for " << str << " over all runs has a wall-clock time of " << this->totalMax << std::endl;
      std::cout << "Min time for " << str << " over all runs has a wall-clock time of " << this->totalMin << std::endl;
      std::cout << "Average time for " << str << " over all runs has a wall-clock time of " << this->totalAvg << std::endl;
    }
*/
  }

  size_t setStartTime(const std::string& funcName)
  {
    if (table.find(funcName) != table.end())
    {
      this->table[funcName].first.push_back(MPI_Wtime());
    }
    else
    {// Difference between this and above is the necessary initialization of double timer
      this->table[funcName].first.push_back(MPI_Wtime());
      this->table[funcName].second = 0;
    }
    return table[funcName].first.size()-1;
  }

  void setEndTime(const std::string& funcName, size_t index)
  {
    double temp = MPI_Wtime();
    double numSec = temp - table[funcName].first[index];
    this->table[funcName].second += numSec;
  }

private:
  std::map<std::string,std::pair<std::vector<double>,double> > table;
};

#endif /* TIMER_H_ */
