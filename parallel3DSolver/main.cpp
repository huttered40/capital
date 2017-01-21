/*

	Program to start the Recursive 3D LU Factorization Solver
	Author: Edward Hutter
*/

/* Local Includes */
#include "solver.h"

/* System Includes */
#include <ctime>	//clock

using namespace std;

int main(int argc, char *argv[])
{
  int rank,size,provided;
  bool tracker=false;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_SINGLE,&provided);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  solver<double> mySolver(rank,size,3,4096);		// 8 might not be the best matrix size to use

  // Bug check here : Why isnt there a call to startUp and collectDataCyclic methods before lapackTest??
  if (size == 1)
  {
    clock_t start;
    double duration;
    start = clock(); 	// start time
    mySolver.lapackTest(16);
    duration = (clock() - start) / (double)CLOCKS_PER_SEC;

    cout << "Time - " << duration << endl;
    MPI_Finalize();
    return 0; 
  }

  mySolver.startUp(tracker);
  if (tracker)
  {
    cout << "Number of processors does not fit\n";
    MPI_Finalize();
    return 0;
  }
  mySolver.collectDataCyclic();

  // So I start my timings after the data is distributed, which involved no communication
  clock_t start;
  double duration;
  start = clock(); 	// start time

  //mySolver.solveScalapack();	// run benchmark
  mySolver.solve();		// run algorithm

  duration = (clock() - start) / (double)CLOCKS_PER_SEC;

  // I want the average of each process's runtime, so I can use a reduction
  double totalSum=0.;
  double totalMax=0.;  // will only be valid on root process
  double durationCopy = duration;
  MPI_Reduce(&duration,&totalSum,1,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD);
  MPI_Reduce(&durationCopy,&totalMax,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);

  if (rank == 0)
  {
    cout << "Average algorithm runtime - " << totalSum/size << endl;
    cout << "Max algorithm runtime - " << totalMax << endl;
  }
  // If this works, then I can print out the data to see if its correct
  //mySolver.printL();

  MPI_Finalize();
  return 0;
}
