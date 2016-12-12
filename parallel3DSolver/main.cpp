/*

	Program to start the Recursive 3D LU Factorization Solver
	Author: Edward Hutter
*/

/* Local Includes */
#include "solver.h"

/* System Includes */


using namespace std;

int main(int argc, char *argv[])
{
  int rank,size,provided;
  bool tracker=false;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_SINGLE,&provided);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  solver<double> mySolver(rank,size,3,16);		// 8 might not be the best matrix size to use
  mySolver.startUp(tracker);
  if (tracker)
  {
    MPI_Finalize();
    return 0;
  }
  mySolver.collectDataCyclic();
  mySolver.solve();

  // If this works, then I can print out the data to see if its correct
  mySolver.printL();
  MPI_Finalize();
  return 0;
}
