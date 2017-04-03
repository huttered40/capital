/*

	Program to start the Recursive 3D LU Factorization Solver
	Author: Edward Hutter
*/

/* Local Includes */
#include "cholesky.h"

/* System Includes */
#include <ctime>	//clock

using namespace std;

int main(int argc, char **argv)
{
  int rank,size,provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_SINGLE,&provided);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  cholesky<double> mySolver(rank,size,3, argc, argv);		// last argument is matrix dimension. We can change it to be any power of 2

/*
  if (size == 1)
  {
    clock_t start;
    double duration;
    start = clock(); 	// start time
    int trySize = 16;
    std::vector<double> data(trySize*trySize);
    std::vector<double> dataL(trySize*trySize);
    std::vector<double> dataInverse(trySize*trySize);
    mySolver.lapackTest(data, dataL, dataInverse, trySize);
    duration = (clock() - start) / (double)CLOCKS_PER_SEC;

    cout << "Time - " << duration << endl;
    MPI_Finalize();
    return 0; 
  }
*/

  // So I start my timings after the data is distributed, which involved no communication
  clock_t start;
  double duration;

  uint64_t matSize = atoi(argv[1]);
  uint64_t matSizeL = matSize;
  matSizeL *= (matSize+1);
  matSizeL >>= 1;
  std::vector<double> matA(matSize*matSize);
  std::vector<double> matL(matSizeL);
  std::vector<double> matLInverse(matSizeL); 

  start = clock(); 							// start timer
  mySolver.choleskySolve(matA, matL, matLInverse,false);		// run algorithm
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

  //mySolver.scalapackCholesky();			// dummy function for now. Doesnt do anything
  // for now, comment this out, then of course comment it back in and pass in matA, matL, matLInverse to check for correctness
  mySolver.getResidualLayer(matA, matL, matLInverse);

  MPI_Finalize();
  return 0;
}
