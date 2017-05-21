/*
	Program to factorize matrix A into a product of matrices Q and R (A=QR) over a tunable processor grid
	
	Author: Edward Hutter
*/

/* Local Includes */
#include "qr.h"

/* System Includes */
#include <ctime>	//clock

using namespace std;

int main(int argc, char **argv)
{
  int rank,size,provided;
  MPI_Init_thread(&argc,&argv,MPI_THREAD_SINGLE,&provided);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&size);

  qr<double> mySolver(rank,size,argc, argv, MPI_COMM_WORLD);

  // So I start my timings after the data is distributed, which involved no communication
  clock_t start;
  double duration;

  uint32_t m = atoi(argv[1]);
  uint32_t n = atoi(argv[2]);
  uint32_t pGridDimTune = atoi(argv[3]);	// This is the tunable p-grid parameter. For now, its c=1
  uint32_t pGridDimReact = size/(pGridDimTune*pGridDimTune);	// use 64-bit trick later. This division is expensive?
  uint32_t pGridRowPartition = m/pGridDimReact;
  uint32_t pGridColPartition = n/pGridDimTune;				// will be n when c==1 (in 1D case)
  // 64-bit trick needed below? Change later if necessary.
  uint64_t matASize = pGridRowPartition*pGridColPartition;
  uint64_t matQSize = matASize;
  uint64_t matRSize = pGridColPartition*pGridColPartition;	// In the c==1 case, every processor will own their own entire R (n x n)
  std::vector<double> matA(matASize);
  std::vector<double> matQ(matQSize);
  std::vector<double> matR(matRSize); 

  start = clock(); 							// start timer
  mySolver.qrSolve(matA, matQ, matR, false, 1);		// run algorithm
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
  mySolver.getResidual(matA, matQ, matR);

  MPI_Finalize();
  return 0;
}
