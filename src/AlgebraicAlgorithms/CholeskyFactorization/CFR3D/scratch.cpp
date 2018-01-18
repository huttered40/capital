/* Test program to pinpoint a bug in AMPI's MPI_Sendrecv_replace */
/* Author: Edward Hutter */

// System includes
#include <iostream>
#include <cstdlib>
#include <utility>
#include <cmath>
#include <string>
#include <mpi.h>

using namespace std;

int main(int argc, char** argv)
{

  // argv[1] - Matrix size x where x represents 2^x.
  // So in future, we might want t way to test non power of 2 dimension matrices

  int rank,size,provided;
  int rankWorld, rankSlice;
  int sizeWorld, sizeSlice;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  MPI_Comm slice2D,commWorld;
  MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &commWorld);

  for (int i=0; i<100; i++)
  {
    MPI_Comm_rank(commWorld, &rankWorld);
    MPI_Comm_size(commWorld, &sizeWorld);

    int pGridDimensionSize = ceil(pow(sizeWorld,1./3.));
    int helper = pGridDimensionSize;
    helper *= helper;
    int pGridCoordZ = rankWorld/helper;

    // Attain the communicator with only processors on the same 2D slice
    MPI_Barrier(commWorld);
    std::cout << "rank " << rankWorld << " here, has pGridCoordZ - " << pGridCoordZ << std::endl;
    MPI_Comm_split(commWorld, pGridCoordZ, rankWorld, &slice2D);
    MPI_Comm_rank(slice2D, &rankSlice);
    MPI_Comm_size(slice2D, &sizeSlice);
    MPI_Comm_free(&slice2D);
  }
  MPI_Comm_free(&commWorld);
  MPI_Finalize();
  return 0;
}
