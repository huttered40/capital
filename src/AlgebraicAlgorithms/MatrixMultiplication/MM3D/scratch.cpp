/* Test case for Sam
   There is an AMPI bug with MPI_Type_vector + MPI_Allgather
*/

#include <iostream>
#include <cmath>
#include <vector>
#include <mpi.h>

using namespace std;

int main(int argc, char** argv)
{
  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // size -- total number of processors in the 3D grid
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int pGridDimensionSize = std::nearbyint(pow(size,1./3.));
  int helper = pGridDimensionSize;
  helper *= helper;
  int pCoordX = rank%pGridDimensionSize;
  int pCoordY = (rank%helper)/pGridDimensionSize;
  int pCoordZ = rank/helper;
  MPI_Comm rowComm, columnComm, sliceComm;

  // First, split the 3D Cube processor grid communicator into groups based on what 2D slice they are located on.
  // Then, subdivide further into row groups and column groups
  MPI_Comm_split(MPI_COMM_WORLD, pCoordZ, rank, &sliceComm);
  MPI_Comm_split(sliceComm, pCoordY, pCoordX, &rowComm);
  MPI_Comm_split(sliceComm, pCoordX, pCoordY, &columnComm);

  int globalNumRows = (1<<8);
  int localNumRows = globalNumRows/pGridDimensionSize;
  int globalNumColumns = (1<<8);
  int localNumColumns = globalNumColumns/pGridDimensionSize;

  int vecSize = localNumRows*localNumColumns;
  std::vector<double> data(vecSize);
  for (int i=0; i<data.size(); i++)
  {
    data[i] = drand48();
  }

  // Now lets create a MPI_Type_vector and Allgather it.
  std::vector<double> collectMatrix(vecSize);			// will need to change upon Serialize changes
  int shift = (pCoordZ + pCoordY) % pGridDimensionSize;
  int blockLength = localNumRows/pGridDimensionSize;
  int dataOffset = blockLength*shift;
  MPI_Datatype matrixColumnData;
  MPI_Type_vector(localNumColumns,blockLength,localNumRows,MPI_DOUBLE,&matrixColumnData);
  MPI_Type_commit(&matrixColumnData);
  int messageSize = vecSize/pGridDimensionSize;
  MPI_Allgather(&data[dataOffset], 1, matrixColumnData, &collectMatrix[0], messageSize, MPI_DOUBLE, columnComm);
// AMPI has trouble here. Check back with Sam. Then recheck for correctness.
  // debugging
  for (int i=0; i<vecSize; i++)
  {
    std::cout << "check val - " << collectMatrix[i] << std::endl;
  }

  MPI_Finalize();
  return 0;
}
