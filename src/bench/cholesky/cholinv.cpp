/* Author: Edward Hutter */

#include "../../alg/cholesky/cholinv/cholinv.h"
#include "validate/validate.h"

using namespace std;

int main(int argc, char** argv){
  using MatrixTypeA = matrix<double,int64_t,square,cyclic>;

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  char dir = 'U';
  int64_t globalMatrixSize = atoi(argv[1]);
  int64_t pGridDimensionC = atoi(argv[2]);
  int64_t inverseCutOffMultiplier = atoi(argv[3]); // multiplies baseCase dimension by sucessive 2
  int64_t num_chunks        = atoi(argv[4]);
  int64_t numIterations = atoi(argv[5]);

  using cholesky_type = typename cholesky::cholinv<>;
  size_t pGridCubeDim = std::nearbyint(std::ceil(pow(size,1./3.)));
  pGridDimensionC = pGridCubeDim/pGridDimensionC;
 
  { 
  auto SquareTopo = topo::square(MPI_COMM_WORLD,pGridDimensionC,num_chunks);
  MatrixTypeA A(globalMatrixSize,globalMatrixSize, SquareTopo.d, SquareTopo.d);
  MatrixTypeA T(globalMatrixSize,globalMatrixSize, SquareTopo.d, SquareTopo.d);
  MatrixTypeA saveA = A;
  double iterTimeGlobal,iterErrorGlobal;
  // Generate algorithmic structure via instantiating packs
  cholesky_type::pack pack(inverseCutOffMultiplier,dir);

  for (size_t i=0; i<numIterations; i++){
    A.distribute_symmetric(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c,true);
    MPI_Barrier(MPI_COMM_WORLD);		// make sure each process starts together
    critter::start();
    cholesky_type::invoke(A, T, pack, SquareTopo);
    critter::stop();

    A.distribute_symmetric(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c,true);
    double startTime=MPI_Wtime();
    cholesky_type::invoke(A, T, pack, SquareTopo);
    double iterTimeLocal=MPI_Wtime() - startTime;
    MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    saveA.distribute_symmetric(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c,true);
    double iterErrorLocal = cholesky::validate<cholesky_type>::invoke(saveA, A, dir, SquareTopo);
    MPI_Reduce(&iterErrorLocal, &iterErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank==0){
      std::cout << iterTimeGlobal << " " << iterErrorGlobal << std::endl;
    }
  }
  }
  MPI_Finalize();
  return 0;
}
