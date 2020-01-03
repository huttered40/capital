/* Author: Edward Hutter */

#include "../../alg/cholesky/cholinv/cholinv.h"
#include "../../test/cholesky/validate.h"

using namespace std;

int main(int argc, char** argv){
  using T = double; using U = int64_t;
  using MatrixTypeA = matrix<T,U,rect>;
  using namespace cholesky::policy;

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  char dir = 'U';
  U globalMatrixSize = atoi(argv[1]);
  U pGridDimensionC = atoi(argv[2]);
  bool complete_inv = atoi(argv[3]);
  U split = atoi(argv[4]); // multiplies baseCase dimension by sucessive 2
  U bcMultiplier = atoi(argv[5]); // multiplies baseCase dimension by sucessive 2
  size_t num_chunks        = atoi(argv[6]);
  size_t numIterations = atoi(argv[7]);
  size_t id = atoi(argv[8]);	// 0 for critter-only, 1 for critter+production, 2 for critter+production+numerical

  using cholesky_type = typename cholesky::cholinv<cholinv::NoSerialize>;
  size_t pGridCubeDim = std::nearbyint(std::ceil(pow(size,1./3.)));
  pGridDimensionC = pGridCubeDim/pGridDimensionC;
  T iterErrorLocal; auto mpi_dtype = mpi_type<T>::type;
  { 
    auto SquareTopo = topo::square(MPI_COMM_WORLD,pGridDimensionC,num_chunks);
    MatrixTypeA A(globalMatrixSize,globalMatrixSize, SquareTopo.d, SquareTopo.d);
    double iterTimeGlobal,iterErrorGlobal;
    A.distribute_symmetric(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c,true);
    cholesky_type::info<T,U> pack(complete_inv,split,bcMultiplier,dir);

    for (size_t i=0; i<numIterations; i++){
      // Generate algorithmic structure via instantiating packs
      MPI_Barrier(MPI_COMM_WORLD);		// make sure each process starts together
      critter::start();
      cholesky_type::factor(A, pack, SquareTopo);
      critter::stop();
  
      if (id>0){
        double startTime=MPI_Wtime();
        cholesky_type::factor(A, pack, SquareTopo);
        double iterTimeLocal=MPI_Wtime() - startTime;
        MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, mpi_dtype, MPI_MAX, 0, MPI_COMM_WORLD);
        if (id>1){
          iterErrorLocal = cholesky::validate<cholesky_type>::invoke(A, cholesky_type::construct_R(pack,SquareTopo), pack, SquareTopo);
          MPI_Reduce(&iterErrorLocal, &iterErrorGlobal, 1, mpi_dtype, MPI_MAX, 0, MPI_COMM_WORLD);
        }
      }
      if (rank==0){
        std::cout << iterTimeGlobal << " " << iterErrorGlobal << std::endl;
      }
    }
  }
  MPI_Finalize();
  return 0;
}
