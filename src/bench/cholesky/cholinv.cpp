/* Author: Edward Hutter */

#include "../../alg/cholesky/cholinv/cholinv.h"
#include "../../test/cholesky/validate.h"

using namespace std;

int main(int argc, char** argv){
  using T = double; using U = int64_t; using MatrixType = matrix<T,U,rect>; using namespace cholesky;

  int rank,size,provided; MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank); MPI_Comm_size(MPI_COMM_WORLD, &size);

  char dir          = 'U';
  U num_rows        = atoi(argv[1]);// number of rows in global matrix
  U rep_div         = atoi(argv[2]);// cuts the depth of cubic process grid (only trivial support of value '1' is supported)
  bool complete_inv = atoi(argv[3]);// decides whether to complete inverse in cholinv
  U split           = atoi(argv[4]);// split factor in cholinv
  U bcMultiplier    = atoi(argv[5]);// base case depth factor in cholinv
  size_t num_chunks = atoi(argv[6]);// splits up communication in summa into nonblocking chunks
  size_t num_iter   = atoi(argv[7]);// number of simulations of the algorithm for performance testing
  size_t id         = atoi(argv[8]);// 0 for critter-only, 1 for critter+production, 2 for critter+production+numerical

  using cholesky_type = typename cholesky::cholinv<policy::cholinv::Serialize,policy::cholinv::SaveIntermediates>;
  size_t process_cube_dim = std::nearbyint(std::ceil(pow(size,1./3.)));
  size_t rep_factor = process_cube_dim/rep_div; double time_global = 0; double time_local = 0;
  T residual_error_local,residual_error_global; auto mpi_dtype = mpi_type<T>::type;
  { 
    auto SquareTopo = topo::square(MPI_COMM_WORLD,rep_factor,num_chunks);
    MatrixType A(num_rows,num_rows, SquareTopo.d, SquareTopo.d);
    A.distribute_symmetric(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c,true);
    // Generate algorithmic structure via instantiating packs
    cholesky_type::info<T,U> pack(complete_inv,split,bcMultiplier,dir);

    for (size_t i=0; i<num_iter; i++){
      MPI_Barrier(MPI_COMM_WORLD);
      critter::start();
      cholesky_type::factor(A, pack, SquareTopo);
      critter::stop();
  
      if (id>0){
        time_local=MPI_Wtime();
        cholesky_type::factor(A, pack, SquareTopo);
        time_local=MPI_Wtime()-time_local;
        MPI_Reduce(&time_local, &time_global, 1, mpi_dtype, MPI_MAX, 0, MPI_COMM_WORLD);
        if (rank==0){ std::cout << time_global << std::endl; }
        if (id>1){
          residual_error_local = cholesky::validate<cholesky_type>::invoke(A, pack, SquareTopo);
          MPI_Reduce(&residual_error_local, &residual_error_global, 1, mpi_dtype, MPI_MAX, 0, MPI_COMM_WORLD);
          if (rank==0){ std::cout << residual_error_global << std::endl; }
        }
      }
    }
  }
  MPI_Finalize();
  return 0;
}
