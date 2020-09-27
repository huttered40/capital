/* Author: Edward Hutter */

#include <iomanip>

#include "../../../src/alg/cholesky/cholinv/cholinv.h"
#include "../../../test/cholesky/validate.h"

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
  size_t layout     = atoi(argv[6]);// arranges sub-communicator layout
  size_t num_chunks = atoi(argv[7]);// splits up communication in summa into nonblocking chunks
  size_t num_iter   = atoi(argv[8]);// number of simulations of the algorithm for performance testing

  using cholesky_type0 = typename cholesky::cholinv<policy::cholinv::NoSerialize,policy::cholinv::SaveIntermediates,policy::cholinv::NoReplication>;
  using cholesky_type1 = typename cholesky::cholinv<policy::cholinv::NoSerialize,policy::cholinv::SaveIntermediates,policy::cholinv::ReplicateCommComp>;
  using cholesky_type2 = typename cholesky::cholinv<policy::cholinv::NoSerialize,policy::cholinv::SaveIntermediates,policy::cholinv::ReplicateComp>;
  size_t process_cube_dim = std::nearbyint(std::ceil(pow(size,1./3.)));
  size_t rep_factor = process_cube_dim/rep_div; double time_global;
  T residual_error_local,residual_error_global; auto mpi_dtype = mpi_type<T>::type;
  { 
    auto SquareTopo = topo::square(MPI_COMM_WORLD,rep_factor,layout,num_chunks);
    MatrixType A(num_rows,num_rows, SquareTopo.d, SquareTopo.d);
    A.distribute_symmetric(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c,true);
    // Generate algorithmic structure via instantiating packs

    // Initial simulations to warm cache, etc.
    cholesky_type0::info<T,U> pack_init(complete_inv,split,bcMultiplier,dir);
    cholesky_type0::factor(A,pack_init,SquareTopo);

    size_t space_dim = 15;
    // Stage 1: autotune each schedule variant individually

    PMPI_Barrier(MPI_COMM_WORLD);
    volatile double st4 = MPI_Wtime();
    for (auto k=0; k<space_dim; k++){
      if (k/5==0){
        cholesky_type0::info<T,U> pack(complete_inv,split,bcMultiplier+k%5,dir);
        // First tune the space constrained to a particular variant
        for (size_t i=0; i<num_iter; i++){
          critter::start(true,false);
          cholesky_type0::factor(A,pack,SquareTopo);
          critter::stop();
          if (i<(num_iter-1)) critter::record(k,false,true);
          else                critter::record(k,true,true);
        }
        // Then reconstruct the execution times
	for (size_t i=0; i<num_iter; i++){
	  critter::start(true,true);
	  cholesky_type0::factor(A,pack,SquareTopo);
	  critter::stop();
	  critter::record(k,false,true);
        }
      }
      else if (k/5==1){
        cholesky_type1::info<T,U> pack(complete_inv,split,bcMultiplier+k%5,dir);
        for (size_t i=0; i<num_iter; i++){
          critter::start(true,false);
          cholesky_type1::factor(A,pack,SquareTopo);
          critter::stop();
          if (i<(num_iter-1)) critter::record(k,false,true);
          else                critter::record(k,true,true);
        }
        // Then reconstruct the execution times
	for (size_t i=0; i<num_iter; i++){
	  critter::start(true,true);
	  cholesky_type1::factor(A,pack,SquareTopo);
	  critter::stop();
	  critter::record(k,false,true);
        }
      }
      else if (k/5==2){
        cholesky_type2::info<T,U> pack(complete_inv,split,bcMultiplier+k%5,dir);
        for (size_t i=0; i<num_iter; i++){
          critter::start(true,false);
          cholesky_type2::factor(A,pack,SquareTopo);
          critter::stop();
          if (i<(num_iter-1)) critter::record(k,false,true);
          else                critter::record(k,true,true);
        }
        // Then reconstruct the execution times
	for (size_t i=0; i<num_iter; i++){
	  critter::start(true,true);
	  cholesky_type2::factor(A,pack,SquareTopo);
	  critter::stop();
	  critter::record(k,false,true);
        }
      }
      critter::clear();
      if (rank==0) std::cout << "progress stage 1 - " << k << std::endl;
    }
    st4 = MPI_Wtime() - st4;
    if (rank==0) std::cout << "wallclock time of stage 1 - " << st4 << std::endl;

  }
  MPI_Finalize();
  return 0;
}
