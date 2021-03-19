/* Author: Edward Hutter */

#include "../../src/alg/cholesky/cholinv/cholinv.h"
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
  size_t layout     = atoi(argv[6]);// arranges sub-communicator layout
  size_t num_chunks = atoi(argv[7]);// splits up communication in summa into nonblocking chunks
  size_t num_iter   = atoi(argv[8]);// number of simulations of the algorithm for performance testing

#ifdef CRITTER
  std::vector<std::string> symbols = {
					"CI::factor","CI::factor_diag","CI::trsm","CI::tmu"
                                     };
  critter::init(symbols);
#endif

  using cholesky_type = typename cholesky::cholinv<policy::cholinv::Serialize,policy::cholinv::SaveIntermediates,policy::cholinv::NoReplication>;
//  using cholesky_type = typename cholesky::cholinv<policy::cholinv::Serialize,policy::cholinv::SaveIntermediates,policy::cholinv::ReplicationCommComp>;
//  using cholesky_type = typename cholesky::cholinv<policy::cholinv::Serialize,policy::cholinv::SaveIntermediates,policy::cholinv::ReplicateComp>;
  size_t process_cube_dim = std::nearbyint(std::ceil(pow(size,1./3.)));
  size_t rep_factor = process_cube_dim/rep_div;
  T residual_error_local,residual_error_global; auto mpi_dtype = mpi_type<T>::type;
  { 
    auto SquareTopo = topo::square(MPI_COMM_WORLD,rep_factor,layout,num_chunks);
    MatrixType A(num_rows,num_rows, SquareTopo.d, SquareTopo.d);
    A.distribute_symmetric(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c,true);
    // Generate algorithmic structure via instantiating packs
    cholesky_type::info<T,U> pack(complete_inv,split,bcMultiplier,dir);
    // Warm cache and BLAS/LAPACK/MPI routines
    cholesky_type::factor(A, pack, SquareTopo);

    for (size_t i=0; i<num_iter; i++){
      MPI_Barrier(MPI_COMM_WORLD);
#ifdef CRITTER
      critter::start();
#else
      auto start_time = MPI_Wtime();
#endif
      cholesky_type::factor(A, pack, SquareTopo);
#ifdef CRITTER
      critter::stop();
      critter::record();
#else
      auto total_time = MPI_Wtime()-start_time;
      if (rank==0) std::cout << "total time - " << total_time << std::endl;
#endif
/* For calculating error. No longer relevant.
      cholesky_type::factor(A, pack, SquareTopo);
      residual_error_local = cholesky::validate<cholesky_type>::residual(A, pack, SquareTopo);
      MPI_Reduce(&residual_error_local, &residual_error_global, 1, mpi_dtype, MPI_MAX, 0, MPI_COMM_WORLD);
      if (rank==0){ std::cout << residual_error_global << std::endl; }
*/
    }
  }
  MPI_Finalize();
  return 0;
}
