/* Author: Edward Hutter */

#include "../../alg/qr/cacqr/cacqr.h"
#include "../../test/qr/validate.h"

using namespace std;

int main(int argc, char** argv){
  using T = double; using U = int64_t; using MatrixType = matrix<T,U,rect>;

  int rank,size,provided; MPI_Init_thread(&argc,&argv,MPI_THREAD_SINGLE,&provided);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);

  size_t variant    = atoi(argv[1]);// 1 - cacqr, 2 - cacqr2
  U num_rows        = atoi(argv[2]);// number of rows in global matrix
  U num_columns     = atoi(argv[3]);// number of columns in global matrix
  U rep_factor      = atoi(argv[4]);// decides depth of process grid and replication factor of matrix
  bool complete_inv = atoi(argv[5]);// decides whether to complete inverse in cholinv
  U split           = atoi(argv[6]);// split factor in cholinv
  U bcMultiplier    = atoi(argv[7]);// base case depth factor in cholinv
  size_t num_chunks = atoi(argv[8]);// splits up communication in summa into nonblocking chunks
  size_t num_iter   = atoi(argv[9]);// number of simulations of the algorithm for performance testing
  size_t id         = atoi(argv[10]);// 0 for critter-only, 1 for critter+production, 2 for critter+production+numerical

  using qr_type = typename qr::cacqr<qr::policy::cacqr::Serialize,qr::policy::cacqr::SaveIntermediates>;
  {
    double time_global = 0;
    T residual_error,orthogonality_error; auto mpi_dtype = mpi_type<T>::type;
    auto RectTopo = topo::rect(MPI_COMM_WORLD,rep_factor,num_chunks);
    MatrixType A(num_columns,num_rows,RectTopo.c,RectTopo.d);
    A.distribute_random(RectTopo.x, RectTopo.y, RectTopo.c, RectTopo.d, rank/RectTopo.c);
    // Generate algorithmic structure via instantiating packs
    cholesky::cholinv<>::info<T,U> ci_pack(complete_inv,split,bcMultiplier,'U');
    qr_type::info<T,U,decltype(ci_pack)::alg_type> pack(variant,ci_pack);

    for (size_t i=0; i<num_iter; i++){
      MPI_Barrier(MPI_COMM_WORLD);
      critter::start();
      qr_type::factor(A, pack, RectTopo);
      critter::stop();

      if (id>0){
        volatile double time_local = MPI_Wtime();
        qr_type::factor(A, pack, RectTopo);
        time_global = MPI_Wtime() - time_local;
        MPI_Allreduce(MPI_IN_PLACE,&time_global,1,mpi_dtype,MPI_MAX,MPI_COMM_WORLD);
        if (rank==0){ std::cout << time_global << std::endl; }
        if (id > 1){
          auto residual_local = qr::validate<qr_type>::residual(A,pack,RectTopo);
          auto orthogonality_local = qr::validate<qr_type>::orthogonality(A,pack,RectTopo);
          MPI_Reduce(&residual_local,&residual_error,1,mpi_dtype,MPI_MAX,0,MPI_COMM_WORLD);
          MPI_Reduce(&orthogonality_local,&orthogonality_error,1,mpi_dtype,MPI_MAX,0,MPI_COMM_WORLD);
          if (rank==0){ std::cout << residual_error << " " << orthogonality_error << std::endl; }
        }
      }
    }
  }

  MPI_Finalize();
  return 0;
}
