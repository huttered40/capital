/* Author: Edward Hutter */

#include "../../src/alg/qr/cacqr/cacqr.h"
#include "../../test/qr/validate.h"

using namespace std;

int main(int argc, char** argv){
  using T = double; using U = int64_t; using MatrixType = matrix<T,U,rect>;

  int rank,size,provided; MPI_Init_thread(&argc,&argv,MPI_THREAD_SINGLE,&provided);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);

  size_t variant     = atoi(argv[1]);// 1 - cacqr, 2 - cacqr2
  U num_rows         = atoi(argv[2]);// number of rows in global matrix
  U num_columns      = atoi(argv[3]);// number of columns in global matrix
  U rep_factor_start = atoi(argv[4]);// decides depth of process grid and replication factor of matrix
  U rep_factor_end   = atoi(argv[5]);// decides depth of process grid and replication factor of matrix
  bool complete_inv  = atoi(argv[6]);// decides whether to complete inverse in cholinv
  U split            = atoi(argv[7]);// split factor in cholinv
  U bcMultiplier_start = atoi(argv[8]);// base case depth factor in cholinv
  U bcMultiplier_end  = atoi(argv[9]);// base case depth factor in cholinv
  size_t layout     = atoi(argv[10]);// arranges sub-communicator layout
  size_t num_chunks = atoi(argv[11]);// splits up communication in summa into nonblocking chunks
  size_t num_iter   = atoi(argv[12]);// number of simulations of the algorithm for performance testing

  using qr_type = qr::cacqr<qr::policy::cacqr::Serialize,qr::policy::cacqr::SaveIntermediates>;
  {
    T residual_error,orthogonality_error; auto mpi_dtype = mpi_type<T>::type;

    for (int i=rep_factor_start; i<=rep_factor_end; i++){
      auto RectTopo = topo::rect(MPI_COMM_WORLD,i,layout,num_chunks);
      MatrixType A(num_columns,num_rows,RectTopo.c,RectTopo.d);
      A.distribute_random(RectTopo.x, RectTopo.y, RectTopo.c, RectTopo.d, rank/RectTopo.c);

      for (int j=bcMultiplier_start; j<=bcMultiplier_end; j++){
        // Generate algorithmic structure via instantiating packs
        cholesky::cholinv<cholesky::policy::cholinv::Serialize,cholesky::policy::cholinv::SaveIntermediates,
                          cholesky::policy::cholinv::NoReplication>::info<T,U> ci_pack(complete_inv,split,j,'U');
        qr_type::info<T,U,decltype(ci_pack)::alg_type> pack(variant,ci_pack);

        for (size_t k=0; k<num_iter; k++){
          A.distribute_random(RectTopo.x, RectTopo.y, RectTopo.c, RectTopo.d, rank/RectTopo.c);
          PMPI_Barrier(MPI_COMM_WORLD);
          qr_type::factor(A, pack, RectTopo);
        }
        A.distribute_random(RectTopo.x, RectTopo.y, RectTopo.c, RectTopo.d, rank/RectTopo.c);
        PMPI_Barrier(MPI_COMM_WORLD);
        auto start_time = MPI_Wtime();
        qr_type::factor(A, pack, RectTopo);
	auto end_time = MPI_Wtime() - start_time;
        PMPI_Allreduce(MPI_IN_PLACE,&end_time,1,MPI_DOUBLE,MPI_MAX,MPI_COMM_WORLD);
        if (rank == 0) std::cout << num_rows << " " << num_columns << " " << i << " " << j << " " << end_time << std::endl;

        A.distribute_random(RectTopo.x, RectTopo.y, RectTopo.c, RectTopo.d, rank/RectTopo.c);
#ifdef CRITTER
        critter::start();
#endif
        qr_type::factor(A, pack, RectTopo);
#ifdef CRITTER
        critter::stop();
        critter::record();
#endif
/*
      qr_type::factor(A, pack, RectTopo);
      auto residual_local = qr::validate<qr_type>::residual(A,pack,RectTopo);
      auto orthogonality_local = qr::validate<qr_type>::orthogonality(A,pack,RectTopo);
      MPI_Reduce(&residual_local,&residual_error,1,mpi_dtype,MPI_MAX,0,MPI_COMM_WORLD);
      MPI_Reduce(&orthogonality_local,&orthogonality_error,1,mpi_dtype,MPI_MAX,0,MPI_COMM_WORLD);
      if (rank==0){ std::cout << residual_error << " " << orthogonality_error << std::endl; }
*/
      }
    }
  }
  MPI_Finalize();
  return 0;
}
