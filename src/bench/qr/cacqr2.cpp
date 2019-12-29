/* Author: Edward Hutter */

#include "../../alg/qr/cacqr/cacqr.h"
#include "validate/validate.h"

using namespace std;

int main(int argc, char** argv){
  using T = double; using U = int64_t;
  using MatrixTypeS = matrix<T,U,square,cyclic>;
  using MatrixTypeR = matrix<T,U,rect,cyclic>;

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  U globalMatrixDimensionM = atoi(argv[1]);
  U globalMatrixDimensionN = atoi(argv[2]);
  U dimensionC = atoi(argv[3]);
  U inverseCutOffMultiplier = atoi(argv[4]);
  U num_chunks        = atoi(argv[5]);
  U numIterations=atoi(argv[6]);

  using qr_type = typename qr::cacqr2<qr::policy::cacqr::SerializeSymmetricToTriangle>;
  {
    double iterTimeGlobal = 0; double iterTimeLocal = 0;
    T residualErrorGlobal,orthogonalityErrorGlobal;
    auto mpi_dtype = mpi_type<T>::type;
    auto RectTopo = topo::rect(MPI_COMM_WORLD,dimensionC, num_chunks);
    // Generate algorithmic structure via instantiating packs
    cholesky::cholinv<>::pack ci_pack(inverseCutOffMultiplier,'U');
    qr_type::pack<decltype(ci_pack)::alg_type> pack(ci_pack);
    MatrixTypeR A(globalMatrixDimensionN,globalMatrixDimensionM, RectTopo.c, RectTopo.d);
    MatrixTypeS R(globalMatrixDimensionN,globalMatrixDimensionN, RectTopo.c, RectTopo.c);
    MatrixTypeR saveA = A;

    for (size_t i=0; i<numIterations; i++){
      // reset the matrix before timer starts
      A.distribute_random(RectTopo.x, RectTopo.y, RectTopo.c, RectTopo.d, rank/RectTopo.c);
      MPI_Barrier(MPI_COMM_WORLD);	// make sure each process starts together
      critter::start();
      qr_type::invoke(A, R, pack, RectTopo);
      critter::stop();

      A.distribute_random(RectTopo.x, RectTopo.y, RectTopo.c, RectTopo.d, rank/RectTopo.c);
      volatile double startTime=MPI_Wtime();
      qr_type::invoke(A, R, pack, RectTopo);
      iterTimeLocal = MPI_Wtime() - startTime;

      MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, mpi_dtype, MPI_MAX, 0, MPI_COMM_WORLD);
      saveA.distribute_random(RectTopo.x, RectTopo.y, RectTopo.c, RectTopo.d, rank/RectTopo.c);
      auto error = qr::validate<qr_type>::invoke(saveA, A, R, RectTopo);
      MPI_Reduce(&error.first, &residualErrorGlobal, 1, mpi_dtype, MPI_MAX, 0, MPI_COMM_WORLD);
      MPI_Reduce(&error.second, &orthogonalityErrorGlobal, 1, mpi_dtype, MPI_MAX, 0, MPI_COMM_WORLD);

      if (rank==0){
        std::cout << iterTimeGlobal << " " << residualErrorGlobal << " " << orthogonalityErrorGlobal << std::endl;
      }
    }
  }

  MPI_Finalize();
  return 0;
}
