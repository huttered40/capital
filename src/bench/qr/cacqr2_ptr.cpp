/* Author: Edward Hutter */

#include "../../alg/qr/cacqr/cacqr.h"
#include "validate/validate.h"

using namespace std;

int main(int argc, char** argv){
  using T = double; using U = int64_t;
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
  auto mpi_dtype = mpi_type<T>::type;
  using qr_type = typename qr::cacqr2<qr::policy::cacqr::SerializeSymmetricToTriangle>;
  {
    double iterTimeGlobal = 0; double iterTimeLocal = 0;
    auto RectTopo = topo::rect(MPI_COMM_WORLD,dimensionC, num_chunks);
    // Generate algorithmic structure via instantiating packs
    cholesky::cholinv<>::pack ci_pack(inverseCutOffMultiplier,'U');
    qr_type::pack<decltype(ci_pack)::alg_type> pack(ci_pack);
    //TODO: deal with non-power-2 later
    U localMatrixDimensionM = (globalMatrixDimensionM/RectTopo.d);
    U localMatrixDimensionN = (globalMatrixDimensionN/RectTopo.c);
    T* A     = new T[localMatrixDimensionN * localMatrixDimensionM];
    T* R     = new T[localMatrixDimensionN * localMatrixDimensionN];

    for (size_t i=0; i<numIterations; i++){
      // reset the matrix before timer starts
      util::random_fill(A, localMatrixDimensionM, localMatrixDimensionN, globalMatrixDimensionM, globalMatrixDimensionN, RectTopo.x, RectTopo.y, RectTopo.c, RectTopo.d, rank/RectTopo.c);
      MPI_Barrier(MPI_COMM_WORLD);	// make sure each process starts together
      critter::start();
      auto ptrs = qr_type::invoke(A, R, localMatrixDimensionM, localMatrixDimensionN, globalMatrixDimensionM, globalMatrixDimensionN, pack, RectTopo);
      critter::stop();

      util::random_fill(ptrs.first, localMatrixDimensionM, localMatrixDimensionN, globalMatrixDimensionM, globalMatrixDimensionN, RectTopo.x, RectTopo.y, RectTopo.c, RectTopo.d, rank/RectTopo.c);
      volatile double startTime=MPI_Wtime();
      auto ptrs2 = qr_type::invoke(ptrs.first, ptrs.second, localMatrixDimensionM, localMatrixDimensionN, globalMatrixDimensionM, globalMatrixDimensionN, pack, RectTopo);
      iterTimeLocal = MPI_Wtime() - startTime;
      MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, mpi_dtype, MPI_MAX, 0, MPI_COMM_WORLD);

      if (rank==0){
        std::cout << iterTimeGlobal << std::endl;
      }
      delete[] ptrs.first; delete[] ptrs.second; delete[] ptrs2.first; delete[] ptrs2.second;
    }
    delete[] A; delete[] R;
  }

  MPI_Finalize();
  return 0;
}
