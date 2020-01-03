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
  bool complete_inv = atoi(argv[4]);
  U bcMultiplier = atoi(argv[5]);
  size_t num_chunks        = atoi(argv[6]);
  size_t numIterations=atoi(argv[7]);
  size_t id = atoi(argv[8]);	// 0 for critter-only, 1 for critter+production, 2 for critter+production+numerical

  auto mpi_dtype = mpi_type<T>::type;
  using qr_type = typename qr::cacqr<>;
  {
    double iterTimeGlobal = 0; double iterTimeLocal = 0;
    auto RectTopo = topo::rect(MPI_COMM_WORLD,dimensionC, num_chunks);
    //TODO: deal with non-power-2 later
    U localMatrixDimensionM = (globalMatrixDimensionM/RectTopo.d);
    U localMatrixDimensionN = (globalMatrixDimensionN/RectTopo.c);
    T* A     = new T[localMatrixDimensionN * localMatrixDimensionM];
    T* R     = new T[localMatrixDimensionN * localMatrixDimensionN];

    for (size_t i=0; i<numIterations; i++){
      // Generate algorithmic structure via instantiating packs
      cholesky::cholinv<>::info<T,U> ci_pack(complete_inv,bcMultiplier,'U');
      qr_type::info<T,U,decltype(ci_pack)::alg_type> pack(ci_pack);
      // reset the matrix before timer starts
      util::random_fill(A, localMatrixDimensionM, localMatrixDimensionN, globalMatrixDimensionM, globalMatrixDimensionN, RectTopo.x, RectTopo.y, RectTopo.c, RectTopo.d, rank/RectTopo.c);
      MPI_Barrier(MPI_COMM_WORLD);	// make sure each process starts together
      critter::start();
      auto ptrs = qr_type::factor(A, R, localMatrixDimensionM, localMatrixDimensionN, globalMatrixDimensionM, globalMatrixDimensionN, pack, RectTopo);
      critter::stop();

      if (id>0){
        util::random_fill(ptrs.first, localMatrixDimensionM, localMatrixDimensionN, globalMatrixDimensionM, globalMatrixDimensionN, RectTopo.x, RectTopo.y, RectTopo.c, RectTopo.d, rank/RectTopo.c);
        volatile double startTime=MPI_Wtime();
        auto ptrs2 = qr_type::factor(ptrs.first, ptrs.second, localMatrixDimensionM, localMatrixDimensionN, globalMatrixDimensionM, globalMatrixDimensionN, pack, RectTopo);
        iterTimeLocal = MPI_Wtime() - startTime;
        MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, mpi_dtype, MPI_MAX, 0, MPI_COMM_WORLD);
      }
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
