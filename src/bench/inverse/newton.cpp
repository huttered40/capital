/* Author: Edward Hutter */

#include "../../alg/inverse/newton/newton.h"
#include "validate/validate.h"

using namespace std;

int main(int argc, char** argv){
  using MatrixTypeA = matrix<double,size_t,square,cyclic>;

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  util::InitialGEMM<double>();

  char dir = 'U';
  size_t globalMatrixSize = atoi(argv[1]);
  size_t pGridDimensionC = atoi(argv[2]);
  // TODO: Param for max number of iterations?
  size_t numIterations = atoi(argv[3]);

  size_t pGridCubeDim = std::nearbyint(std::ceil(pow(size,1./3.)));
  pGridDimensionC = pGridCubeDim/pGridDimensionC;

  for (size_t i=0; i<numIterations; i++){
    // Create new topology each outer-iteration so the instance goes out of scope before MPI_Finalize
    auto SquareTopo = topo::square(MPI_COMM_WORLD,pGridDimensionC);
    // Reset matrixA
    MatrixTypeA matA(globalMatrixSize,globalMatrixSize, SquareTopo.d, SquareTopo.d);
    double iterTimeGlobal,iterErrorGlobal;
    matA.DistributeDebug(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d);
    MPI_Barrier(MPI_COMM_WORLD);		// make sure each process starts together
    critter::start();
    inverse::newton::invoke(matA, SquareTopo,1e-14,10);
    critter::stop();

    matA.DistributeDebug(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d);
    double startTime=MPI_Wtime();
    inverse::newton::invoke(matA, SquareTopo,1e-14,10);
    double iterTimeLocal=MPI_Wtime() - startTime;

    // debug
    //matA.print();

    MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MatrixTypeA saveA = matA;
    saveA.DistributeDebug(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d);
    double iterErrorLocal = inverse::validate<inverse::newton>::invoke(saveA, matA, SquareTopo);
    MPI_Reduce(&iterErrorLocal, &iterErrorGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    std::vector<double> Outputs(2);
    Outputs[0] = iterTimeGlobal; Outputs[1] = iterErrorGlobal;
    critter::print(Outputs.size(), &Outputs[0]);
  }
  MPI_Finalize();
  return 0;
}
