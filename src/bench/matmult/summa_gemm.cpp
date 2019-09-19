/* Author: Edward Hutter */

#include "../../alg/matmult/summa/summa.h"
#include "validate/validate.h"

using namespace std;

int main(int argc, char** argv){
  using MatrixTypeR = matrix<double,size_t,rect,cyclic>;
  using MatrixTypeLT = matrix<double,size_t,lowertri,cyclic>;
  using MatrixTypeUT = matrix<double,size_t,uppertri,cyclic>;

  int rank,size,provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  // size -- total number of processors in the 3D grid
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  util::InitialGEMM<double>();

  /*
    Choices for methodKey2: 0) Broadcast + Allreduce
			    1) Allgather + Allreduce
  */
  size_t globalMatrixSizeM = atoi(argv[1]);
  size_t globalMatrixSizeN = atoi(argv[2]);
  size_t globalMatrixSizeK = atoi(argv[3]);
  size_t pGridDimensionC = atoi(argv[4]);
  size_t methodKey2 = atoi(argv[5]);
  size_t numIterations = atoi(argv[6]);

  for (size_t test=0; test<2; test++){
    // Create new topology each outer-iteration so the instance goes out of scope before MPI_Finalize
    auto SquareTopo = topo::square(MPI_COMM_WORLD,pGridDimensionC);

    // Loop for getting a good range of results.
    for (size_t i=0; i<numIterations; i++){
      MatrixTypeR matA(globalMatrixSizeK,globalMatrixSizeM,SquareTopo.d,SquareTopo.d);
      MatrixTypeR matB(globalMatrixSizeN,globalMatrixSizeK,SquareTopo.d,SquareTopo.d);
      MatrixTypeR matC(globalMatrixSizeN,globalMatrixSizeM,SquareTopo.d,SquareTopo.d);
      blas::ArgPack_gemm<double> blasArgs(blas::Order::AblasColumnMajor, blas::Transpose::AblasNoTrans, blas::Transpose::AblasNoTrans, 1., 0.);
      double iterTimeGlobal;
      // Note: I think these calls below are still ok given the new topology mapping on Blue Waters/Stampede2
      matA.DistributeRandom(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c);
      matB.DistributeRandom(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c*(-1));
      matC.DistributeRandom(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c*(-1));
      MPI_Barrier(MPI_COMM_WORLD);		// make sure each process starts together
      critter::start();
      double startTime=MPI_Wtime();
      matmult::summa::invoke(matA, matB, matC, SquareTopo, blasArgs, methodKey2);
      double iterTimeLocal=MPI_Wtime()-startTime;
      critter::stop();

      switch(test){
        case 1:{
          MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
          std::vector<double> Outputs(1);
	  Outputs[0] = iterTimeGlobal;
          critter::print(i==0, size, Outputs.size(), &Outputs[0]);
          //matmult::validate<summa3d>::validateLocal(matA,matB,matC,MPI_COMM_WORLD,blasArgs);
	  break;
	}
      }
    }
  }

  MPI_Finalize();
  return 0;
}
