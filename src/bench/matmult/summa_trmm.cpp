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

    TRMM has some special arguments. We will just fix 1) and 1) below
    Choices for matrixUpLo: 0) Lower-triangular
                            1) Upper-triangular
    Choices for triangleSide: 0) Triangle * Rectangular (matrixA * matrixB)
                              1) Rectangular * Triangle (matrixB * matrixA)
  */
  size_t globalMatrixSizeM = atoi(argv[1]);
  size_t globalMatrixSizeN = atoi(argv[2]);
  size_t pGridDimensionC = atoi(argv[3]);
  size_t methodKey2 = atoi(argv[4]);
  size_t numIterations = atoi(argv[5]);
  size_t ppn=atoi(argv[6]);
  size_t tpr=atoi(argv[7]);
  std::string fileStr1 = argv[8];	// Critter
  std::string fileStr2 = argv[9];	// Performance/Residual/DevOrth

  std::vector<size_t> Inputs{matA.getNumRowsGlobal(),matA.getNumColumnsGlobal(),matB.getNumColumnsGlobal(),pGridDimensionC,numIterations,ppn,tpr};
  std::vector<const char*> InputNames{"m","n","k","c","numiter","ppn","tpr"};

  for (size_t test=0; test<2; test++){
    // Create new topology each outer-iteration so the instance goes out of scope before MPI_Finalize
    auto SquareTopo = topo::square(MPI_COMM_WORLD,pGridDimensionC);

    switch(test){
      case 0:
        critter::init(1,fileStr1);
      case 1:
        critter::init(0,fileStr2);
    }

    // Loop for getting a good range of results.
    for (size_t i=0; i<numIterations; i++){
      MatrixTypeR matB(globalMatrixSizeN,globalMatrixSizeM, SquareTopo.d,SquareTopo.d);
      MatrixTypeUT matA(globalMatrixSizeN,globalMatrixSizeN, SquareTopo.d,SquareTopo.d);
      blasEngineArgumentPackage_trmm<double> blasArgs(blasEngineOrder::AblasColumnMajor, blasEngineSide::AblasRight, blasEngineUpLo::AblasUpper,
        blasEngineTranspose::AblasNoTrans, blasEngineDiag::AblasNonUnit, 1.);
      double iterTimeGlobal;
      // Note: I think these calls below are still ok given the new topology mapping on Blue Waters/Stampede2
      matA.DistributeRandom(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c);
      matB.DistributeRandom(SquareTopo.x, SquareTopo.y, SquareTopo.d, SquareTopo.d, rank/SquareTopo.c*(-1));
      MPI_Barrier(MPI_COMM_WORLD);		// make sure each process starts together
      critter::reset();
      double startTime=MPI_Wtime();
      matmult::summa3d::invoke(matA, matB, SquareTopo, blasArgs, methodKey2);
      double iterTimeLocal=MPI_Wtime()-startTime;
      switch(test){
        case 0:{
          critter::print(i==0, "MatrixMultiplication", size, Inputs.size(), &Inputs[0], &InputNames[0]);
	  break;
	}
        case 1:{
          MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
          std::vector<double> Outputs(1);
	  Outputs[0] = iterTimeGlobal;
          critter::print(i==0, "MatrixMultiplication", size, Inputs.size(), &Inputs[0], &InputNames[0], Outputs.size(), &Outputs[0]);
          //matmult::validate<summa3d>::validateLocal(matA,matB,matC,MPI_COMM_WORLD,blasArgs);
	  break;
	}
      }
    }
    critter::finalize();
  }

  MPI_Finalize();
  return 0;
}