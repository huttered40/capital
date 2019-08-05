/* Author: Edward Hutter */

#include "../../alg/matmult/summa3d/summa3d.h"
#include "validate/validate.h"

using namespace std;

int main(int argc, char** argv){
  using MatrixTypeR = matrix<double,int64_t,rectangular,cyclic>;
  using MatrixTypeLT = matrix<double,int64_t,lowertri,cyclic>;
  using MatrixTypeUT = matrix<double,int64_t,uppertri,cyclic>;

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
  int64_t globalMatrixSizeM = atoi(argv[1]);
  int64_t globalMatrixSizeN = atoi(argv[2]);
  int64_t globalMatrixSizeK = atoi(argv[3]);
  int64_t pGridDimensionC = atoi(argv[4]);
  size_t methodKey2 = atoi(argv[5]);
  int64_t numIterations = atoi(argv[6]);
  std::string fileStr1 = argv[7];	// Critter
  std::string fileStr2 = argv[8];	// Performance/Residual/DevOrth

  size_t CubeFaceDim = std::nearbyint(std::pow(size/pGridDimensionC,1./2.));
  size_t CubeTopFaceSize = pGridDimensionC*CubeFaceDim;
  size_t pCoordY = rank/CubeTopFaceSize;
  size_t pCoordX = (rank%CubeTopFaceSize)/pGridDimensionC;
  std::vector<size_t> Inputs{matA.getNumRowsGlobal(),matA.getNumColumnsGlobal(),matB.getNumColumnsGlobal(),pGridDimensionC};
  std::vector<const char*> InputNames{"m","n","k","c"};

  for (size_t test=0; test<2; test++){
    switch(test){
      case 0:
        critter::init(1,fileStr1);
      case 1:
        critter::init(0,fileStr2);
    }

    // Loop for getting a good range of results.
    for (size_t i=0; i<numIterations; i++){
      MatrixTypeR matA(globalMatrixSizeK,globalMatrixSizeM,CubeFaceDim,CubeFaceDim);
      MatrixTypeR matB(globalMatrixSizeN,globalMatrixSizeK,CubeFaceDim,CubeFaceDim);
      MatrixTypeR matC(globalMatrixSizeN,globalMatrixSizeM,CubeFaceDim,CubeFaceDim);
      blasEngineArgumentPackage_gemm<double> blasArgs(blasEngineOrder::AblasColumnMajor, blasEngineTranspose::AblasNoTrans, blasEngineTranspose::AblasNoTrans, 1., 0.);
      double iterTimeGlobal;
      // Note: I think these calls below are still ok given the new topology mapping on Blue Waters/Stampede2
      matA.DistributeRandom(pCoordX, pCoordY, CubeFaceDim, CubeFaceDim, pCoordX*CubeFaceDim + pCoordY);
      matB.DistributeRandom(pCoordX, pCoordY, CubeFaceDim, CubeFaceDim, (pCoordX*CubeFaceDim + pCoordY)*(-1));
      matC.DistributeRandom(pCoordX, pCoordY, CubeFaceDim, CubeFaceDim, (pCoordX*CubeFaceDim + pCoordY)*(-1));
      MPI_Barrier(MPI_COMM_WORLD);		// make sure each process starts together
      critter::reset();
      double startTime=MPI_Wtime();
      matmult::summa3d::invoke(matA, matB, matC, topo::square(MPI_COMM_WORLD,pGridDimensionC), blasArgs, methodKey2);
      double iterTimeLocal=MPI_Wtime()-startTime;
      switch(test){
        case 0:{
          critter::print("MatrixMultiplication", size, Inputs.size(), &Inputs[0], &InputNames[0]);
	  break;
	}
        case 1:{
          MPI_Reduce(&iterTimeLocal, &iterTimeGlobal, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
          std::vector<double> Outputs(1);
	  Outputs[0] = iterTimeGlobal;
          critter::print("MatrixMultiplication", size, Inputs.size(), &Inputs[0], &InputNames[0], Outputs.size(), &Outputs[0]);
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
