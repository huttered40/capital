/* Author: Edward Hutter */


template<typename T,typename U,
  template<typename,typename,template<typename,typename,int> class> class StructureA,		// Note: this vould be either rectangular or square.
  template<typename,typename,template<typename,typename,int> class> class StructureQ,		// Note: this vould be either rectangular or square.
  template<typename,typename,template<typename,typename,int> class> class StructureR,		// Note: this vould be either square or triangular
  template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,StructureA,StructureQ,StructureR,blasEngine>::Factor1D(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureQ,Distribution>& matrixQ,
    Matrix<T,U,StructureR,Distribution>& matrixR, U dimensionX, U dimensionY, MPI_Comm commWorld)
{
  // We assume data is owned relative to a 1D processor grid, so every processor owns a chunk of data consisting of
  //   all columns and a block of rows.

  // Decision, does user pass in just the local dimensions, or both the local X and Y, as well as the global?
  //   Note that that info exists in the matrix structures themselves, so we can take advantage of that and
  //   for now, just not use those arguments. Maybe we will get of them later.

  int numPEs;
  MPI_Comm_size(commWorld, &numPEs);
  U localDimensionY = dimensionY/numPEs;		// no error check here, but hopefully 

  Matrix<T,U,StructureQ,Distribution> matrixQ2(std::vector<T>(dimensionX*localDimensionY), dimensionX, localDimensionY, dimensionX,
    dimensionY, true);
  Matrix<T,U,StructureR,Distribution> matrixR1(std::vector<T>(dimensionX*dimensionX), dimensionX, dimensionX, dimensionX,
    dimensionX, true);
  Matrix<T,U,StructureR,Distribution> matrixR2(std::vector<T>(dimensionX*dimensionX), dimensionX, dimensionX, dimensionX,
    dimensionX, true);
  Factor1D_cqr(matrixA, matrixQ2, matrixR1, dimensionX, dimensionY, commWorld);
  Factor1D_cqr(matrixQ2, matrixQ, matrixR2, dimensionX, dimensionY, commWorld);

  // Try gemm first, then try trmm later.
  blasEngineArgumentPackage_gemm<T> gemmPack1;
  gemmPack1.order = blasEngineOrder::AblasRowMajor;
  gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
  gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
  gemmPack1.alpha = 1.;
  gemmPack1.beta = 0.;
  blasEngine<T,U>::_gemm(matrixR2.getRawData(), matrixR1.getRawData(), matrixR.getRawData(), dimensionX, dimensionX,
    dimensionX, dimensionX, dimensionX, dimensionX, dimensionX, dimensionX, dimensionX, gemmPack1);
}

template<typename T,typename U,
  template<typename,typename,template<typename,typename,int> class> class StructureA,		// Note: this vould be either rectangular or square.
  template<typename,typename,template<typename,typename,int> class> class StructureQ,		// Note: this vould be either rectangular or square.
  template<typename,typename,template<typename,typename,int> class> class StructureR,		// Note: this vould be either square or triangular
  template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,StructureA,StructureQ,StructureR,blasEngine>::Factor3D(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureQ,Distribution>& matrixQ,
    Matrix<T,U,StructureR,Distribution>& matrixR, U dimensionX, U dimensionY, MPI_Comm commWorld)
{
  std::cout << "In factor3D\n";
}

template<typename T,typename U,
  template<typename,typename,template<typename,typename,int> class> class StructureA,		// Note: this vould be either rectangular or square.
  template<typename,typename,template<typename,typename,int> class> class StructureQ,		// Note: this vould be either rectangular or square.
  template<typename,typename,template<typename,typename,int> class> class StructureR,		// Note: this vould be either square or triangular
  template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,StructureA,StructureQ,StructureR,blasEngine>::FactorTunable(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureQ,Distribution>& matrixQ,
    Matrix<T,U,StructureR,Distribution>& matrixR, U dimensionX, U dimensionY, MPI_Comm commWorld)
{
  std::cout << "In factorTunable\n";
}


template<typename T,typename U,
  template<typename,typename,template<typename,typename,int> class> class StructureA,		// Note: this vould be either rectangular or square.
  template<typename,typename,template<typename,typename,int> class> class StructureQ,		// Note: this vould be either rectangular or square.
  template<typename,typename,template<typename,typename,int> class> class StructureR,		// Note: this vould be either square or triangular
  template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,StructureA,StructureQ,StructureR,blasEngine>::Factor1D_cqr(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureQ,Distribution>& matrixQ,
    Matrix<T,U,StructureR,Distribution>& matrixR, U dimensionX, U dimensionY, MPI_Comm commWorld)
{
  int numPEs;
  MPI_Comm_size(commWorld, &numPEs);
  U localDimensionY = dimensionY/numPEs;

  // Try gemm first, then try syrk later.
  std::vector<T> localMMvec(dimensionX*dimensionX);
  blasEngineArgumentPackage_gemm<T> gemmPack1;
  gemmPack1.order = blasEngineOrder::AblasRowMajor;
  gemmPack1.transposeA = blasEngineTranspose::AblasTrans;
  gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
  gemmPack1.alpha = 1.;
  gemmPack1.beta = 0.;

  std::vector<T>& local = matrixA.getVectorData();
  std::cout << "Size - " << local.size() << " and dimensionX - " << dimensionX << " and localDimensionY - " << localDimensionY << std::endl;

  blasEngine<T,U>::_gemm(matrixA.getRawData(), matrixA.getRawData(), &localMMvec[0], localDimensionY, dimensionX,
    dimensionX, localDimensionY, dimensionX, dimensionX, dimensionX, dimensionX, dimensionX, gemmPack1);

  // MPI_Allreduce first to replicate the dimensionY x dimensionY matrix on each processor
  MPI_Allreduce(MPI_IN_PLACE, &localMMvec[0], localMMvec.size(), MPI_DOUBLE, MPI_SUM, commWorld);

}

template<typename T,typename U,
  template<typename,typename,template<typename,typename,int> class> class StructureA,		// Note: this vould be either rectangular or square.
  template<typename,typename,template<typename,typename,int> class> class StructureQ,		// Note: this vould be either rectangular or square.
  template<typename,typename,template<typename,typename,int> class> class StructureR,		// Note: this vould be either square or triangular
  template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,StructureA,StructureQ,StructureR,blasEngine>::Factor3D_cqr(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureQ,Distribution>& matrixQ,
    Matrix<T,U,StructureR,Distribution>& matrixR, U dimensionX, U dimensionY, MPI_Comm commWorld)
{
  std::cout << "I am in Factor3D_cqr\n";
}

template<typename T,typename U,
  template<typename,typename,template<typename,typename,int> class> class StructureA,		// Note: this vould be either rectangular or square.
  template<typename,typename,template<typename,typename,int> class> class StructureQ,		// Note: this vould be either rectangular or square.
  template<typename,typename,template<typename,typename,int> class> class StructureR,		// Note: this vould be either square or triangular
  template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,StructureA,StructureQ,StructureR,blasEngine>::FactorTunable_cqr(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureQ,Distribution>& matrixQ,
    Matrix<T,U,StructureR,Distribution>& matrixR, U dimensionX, U dimensionY, MPI_Comm commWorld)
{
  std::cout << "I am in FactorTunable_cqr\n";
}
