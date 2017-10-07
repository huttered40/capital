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

  int numPEs;
  MPI_Comm_size(commWorld, &numPEs);
  U localDimensionY = dimensionY/numPEs;		// no error check here, but hopefully 

  Matrix<T,U,StructureQ,Distribution> matrixQ2(std::vector<T>(dimensionX*localDimensionY), dimensionX, localDimensionY, dimensionX,
    dimensionY, true);
  Matrix<T,U,StructureR,Distribution> matrixR1(std::vector<T>(dimensionX*dimensionX), dimensionX, dimensionX, dimensionX,
    dimensionX, true);
  Matrix<T,U,StructureR,Distribution> matrixR2(std::vector<T>(dimensionX*dimensionX), dimensionX, dimensionX, dimensionX,
    dimensionX, true);
  Factor1D_cqr(matrixA, matrixQ2, matrixR1, dimensionX, localDimensionY, commWorld);
  Factor1D_cqr(matrixQ2, matrixQ, matrixR2, dimensionX, localDimensionY, commWorld);

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
  // We assume data is owned relative to a 3D processor grid

  int numPEs, myRank, pGridDimensionSize;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_size(commWorld, &myRank);
  auto commInfo3D = setUpCommunicators(commWorld);

  // Simple asignments like these don't need pass-by-reference. Remember the new pass-by-value semantics are efficient anyways
  MPI_Comm rowComm = std::get<0>(commInfo3D);
  MPI_Comm_size(rowComm, &pGridDimensionSize);

  U localDimensionX = dimensionX/pGridDimensionSize;		// no error check here, but hopefully 
  U localDimensionY = dimensionY/pGridDimensionSize;		// no error check here, but hopefully 

  Matrix<T,U,StructureQ,Distribution> matrixQ2(std::vector<T>(localDimensionX*localDimensionY), localDimensionX, localDimensionY, dimensionX,
    dimensionY, true);
  Matrix<T,U,StructureR,Distribution> matrixR1(std::vector<T>(localDimensionX*localDimensionX), localDimensionX, localDimensionX, dimensionX,
    dimensionX, true);
  Matrix<T,U,StructureR,Distribution> matrixR2(std::vector<T>(localDimensionX*localDimensionX), localDimensionX, localDimensionX, dimensionX,
    dimensionX, true);
  Factor3D_cqr(matrixA, matrixQ2, matrixR1, localDimensionX, localDimensionY, commWorld);
  Factor3D_cqr(matrixQ2, matrixQ, matrixR2, localDimensionX, localDimensionY, commWorld);

  // Try gemm first, then try trmm later.
  blasEngineArgumentPackage_gemm<T> gemmPack1;
  gemmPack1.order = blasEngineOrder::AblasRowMajor;
  gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
  gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
  gemmPack1.alpha = 1.;
  gemmPack1.beta = 0.;

  // Later optimization - Serialize all 3 matrices into UpperTriangular first, then call this with those matrices, so we don't have to
  //   send half of the data!
  SquareMM3D<T,U,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare,blasEngine>::Multiply(matrixR2, matrixR1,
    matrixR, localDimensionX, localDimensionX, localDimensionX, commWorld, gemmPack1);
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
    Matrix<T,U,StructureR,Distribution>& matrixR, U localDimensionX, U localDimensionY, MPI_Comm commWorld)
{
  int numPEs;
  MPI_Comm_size(commWorld, &numPEs);

  // Try gemm first, then try syrk later.
  std::vector<T> localMMvec(localDimensionX*localDimensionX);
  blasEngineArgumentPackage_gemm<T> gemmPack1;
  gemmPack1.order = blasEngineOrder::AblasRowMajor;
  gemmPack1.transposeA = blasEngineTranspose::AblasTrans;
  gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
  gemmPack1.alpha = 1.;
  gemmPack1.beta = 0.;

  std::vector<T>& local = matrixA.getVectorData();

  blasEngine<T,U>::_gemm(matrixA.getRawData(), matrixA.getRawData(), matrixR.getRawData(), localDimensionY, localDimensionX,
    localDimensionX, localDimensionY, localDimensionX, localDimensionX, localDimensionX, localDimensionX, localDimensionX, gemmPack1);

  // MPI_Allreduce first to replicate the dimensionY x dimensionY matrix on each processor
  // Optimization potential: only allReduce half of this matrix because its symmetric
  //   but only try this later to see if it actually helps, because to do this, I will have to serialize and re-serialize. Would only make sense if dimensionX is huge.
  MPI_Allreduce(MPI_IN_PLACE, matrixR.getRawData(), localDimensionX*localDimensionX, MPI_DOUBLE, MPI_SUM, commWorld);

  // Now, localMMvec is replicated on every processor in commWorld
  LAPACKE_dpotrf(LAPACK_ROW_MAJOR, 'U', localDimensionX, matrixR.getRawData(), localDimensionX);

  // Need a true copy to avoid corrupting R-inverse.
  std::vector<T> RI = matrixR.getVectorData();

  // Next: sequential triangular inverse. Question: does DTRTRI require packed storage or square storage? I think square, so that it can use BLAS-3.
  LAPACKE_dtrtri(LAPACK_ROW_MAJOR, 'U', 'N', localDimensionX, &RI[0], localDimensionX);

  // Finish by performing local matrix multiplication Q = A*R^{-1}
  gemmPack1.transposeA = blasEngineTranspose::AblasNoTrans;
  blasEngine<T,U>::_gemm(matrixA.getRawData(), &RI[0], matrixQ.getRawData(), localDimensionX, localDimensionY,
    localDimensionX, localDimensionX, localDimensionX, localDimensionY, localDimensionY, localDimensionX, localDimensionX, gemmPack1);
   
}

template<typename T,typename U,
  template<typename,typename,template<typename,typename,int> class> class StructureA,		// Note: this vould be either rectangular or square.
  template<typename,typename,template<typename,typename,int> class> class StructureQ,		// Note: this vould be either rectangular or square.
  template<typename,typename,template<typename,typename,int> class> class StructureR,		// Note: this vould be either square or triangular
  template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,StructureA,StructureQ,StructureR,blasEngine>::Factor3D_cqr(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureQ,Distribution>& matrixQ,
    Matrix<T,U,StructureR,Distribution>& matrixR, U localDimensionX, U localDimensionY, MPI_Comm commWorld)
{
  int numPEs, myRank, pGridDimensionSize;
  MPI_Comm_size(commWorld, &numPEs);
  MPI_Comm_size(commWorld, &myRank);
  auto commInfo3D = setUpCommunicators(commWorld);

  MPI_Comm rowComm = std::get<0>(commInfo3D);
  MPI_Comm columnComm = std::get<1>(commInfo3D);
  MPI_Comm sliceComm = std::get<2>(commInfo3D);
  MPI_Comm depthComm = std::get<3>(commInfo3D);
  int pGridCoordX = std::get<4>(commInfo3D);
  int pGridCoordY = std::get<5>(commInfo3D);
  int pGridCoordZ = std::get<6>(commInfo3D);
  MPI_Comm_size(rowComm, &pGridDimensionSize);

  // Need to perform the multiple steps to get our partition of matrixA
  std::vector<T>& dataA = matrixA.getVectorData();
  U sizeA = matrixA.getNumElems();
  std::vector<T> foreignA;	// dont fill with data first, because if root its a waste,
                                //   but need it to outside to get outside scope
  bool isRootRow = ((pGridCoordX == pGridCoordZ) ? true : false);
  bool isRootColumn = ((pGridCoordY == pGridCoordZ) ? true : false);

  BroadcastPanels((isRootRow ? dataA : foreignA), sizeA, isRootRow, pGridCoordZ, rowComm);

  std::vector<T> localB(localDimensionX*localDimensionX);
  blasEngineArgumentPackage_gemm<T> gemmPack1;
  gemmPack1.order = blasEngineOrder::AblasRowMajor;
  gemmPack1.transposeA = blasEngineTranspose::AblasTrans;
  gemmPack1.transposeB = blasEngineTranspose::AblasNoTrans;
  gemmPack1.alpha = 1.;
  gemmPack1.beta = 0.;
  blasEngine<T,U>::_gemm((isRootRow ? &dataA[0] : &foreignA[0]), &dataA[0], &localB[0], localDimensionY, localDimensionX,
    localDimensionX, localDimensionY, localDimensionX, localDimensionX, localDimensionX, localDimensionX, localDimensionX, gemmPack1);

  MPI_Reduce((isRootColumn ? MPI_IN_PLACE : &localB[0]), &localB[0], localDimensionX*localDimensionX, MPI_DOUBLE,
    MPI_SUM, pGridCoordZ, columnComm);

  MPI_Bcast(&localB[0], localDimensionX*localDimensionX, MPI_DOUBLE, pGridCoordZ, depthComm);

  // Stuff localB vector into its own matrix so that we can pass it into CFR3D
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixB(std::move(localB), localDimensionX, localDimensionX,
    localDimensionX*pGridDimensionSize, localDimensionX*pGridDimensionSize, true);

  // Create an extra matrix for R-inverse
  Matrix<T,U,MatrixStructureSquare,Distribution> matrixRI(std::vector<T>(localDimensionX*localDimensionX), localDimensionX, localDimensionX,
    localDimensionX*pGridDimensionSize, localDimensionX*pGridDimensionSize, true);
  CFR3D<T,U,MatrixStructureSquare,MatrixStructureSquare,blasEngine>::Factor(matrixB, matrixR, matrixRI, localDimensionX, 'R', commWorld);

  SquareMM3D<T,U,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare,blasEngine>::Multiply(matrixA, matrixRI,
    matrixQ, localDimensionX, localDimensionY, localDimensionX, commWorld, gemmPack1);
}

template<typename T,typename U,
  template<typename,typename,template<typename,typename,int> class> class StructureA,		// Note: this vould be either rectangular or square.
  template<typename,typename,template<typename,typename,int> class> class StructureQ,		// Note: this vould be either rectangular or square.
  template<typename,typename,template<typename,typename,int> class> class StructureR,		// Note: this vould be either square or triangular
  template<typename,typename> class blasEngine>
template<template<typename,typename,int> class Distribution>
void CholeskyQR2<T,U,StructureA,StructureQ,StructureR,blasEngine>::FactorTunable_cqr(Matrix<T,U,StructureA,Distribution>& matrixA, Matrix<T,U,StructureQ,Distribution>& matrixQ,
    Matrix<T,U,StructureR,Distribution>& matrixR, U localDimensionX, U localDimensionY, MPI_Comm commWorld)
{
  std::cout << "I am in FactorTunable_cqr\n";
}


template<typename T,typename U,
  template<typename,typename,template<typename,typename,int> class> class StructureA,		// Note: this vould be either rectangular or square.
  template<typename,typename,template<typename,typename,int> class> class StructureQ,		// Note: this vould be either rectangular or square.
  template<typename,typename,template<typename,typename,int> class> class StructureR,		// Note: this vould be either square or triangular
  template<typename,typename> class blasEngine>
void CholeskyQR2<T,U,StructureA, StructureQ, StructureR, blasEngine>::BroadcastPanels(
											std::vector<T>& data,
											U size,
											bool isRoot,
											int pGridCoordZ,
											MPI_Comm panelComm
										    )
{
  if (isRoot)
  {
    MPI_Bcast(&data[0], size, MPI_DOUBLE, pGridCoordZ, panelComm);
  }
  else
  {
    data.resize(size);
    MPI_Bcast(&data[0], size, MPI_DOUBLE, pGridCoordZ, panelComm);
  }
}

