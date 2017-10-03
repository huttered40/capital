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
  // we basically allow both Rectangular and Square structures

  // Also note, we need a helper routine for the CholeskyQR stuff. We want these methods to be the CholeskyQR2 routines.

  // set up a syrk
  // MPI_AllReduce
  // Sequential calls to blas and lapack
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
  std::cout << "I am in Factor1D_cqr\n";
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
