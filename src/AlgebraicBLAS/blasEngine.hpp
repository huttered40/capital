/* Author: Edward Hutter */


template<typename T, typename U>
void blasEngine<T,U,MatrixStructureSquare,MatrixStructureSquare,MatrixStructureSquare>::
  multiply(
            T* matrixA,
            T* matrixB,
            T* matrixC,
            U matrixAdimX,
            U matrixAdimY,
            U matrixBdimX,
            U matrixBdimZ,
            U matrixCdimY,
            U matrixCdimZ
          )
{
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, matrixAdimY, matrixCdimZ, matrixAdimX, 1.,
    matrixA, matrixAdimY, matrixB, matrixBdimX, 1., matrixC, matrixCdimY);
}
