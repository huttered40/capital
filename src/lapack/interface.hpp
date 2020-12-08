/* Author: Edward Hutter */

namespace lapack{
void helper::setInfoParameters_potrf(const ArgPack_potrf& srcPackage,
                                     int& destArg1,
                                     char& destArg2){
  destArg1 = (srcPackage.order == Order::AlapackRowMajor ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR);
  destArg2 = (srcPackage.uplo == UpLo::AlapackUpper ? 'U' : 'L');
}

void helper::setInfoParameters_trtri(const ArgPack_trtri& srcPackage,
                                     int& destArg1,
                                     char& destArg2,
                                     char& destArg3){
  destArg1 = (srcPackage.order == Order::AlapackRowMajor ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR);
  destArg2 = (srcPackage.uplo == UpLo::AlapackUpper ? 'U' : 'L');
  destArg3 = (srcPackage.diag == Diag::AlapackUnit ? 'U' : 'N');
}

void helper::setInfoParameters_geqrf(const ArgPack_geqrf& srcPackage,
                                     int& destArg1){
  destArg1 = (srcPackage.order == Order::AlapackRowMajor ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR);
}

void helper::setInfoParameters_orgqr(const ArgPack_orgqr& srcPackage,
                                     int& destArg1){
  destArg1 = (srcPackage.order == Order::AlapackRowMajor ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR);
}

template<>
void engine::_potrf(double* matrixA, int n, int lda, const ArgPack_potrf& srcPackage){
  // First, unpack the info parameter
  int arg1; char arg2;
  helper::setInfoParameters_potrf(srcPackage, arg1, arg2);

#ifdef FUNCTION_SYMBOLS
CRITTER_START(potrf);
#endif
  LAPACKE_dpotrf(arg1, arg2, n, matrixA, lda);
#ifdef FUNCTION_SYMBOLS
CRITTER_STOP(potrf);
#endif
}

template<>
void engine::_trtri(double* matrixA, int n, int lda, const ArgPack_trtri& srcPackage){
  // First, unpack the info parameter
  int arg1; char arg2; char arg3;
  helper::setInfoParameters_trtri(srcPackage, arg1, arg2, arg3);

#ifdef FUNCTION_SYMBOLS
CRITTER_START(trtri);
#endif
  LAPACKE_dtrtri(arg1, arg2, arg3, n, matrixA, lda);
#ifdef FUNCTION_SYMBOLS
CRITTER_STOP(trtri);
#endif
}

template<>
void engine::_geqrf(double* matrixA, double* tau, int m, int n, int lda, const ArgPack_geqrf& srcPackage){
  // First, unpack the info parameter
  int arg1;
  helper::setInfoParameters_geqrf(srcPackage, arg1);

#ifdef FUNCTION_SYMBOLS
CRITTER_START(geqrf);
#endif
  LAPACKE_dgeqrf(arg1, m, n, matrixA, lda, tau);
#ifdef FUNCTION_SYMBOLS
CRITTER_STOP(trtri);
#endif
}

template<>
void engine::_orgqr(double* matrixA, double* tau, int m, int n, int k, int lda, const ArgPack_orgqr& srcPackage){
  // First, unpack the info parameter
  int arg1;
  helper::setInfoParameters_orgqr(srcPackage, arg1);

#ifdef FUNCTION_SYMBOLS
CRITTER_START(orgqr);
#endif
  LAPACKE_dorgqr(arg1, m, n, k, matrixA, lda, tau);
#ifdef FUNCTION_SYMBOLS
CRITTER_STOP(orgqr);
#endif
}
}
