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

template<typename T>
void engine::_potrf(T* matrixA, int n, int lda, const ArgPack_potrf& srcPackage){
  // First, unpack the info parameter
  int arg1; char arg2;
  helper::setInfoParameters_potrf(srcPackage, arg1, arg2);

  auto _potrf_ = GetPOTRFroutine(LType<T>());
#if defined(BGQ) || defined(BLUEWATERS)
  int info;
  _potrf_(&arg2, &n, matrixA, &lda, &info);
#else
  _potrf_(arg1, arg2, n, matrixA, lda);
#endif
}

template<typename T>
void engine::_trtri(T* matrixA, int n, int lda, const ArgPack_trtri& srcPackage){
  // First, unpack the info parameter
  int arg1; char arg2; char arg3;
  helper::setInfoParameters_trtri(srcPackage, arg1, arg2, arg3);

  static auto _trtri_ = GetTRTRIroutine(LType<T>());
#if defined(BGQ) || defined(BLUEWATERS)
  int info;
  _trtri_(&arg2, &arg3, &n, matrixA, &lda, &info);
#else
  _trtri_(arg1, arg2, arg3, n, matrixA, lda);
#endif
}

template<typename T>
void engine::_geqrf(T* matrixA, T* tau, int m, int n, int lda, const ArgPack_geqrf& srcPackage){
  // First, unpack the info parameter
  int arg1;
  helper::setInfoParameters_geqrf(srcPackage, arg1);

  auto _geqrf_ = GetGEQRFroutine(LType<T>());
#if defined(BGQ) || defined(BLUEWATERS)
  int info;
  _geqrf_(&m, &n, matrixA, &lda, tau, &info);
#else
  _geqrf_(arg1, m, n, matrixA, lda, tau);
#endif
}

template<typename T>
void engine::_orgqr(T* matrixA, T* tau, int m, int n, int k, int lda, const ArgPack_orgqr& srcPackage){
  // First, unpack the info parameter
  int arg1;
  helper::setInfoParameters_orgqr(srcPackage, arg1);

  auto _orgqr_ = GetORGQRroutine(LType<T>());
#if defined(BGQ) || defined(BLUEWATERS)
  int info;
  _orgqr_(&m, &n, &k, matrixA, &lda, tau, &info);
#else
  _orgqr_(arg1, m, n, k, matrixA, lda, tau);
#endif
}
}
