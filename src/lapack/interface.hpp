/* Author: Edward Hutter */

namespace lapack{
void lapackHelper::setInfoParameters_potrf(
                                          const lapackEngineArgumentPackage_potrf& srcPackage,
                                          int& destArg1,
                                          char& destArg2){
  destArg1 = (srcPackage.order == lapackEngineOrder::AlapackRowMajor ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR);
  destArg2 = (srcPackage.uplo == lapackEngineUpLo::AlapackUpper ? 'U' : 'L');
}

void lapackHelper::setInfoParameters_trtri(
                                          const lapackEngineArgumentPackage_trtri& srcPackage,
                                          int& destArg1,
                                          char& destArg2,
                                          char& destArg3){
  destArg1 = (srcPackage.order == lapackEngineOrder::AlapackRowMajor ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR);
  destArg2 = (srcPackage.uplo == lapackEngineUpLo::AlapackUpper ? 'U' : 'L');
  destArg3 = (srcPackage.diag == lapackEngineDiag::AlapackUnit ? 'U' : 'N');
}

void lapackHelper::setInfoParameters_geqrf(
                                          const lapackEngineArgumentPackage_geqrf& srcPackage,
                                          int& destArg1){
  destArg1 = (srcPackage.order == lapackEngineOrder::AlapackRowMajor ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR);
}

void lapackHelper::setInfoParameters_orgqr(
                                          const lapackEngineArgumentPackage_orgqr& srcPackage,
                                          int& destArg1){
  destArg1 = (srcPackage.order == lapackEngineOrder::AlapackRowMajor ? LAPACK_ROW_MAJOR : LAPACK_COL_MAJOR);
}

template<typename T>
void lapackEngine::_potrf(
            T* matrixA,
            int n,
            int lda,
            const lapackEngineArgumentPackage_potrf& srcPackage){
  TAU_FSTART(potrf);
  // First, unpack the info parameter
  int arg1; char arg2;
  setInfoParameters_potrf(srcPackage, arg1, arg2);

  auto _potrf_ = GetPOTRFroutine(LType<T>());
#if defined(BGQ) || defined(BLUEWATERS)
  int info;
  _potrf_(&arg2, &n, matrixA, &lda, &info);
#else
  _potrf_(arg1, arg2, n, matrixA, lda);
#endif
  TAU_FSTOP(potrf);
}

template<typename T>
void lapackEngine::_trtri(
            T* matrixA,
            int n,
            int lda,
            const lapackEngineArgumentPackage_trtri& srcPackage){
  TAU_FSTART(trtri);
  // First, unpack the info parameter
  int arg1; char arg2; char arg3;
  setInfoParameters_trtri(srcPackage, arg1, arg2, arg3);

  static auto _trtri_ = GetTRTRIroutine(LType<T>());
#if defined(BGQ) || defined(BLUEWATERS)
  int info;
  _trtri_(&arg2, &arg3, &n, matrixA, &lda, &info);
#else
  _trtri_(arg1, arg2, arg3, n, matrixA, lda);
#endif
  TAU_FSTOP(trtri);
}

template<typename T>
void lapackEngine::_geqrf(
            T* matrixA,
            T* tau,
            int m,
            int n,
            int lda,
            const lapackEngineArgumentPackage_geqrf& srcPackage){
  TAU_FSTART(geqrf);
  // First, unpack the info parameter
  int arg1;
  setInfoParameters_geqrf(srcPackage, arg1);

  auto _geqrf_ = GetGEQRFroutine(LType<T>());
#if defined(BGQ) || defined(BLUEWATERS)
  int info;
  _geqrf_(&m, &n, matrixA, &lda, tau, &info);
#else
  _geqrf_(arg1, m, n, matrixA, lda, tau);
#endif
  TAU_FSTOP(geqrf);
}

template<typename T>
void lapackEngine::_orgqr(
            T* matrixA,
            T* tau,
            int m,
            int n,
            int k,
            int lda,
            const lapackEngineArgumentPackage_orgqr& srcPackage){
  TAU_FSTART(orgqr);
  // First, unpack the info parameter
  int arg1;
  setInfoParameters_orgqr(srcPackage, arg1);

  auto _orgqr_ = GetORGQRroutine(LType<T>());
#if defined(BGQ) || defined(BLUEWATERS)
  int info;
  _orgqr_(&m, &n, &k, matrixA, &lda, tau, &info);
#else
  _orgqr_(arg1, m, n, k, matrixA, lda, tau);
#endif
  TAU_FSTOP(orgqr);
}
}
