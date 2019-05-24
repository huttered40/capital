/* Author: Edward Hutter */


template<typename T>
void lapackHelper::setInfoParameters_potrf(
                                          const lapackEngineArgumentPackage_potrf<T>& srcPackage,
                                          int& destArg1,
                                          char& destArg2){
  destArg1 = (srcPackage.order == lapackEngineOrder::AlapackRowMajor ? AlapackRowMajor : AlapackColMajor);
  destArg2 = (srcPackage.uplo == lapackEngineUpLo::AlapackUpper ? 'U' : 'L');
}

template<typename T>
void lapackHelper::setInfoParameters_trtri(
                                          const lapackEngineArgumentPackage_trmm<T>& srcPackage,
                                          int& destArg1,
                                          char& destArg2,
                                          char& destArg3){
  destArg1 = (srcPackage.order == lapackEngineOrder::AlapackRowMajor ? AlapackRowMajor : AlapackColMajor);
  destArg2 = (srcPackage.uplo == lapackEngineUpLo::AlapackUpper ? 'U' : 'L');
  destArg3 = (srcPackage.diag == lapackEngineDiag::AlapackUnit ? 'U' : 'N');
}

template<typename T>
void lapackHelper::setInfoParameters_geqrf(
                                          const lapackEngineArgumentPackage_syrk<T>& srcPackage,
                                          int& destArg1){
  destArg1 = (srcPackage.order == lapackEngineOrder::AlapackRowMajor ? AlapackRowMajor : AlapackColMajor);
}

template<typename T>
void lapackHelper::setInfoParameters_orgqr(
                                          const lapackEngineArgumentPackage_syrk<T>& srcPackage,
                                          int& destArg1){
  destArg1 = (srcPackage.order == lapackEngineOrder::AlapackRowMajor ? AlapackRowMajor : AlapackColMajor);
}

template<typename T, typename U>
void lapackEngine<T,U>::_potrf(
            T* matrixA,
            U n,
            U lda,
            const lapackEngineArgumentPackage_potrf<T>& srcPackage){
  TAU_FSTART(potrf);
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_UPLO arg2;
  setInfoParameters_potrf(srcPackage, arg1, arg2);
#if defined(BGQ) || defined(BLUEWATERS)
  int info;
  _potrf_(&arg2, &n, matrixA, &lda, &info);
#else
  _potrf_(arg1, arg2, n, matrixA, lda);
#endif
  TAU_FSTOP(potrf);
}

template<typename T, typename U>
void lapackEngine<T,U>::_trtri(
            T* matrixA,
            U n,
            U lda,
            const lapackEngineArgumentPackage_trtri<T>& srcPackage){
  TAU_FSTART(trtri);
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  CBLAS_UPLO arg2;
  CBLAS_DIAG arg3;
  setInfoParameters_trtri(srcPackage, arg1, arg2, arg3);

#if defined(BGQ) || defined(BLUEWATERS)
  int info;
  _trtri_(&arg2, &arg3, &n, matrixA, &lda, &info);
#else
  _trtri_(arg1, arg2, arg3, n, matrixA, lda);
#endif
  TAU_FSTOP(trtri);
}

template<typename T, typename U>
void lapackEngine<T,U>::_geqrf(
            T* matrixA,
            T* tau,
            U m,
            U n,
            U lda,
            const lapackEngineArgumentPackage_geqrf<T>& srcPackage){
  TAU_FSTART(geqrf);
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  setInfoParameters_geqrf(srcPackage, arg1);

#if defined(BGQ) || defined(BLUEWATERS)
  int info;
  _geqrf_(&m, &n, matrixA, &lda, tau, &info);
#else
  _geqrf_(arg1, m, n, matrixA, lda, tau);
#endif
  TAU_FSTOP(geqrf);
}

template<typename T, typename U>
void lapackEngine<T,U>::_orgqr(
            T* matrixA,
            T* tau,
            U m,
            U n,
            U k,
            U lda,
            const lapackEngineArgumentPackage_orgqr<T>& srcPackage){
  TAU_FSTART(orgqr);
  // First, unpack the info parameter
  CBLAS_ORDER arg1;
  setInfoParameters_orgqr(srcPackage, arg1);

#if defined(BGQ) || defined(BLUEWATERS)
  int info;
  _orgqr_(&m, &n, &k, matrixA, &lda, tau, &info);
#else
  _orgqr_(arg1, m, n, k, matrixA, lda, tau);
#endif
  TAU_FSTOP(orgqr);
}
