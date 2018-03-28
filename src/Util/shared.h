/* Author: Edward Hutter */

#ifndef SHARED
#define SHARED

#ifdef TIMER_TYPE
  #define PROFILE
#endif
#ifdef CRITTER_TYPE
  #define CRITTER
#endif
#ifdef PERF_TYPE
  #define PERFORMANCE
#endif

#ifdef FLOAT_TYPE
  #define DATATYPE float
  #define MPI_DATATYPE MPI_FLOAT
#endif
#ifdef DOUBLE_TYPE
  #define DATATYPE double
  #define MPI_DATATYPE MPI_DOUBLE
#endif
#ifdef COMPLEX_FLOAT_TYPE
  #define DATATYPE std::complex<float>
  #define MPI_DATATYPE MPI_C_FLOAT_COMPLEX
#endif
#ifdef COMPLEX_DOUBLE_TYPE
  #define DATATYPE std::complex<double>
  #define MPI_DATATYPE MPI_C_DOUBLE_COMPLEX
#endif
#ifdef INT_TYPE
  #define INTTYPE int
#endif
#ifdef INT64_T_TYPE
  #define INTTYPE int64_t
#endif

#endif /*SHARED*/
