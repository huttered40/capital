/* Author: Edward Hutter */

#ifndef SHARED
#define SHARED

#ifdef TIMER_TYPE
  #define PROFILE
#endif
#ifdef CRITTER_TYPE
  #define CRITTER
#endif

#ifdef FLOAT_TYPE
  #define DATATYPE float
#endif
#ifdef DOUBLE_TYPE
  #define DATATYPE double
#endif
#ifdef COMPLEX_FLOAT_TYPE
  #define DATATYPE std::complex<float>
#endif
#ifdef COMPLEX_DOUBLE_TYPE
  #define DATATYPE std::complex<double>
#endif
#ifdef INT_TYPE
  #define INTTYPE int
#endif
#ifdef INT64_T_TYPE
  #define INTTYPE int64_t
#endif

#endif /*SHARED*/
