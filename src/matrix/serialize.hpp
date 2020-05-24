/* Author: Edward Hutter */


// Helper static method -- fills a range with zeros
template<typename T, typename U>
static void fillZerosContig(T* addr, U size){
  for (U i=0; i<size; i++){
    addr[i] = 0;
  }
}

template<typename SrcType, typename DestType>
void serialize<rect,rect>::invoke(const SrcType& src, DestType& dest, typename SrcType::DimensionType ssx, typename SrcType::DimensionType sex, typename SrcType::DimensionType ssy, typename SrcType::DimensionType sey,
                                  typename SrcType::DimensionType dsx, typename SrcType::DimensionType dex, typename SrcType::DimensionType dsy, typename SrcType::DimensionType dey, size_t src_buffer, size_t dest_buffer){
#ifdef FUNCTION_SYMBOLS
CRITTER_START(serialize);
#endif
  using T = typename SrcType::ScalarType; using U = typename SrcType::DimensionType;
  assert((sex-ssx)==(dex-dsx)); assert((sey-ssy)==(dey-dsy));
  U rangeX = sex-ssx; U rangeY = sey-ssy;
  T* s; if (src_buffer==0) s=src.data(); else if (src_buffer==1) s=src.scratch(); else s=src.pad();
  T* d; if (dest_buffer==0) d=dest.data(); else if (dest_buffer==1) d=dest.scratch(); else d=dest.pad();
  for (U i=0; i<rangeX; i++){
    U dest_idx = dest.offset_local(dsx+i,dsy,dest_buffer); U src_idx = src.offset_local(ssx+i,ssy,src_buffer);
    memcpy(&d[dest_idx],&s[src_idx],rangeY*sizeof(T));
  }
#ifdef FUNCTION_SYMBOLS
CRITTER_STOP(serialize);
#endif
}

template<typename SrcType, typename DestType>
void serialize<rect,uppertri>::invoke(const SrcType& src, DestType& dest, typename SrcType::DimensionType ssx, typename SrcType::DimensionType sex, typename SrcType::DimensionType ssy, typename SrcType::DimensionType sey,
                                      typename SrcType::DimensionType dsx, typename SrcType::DimensionType dex, typename SrcType::DimensionType dsy, typename SrcType::DimensionType dey, size_t src_buffer, size_t dest_buffer){
#ifdef FUNCTION_SYMBOLS
CRITTER_START(serialize);
#endif
  using T = typename SrcType::ScalarType; using U = typename SrcType::DimensionType;
  assert((sex-ssx)==(dex-dsx)); assert((sey-ssy)==(dey-dsy));
  U rangeX = sex-ssx; U rangeY = sey-ssy;
  T* s; if (src_buffer==0) s=src.data(); else if (src_buffer==1) s=src.scratch(); else s=src.pad();
  T* d; if (dest_buffer==0) d=dest.data(); else if (dest_buffer==1) d=dest.scratch(); else d=dest.pad();
  for (U i=0; i<rangeX; i++){
    U dest_idx = dest.offset_local(dsx+i,dsy,dest_buffer); U src_idx = src.offset_local(ssx+i,ssy,src_buffer);
    memcpy(&d[dest_idx],&s[src_idx],(i+1)*sizeof(T));
  }
#ifdef FUNCTION_SYMBOLS
CRITTER_STOP(serialize);
#endif
}

template<typename SrcType, typename DestType>
void serialize<rect,lowertri>::invoke(const SrcType& src, DestType& dest, typename SrcType::DimensionType ssx, typename SrcType::DimensionType sex, typename SrcType::DimensionType ssy, typename SrcType::DimensionType sey,
                                      typename SrcType::DimensionType dsx, typename SrcType::DimensionType dex, typename SrcType::DimensionType dsy, typename SrcType::DimensionType dey, size_t src_buffer, size_t dest_buffer){
#ifdef FUNCTION_SYMBOLS
CRITTER_START(serialize);
#endif
  using T = typename SrcType::ScalarType; using U = typename SrcType::DimensionType;
  assert((sex-ssx)==(dex-dsx)); assert((sey-ssy)==(dey-dsy));
  U rangeX = sex-ssx; U rangeY = sey-ssy;
  T* s; if (src_buffer==0) s=src.data(); else if (src_buffer==1) s=src.scratch(); else s=src.pad();
  T* d; if (dest_buffer==0) d=dest.data(); else if (dest_buffer==1) d=dest.scratch(); else d=dest.pad();
  for (U i=0; i<rangeX; i++){
    U dest_idx = dest.offset_local(dsx+i,dsy+i,dest_buffer); U src_idx = src.offset_local(ssx+i,ssy+i,src_buffer);
    memcpy(&d[dest_idx],&s[src_idx],(rangeY-i)*sizeof(T));
  }
#ifdef FUNCTION_SYMBOLS
CRITTER_STOP(serialize);
#endif
}

template<typename SrcType, typename DestType>
void serialize<uppertri,rect>::invoke(const SrcType& src, DestType& dest, typename SrcType::DimensionType ssx, typename SrcType::DimensionType sex, typename SrcType::DimensionType ssy, typename SrcType::DimensionType sey,
                                      typename SrcType::DimensionType dsx, typename SrcType::DimensionType dex, typename SrcType::DimensionType dsy, typename SrcType::DimensionType dey, size_t src_buffer, size_t dest_buffer){
#ifdef FUNCTION_SYMBOLS
CRITTER_START(serialize);
#endif
  using T = typename SrcType::ScalarType; using U = typename SrcType::DimensionType;
  assert((sex-ssx)==(dex-dsx)); assert((sey-ssy)==(dey-dsy));
  U rangeX = sex-ssx; U rangeY = sey-ssy;
  T* s; if (src_buffer==0) s=src.data(); else if (src_buffer==1) s=src.scratch(); else s=src.pad();
  T* d; if (dest_buffer==0) d=dest.data(); else if (dest_buffer==1) d=dest.scratch(); else d=dest.pad();
  for (U i=0; i<rangeX; i++){
    U dest_idx = dest.offset_local(dsx+i,dsy,dest_buffer); U src_idx = src.offset_local(ssx+i,ssy,src_buffer);
    memcpy(&d[dest_idx],&s[src_idx],(i+1)*sizeof(T));
  }
#ifdef FUNCTION_SYMBOLS
CRITTER_STOP(serialize);
#endif
}

template<typename SrcType, typename DestType>
void serialize<uppertri,uppertri>::invoke(const SrcType& src, DestType& dest, typename SrcType::DimensionType ssx, typename SrcType::DimensionType sex, typename SrcType::DimensionType ssy, typename SrcType::DimensionType sey,
                                          typename SrcType::DimensionType dsx, typename SrcType::DimensionType dex, typename SrcType::DimensionType dsy, typename SrcType::DimensionType dey, size_t src_buffer, size_t dest_buffer){
#ifdef FUNCTION_SYMBOLS
CRITTER_START(serialize);
#endif
  using T = typename SrcType::ScalarType; using U = typename SrcType::DimensionType;
  assert((sex-ssx)==(dex-dsx)); assert((sey-ssy)==(dey-dsy));
  U rangeX = sex-ssx; U rangeY = sey-ssy;
  T* s; if (src_buffer==0) s=src.data(); else if (src_buffer==1) s=src.scratch(); else s=src.pad();
  T* d; if (dest_buffer==0) d=dest.data(); else if (dest_buffer==1) d=dest.scratch(); else d=dest.pad();
  for (U i=0; i<rangeX; i++){
    U dest_idx = dest.offset_local(dsx+i,dsy,dest_buffer); U src_idx = src.offset_local(ssx+i,ssy,src_buffer);
    memcpy(&d[dest_idx],&s[src_idx],(i+1)*sizeof(T));
  }
#ifdef FUNCTION_SYMBOLS
CRITTER_STOP(serialize);
#endif
}

template<typename SrcType, typename DestType>
void serialize<lowertri,rect>::invoke(const SrcType& src, DestType& dest, typename SrcType::DimensionType ssx, typename SrcType::DimensionType sex, typename SrcType::DimensionType ssy, typename SrcType::DimensionType sey,
                                      typename SrcType::DimensionType dsx, typename SrcType::DimensionType dex, typename SrcType::DimensionType dsy, typename SrcType::DimensionType dey, size_t src_buffer, size_t dest_buffer){
#ifdef FUNCTION_SYMBOLS
CRITTER_START(serialize);
#endif
  using T = typename SrcType::ScalarType; using U = typename SrcType::DimensionType;
  assert((sex-ssx)==(dex-dsx)); assert((sey-ssy)==(dey-dsy));
  U rangeX = sex-ssx; U rangeY = sey-ssy;
  T* s; if (src_buffer==0) s=src.data(); else if (src_buffer==1) s=src.scratch(); else s=src.pad();
  T* d; if (dest_buffer==0) d=dest.data(); else if (dest_buffer==1) d=dest.scratch(); else d=dest.pad();
  for (U i=0; i<rangeX; i++){
    U dest_idx = dest.offset_local(dsx+i,dsy+i,dest_buffer); U src_idx = src.offset_local(ssx+i,ssy+i,src_buffer);
    memcpy(&d[dest_idx],&s[src_idx],(rangeY-i)*sizeof(T));
  }
#ifdef FUNCTION_SYMBOLS
CRITTER_STOP(serialize);
#endif
}

template<typename SrcType, typename DestType>
void serialize<lowertri,lowertri>::invoke(const SrcType& src, DestType& dest, typename SrcType::DimensionType ssx, typename SrcType::DimensionType sex, typename SrcType::DimensionType ssy, typename SrcType::DimensionType sey,
                                          typename SrcType::DimensionType dsx, typename SrcType::DimensionType dex, typename SrcType::DimensionType dsy, typename SrcType::DimensionType dey, size_t src_buffer, size_t dest_buffer){
#ifdef FUNCTION_SYMBOLS
CRITTER_START(serialize);
#endif
  using T = typename SrcType::ScalarType; using U = typename SrcType::DimensionType;
  assert((sex-ssx)==(dex-dsx)); assert((sey-ssy)==(dey-dsy));
  U rangeX = sex-ssx; U rangeY = sey-ssy;
  T* s; if (src_buffer==0) s=src.data(); else if (src_buffer==1) s=src.scratch(); else s=src.pad();
  T* d; if (dest_buffer==0) d=dest.data(); else if (dest_buffer==1) d=dest.scratch(); else d=dest.pad();
  for (U i=0; i<rangeX; i++){
    U dest_idx = dest.offset_local(dsx+i,dsy+i,dest_buffer); U src_idx = src.offset_local(ssx+i,ssy+i,src_buffer);
    memcpy(&d[dest_idx],&s[src_idx],(rangeY-i)*sizeof(T));
  }
#ifdef FUNCTION_SYMBOLS
CRITTER_STOP(serialize);
#endif
}
