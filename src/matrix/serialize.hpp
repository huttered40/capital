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
  using T = typename SrcType::ScalarType; using U = typename SrcType::DimensionType;
  assert((sex-ssx)==(dex-dsx)); assert((sey-ssy)==(dey-dsy));
  U rangeX = sex-ssx; U rangeY = sey-ssy;
  T* s; if (src_buffer==0) s=src.data(); else if (src_buffer==1) s=src.scratch(); else s=src.pad();
  T* d; if (dest_buffer==0) d=dest.data(); else if (dest_buffer==1) d=dest.scratch(); else d=dest.pad();
  for (U i=0; i<rangeX; i++){
    U dest_idx = dest.offset_local(dsx+i,dsy); U src_idx = src.offset_local(ssx+i,ssy);
    memcpy(&d[dest_idx],&s[src_idx],rangeY*sizeof(T));
  }
}

template<typename SrcType, typename DestType>
void serialize<rect,uppertri>::invoke(const SrcType& src, DestType& dest, typename SrcType::DimensionType ssx, typename SrcType::DimensionType sex, typename SrcType::DimensionType ssy, typename SrcType::DimensionType sey,
                                      typename SrcType::DimensionType dsx, typename SrcType::DimensionType dex, typename SrcType::DimensionType dsy, typename SrcType::DimensionType dey, size_t src_buffer, size_t dest_buffer){
  using T = typename SrcType::ScalarType; using U = typename SrcType::DimensionType;
  assert((sex-ssx)==(dex-dsx)); assert((sey-ssy)==(dey-dsy));
  U rangeX = sex-ssx; U rangeY = sey-ssy;
  T* s; if (src_buffer==0) s=src.data(); else if (src_buffer==1) s=src.scratch(); else s=src.pad();
  T* d; if (dest_buffer==0) d=dest.data(); else if (dest_buffer==1) d=dest.scratch(); else d=dest.pad();
  for (U i=0; i<rangeX; i++){
    U dest_idx = dest.offset_local(dsx+i,dsy); U src_idx = src.offset_local(ssx+i,ssy);
    memcpy(&d[dest_idx],&s[src_idx],(i+1)*sizeof(T));
  }
}

template<typename SrcType, typename DestType>
void serialize<rect,lowertri>::invoke(const SrcType& src, DestType& dest, typename SrcType::DimensionType ssx, typename SrcType::DimensionType sex, typename SrcType::DimensionType ssy, typename SrcType::DimensionType sey,
                                      typename SrcType::DimensionType dsx, typename SrcType::DimensionType dex, typename SrcType::DimensionType dsy, typename SrcType::DimensionType dey, size_t src_buffer, size_t dest_buffer){
  using T = typename SrcType::ScalarType; using U = typename SrcType::DimensionType;
  assert((sex-ssx)==(dex-dsx)); assert((sey-ssy)==(dey-dsy));
  U rangeX = sex-ssx; U rangeY = sey-ssy;
  T* s; if (src_buffer==0) s=src.data(); else if (src_buffer==1) s=src.scratch(); else s=src.pad();
  T* d; if (dest_buffer==0) d=dest.data(); else if (dest_buffer==1) d=dest.scratch(); else d=dest.pad();
  for (U i=0; i<rangeX; i++){
    U dest_idx = dest.offset_local(dsx+i,dsy+i); U src_idx = src.offset_local(ssx+i,ssy+i);
    memcpy(&d[dest_idx],&s[src_idx],(rangeY-i)*sizeof(T));
  }
}

template<typename SrcType, typename DestType>
void serialize<uppertri,rect>::invoke(const SrcType& src, DestType& dest, typename SrcType::DimensionType ssx, typename SrcType::DimensionType sex, typename SrcType::DimensionType ssy, typename SrcType::DimensionType sey,
                                      typename SrcType::DimensionType dsx, typename SrcType::DimensionType dex, typename SrcType::DimensionType dsy, typename SrcType::DimensionType dey, size_t src_buffer, size_t dest_buffer){
  using T = typename SrcType::ScalarType; using U = typename SrcType::DimensionType;
  assert((sex-ssx)==(dex-dsx)); assert((sey-ssy)==(dey-dsy));
  U rangeX = sex-ssx; U rangeY = sey-ssy;
  T* s; if (src_buffer==0) s=src.data(); else if (src_buffer==1) s=src.scratch(); else s=src.pad();
  T* d; if (dest_buffer==0) d=dest.data(); else if (dest_buffer==1) d=dest.scratch(); else d=dest.pad();
  for (U i=0; i<rangeX; i++){
    U dest_idx = dest.offset_local(dsx+i,dsy); U src_idx = src.offset_local(ssx+i,ssy);
    memcpy(&d[dest_idx],&s[src_idx],(i+1)*sizeof(T));
  }
}

template<typename SrcType, typename DestType>
void serialize<uppertri,uppertri>::invoke(const SrcType& src, DestType& dest, typename SrcType::DimensionType ssx, typename SrcType::DimensionType sex, typename SrcType::DimensionType ssy, typename SrcType::DimensionType sey,
                                          typename SrcType::DimensionType dsx, typename SrcType::DimensionType dex, typename SrcType::DimensionType dsy, typename SrcType::DimensionType dey, size_t src_buffer, size_t dest_buffer){
  using T = typename SrcType::ScalarType; using U = typename SrcType::DimensionType;
  // debug
  int rank; MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  assert((sex-ssx)==(dex-dsx)); assert((sey-ssy)==(dey-dsy));
  U rangeX = sex-ssx; U rangeY = sey-ssy;
  T* s; if (src_buffer==0) s=src.data(); else if (src_buffer==1) s=src.scratch(); else s=src.pad();
  T* d; if (dest_buffer==0) d=dest.data(); else if (dest_buffer==1) d=dest.scratch(); else d=dest.pad();
  for (U i=0; i<rangeX; i++){
    U dest_idx = dest.offset_local(dsx+i,dsy); U src_idx = src.offset_local(ssx+i,ssy);
    memcpy(&d[dest_idx],&s[src_idx],(i+1)*sizeof(T));
  }
}

template<typename SrcType, typename DestType>
void serialize<lowertri,rect>::invoke(const SrcType& src, DestType& dest, typename SrcType::DimensionType ssx, typename SrcType::DimensionType sex, typename SrcType::DimensionType ssy, typename SrcType::DimensionType sey,
                                      typename SrcType::DimensionType dsx, typename SrcType::DimensionType dex, typename SrcType::DimensionType dsy, typename SrcType::DimensionType dey, size_t src_buffer, size_t dest_buffer){
  using T = typename SrcType::ScalarType; using U = typename SrcType::DimensionType;
  assert((sex-ssx)==(dex-dsx)); assert((sey-ssy)==(dey-dsy));
  U rangeX = sex-ssx; U rangeY = sey-ssy;
  T* s; if (src_buffer==0) s=src.data(); else if (src_buffer==1) s=src.scratch(); else s=src.pad();
  T* d; if (dest_buffer==0) d=dest.data(); else if (dest_buffer==1) d=dest.scratch(); else d=dest.pad();
  for (U i=0; i<rangeX; i++){
    U dest_idx = dest.offset_local(dsx+i,dsy+i); U src_idx = src.offset_local(ssx+i,ssy+i);
    memcpy(&d[dest_idx],&s[src_idx],(rangeY-i)*sizeof(T));
  }
}

template<typename SrcType, typename DestType>
void serialize<lowertri,lowertri>::invoke(const SrcType& src, DestType& dest, typename SrcType::DimensionType ssx, typename SrcType::DimensionType sex, typename SrcType::DimensionType ssy, typename SrcType::DimensionType sey,
                                          typename SrcType::DimensionType dsx, typename SrcType::DimensionType dex, typename SrcType::DimensionType dsy, typename SrcType::DimensionType dey, size_t src_buffer, size_t dest_buffer){
  using T = typename SrcType::ScalarType; using U = typename SrcType::DimensionType;
  assert((sex-ssx)==(dex-dsx)); assert((sey-ssy)==(dey-dsy));
  U rangeX = sex-ssx; U rangeY = sey-ssy;
  T* s; if (src_buffer==0) s=src.data(); else if (src_buffer==1) s=src.scratch(); else s=src.pad();
  T* d; if (dest_buffer==0) d=dest.data(); else if (dest_buffer==1) d=dest.scratch(); else d=dest.pad();
  for (U i=0; i<rangeX; i++){
    U dest_idx = dest.offset_local(dsx+i,dsy+i); U src_idx = src.offset_local(ssx+i,ssy+i);
    memcpy(&d[dest_idx],&s[src_idx],(rangeY-i)*sizeof(T));
  }
}


template<typename SrcType>
void serialize<uppertri,rect>::invoke(SrcType& src){

  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  // But now, I am going to have this call the serialize from UT to square, because thats what this will actually be doing
  // I tried a simple static_cast, but it didn't work, so now I will just copy code. Ugh! Fix later.
  using T = typename SrcType::ScalarType; using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local(); U srcNumColumns = src.num_columns_local();
  T* srcVectorData = src.scratch(); T* destVectorData = src.pad();
  U counter{1}; U srcOffset{0}; U destOffset{0}; U counter2{srcNumRows};
  for (U i=0; i<srcNumRows; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], counter*sizeof(T));
    U fillZeros = srcNumRows-counter;
    fillZerosContig(&destVectorData[destOffset+counter], fillZeros);
    srcOffset += counter;
    destOffset += counter2;
    counter++;
  }
}


template<typename SrcType>
void serialize<lowertri,rect>::invoke(SrcType& src){

  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  // But now, I am going to have this call the serialize from UT to square, because thats what this will actually be doing
  // I tried a simple static_cast, but it didn't work, so now I will just copy code. Ugh! Fix later.
  using T = typename SrcType::ScalarType; using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local(); U srcNumColumns = src.num_columns_local();
  T* srcVectorData = src.scratch(); T* destVectorData = src.pad();
  U counter{srcNumRows}; U srcOffset{0}; U destOffset{0};
  for (U i=0; i<srcNumColumns; i++){
    fillZerosContig(&destVectorData[destOffset], i);
    memcpy(&destVectorData[destOffset+i], &srcVectorData[srcOffset], counter*sizeof(T));
    srcOffset += counter;
    destOffset += srcNumRows;
    counter--;
  }
}
