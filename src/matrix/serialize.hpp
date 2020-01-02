/* Author: Edward Hutter */

/*
  Note: For bool dir, if dir == false, we serialize the bigger object into the smaller one, using the necessary indexing
                      if dir == true, we serialize the smaller object into the bigger one, using the opposite indexing as above

                      Both directions use source and dest buffers as the name suggests

	Also, we assume that the source bufer is allocated properly. But, the destination buffer is checked for appropriate size to prevent seg faults
              ans also to aid the user of thi Policy if he doesn't know the correct number of elements given th complicated Structure Policy abstraction.
*/


// Helper static method -- fills a range with zeros
template<typename T, typename U>
static void fillZerosContig(T* addr, U size){
  for (U i=0; i<size; i++){
    addr[i] = 0;
  }
}

template<typename SrcType, typename DestType>
void serialize<rect,rect>::invoke(SrcType& src, DestType& dest){

  // For now, just call square counterpart, it should be the same --- Actually I can't unless I try to do a weird cast.
  // Annoying code bloat here
  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();
  U srcNumElems = srcNumRows*srcNumColumns;

  T* srcVectorData = src.data();
  T* destVectorData = dest.data();

  // direction doesn't matter here since no indexing here
  memcpy(&destVectorData[0], &srcVectorData[0], sizeof(T)*srcNumElems);
  return;
}

template<typename BigType, typename SmallType>
void serialize<rect,rect>::invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                    typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){

  // For now, just call square counterpart, it should be the same --- Actually I can't unless I try to do a weird cast.
  // Annoying code bloat here
  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
  U bigNumRows = big.num_rows_local();
  U bigNumColumns = big.num_columns_local();

  T* bigVectorData = big.data();
  T* smallVectorData = small.data();

  U destIndex = (dir ? cutDimensionYstart+bigNumRows*cutDimensionXstart : 0);
  U srcIndex = (dir ? 0 : cutDimensionYstart+bigNumRows*cutDimensionXstart);
  U srcCounter = (dir ? rangeY : bigNumRows);
  U destCounter = (dir ? bigNumRows : rangeY);
  T* destVectorData = dir ? bigVectorData : smallVectorData;
  T* srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destIndex], &srcVectorData[srcIndex], sizeof(T)*rangeY);		// rangeX is fine. blocks of size rangeX are still being copied.
    destIndex += destCounter;
    srcIndex += srcCounter;
  }
  return;
}

template<typename SrcType, typename DestType>
void serialize<rect,uppertri>::invoke(SrcType& src, DestType& dest){
  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();

  T* srcVectorData = src.data();
  T* destVectorData = dest.data();

  U counter{1};
  U srcOffset{0};
  U destOffset{0};
  U counter2{srcNumColumns};
  for (U i=0; i<srcNumColumns; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], counter*sizeof(T));
    srcOffset += counter2;
    destOffset += counter;
    counter++;
  }
  return;
}

template<typename BigType, typename SmallType>
void serialize<rect,uppertri>::invoke(BigType& src, SmallType& dest, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){
  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  U bigNumRows = src.num_rows_local();
  U bigNumColumns = src.num_columns_local();

  T* bigVectorData = src.data();
  T* smallVectorData = dest.data();

  // We assume that if dir==true, the user passed in a destination matrix that is properly sized
  // If I find a use case where that is not true, then I can pass in big x and y dimensions, but I do not wan to do that.
  // In most cases, the lae matri will be known and we are simply trying to fill in the smaller into the existing bigger

  U destIndex = (dir ? cutDimensionYstart+bigNumRows*cutDimensionXstart : 0);
  U counter{1};
  U srcIndexSave = (dir ? 0 : cutDimensionYstart+bigNumRows*cutDimensionXstart);
  T* destVectorData = (dir ? bigVectorData : smallVectorData);
  T* srcVectorData = (dir ? smallVectorData : bigVectorData);
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destIndex], &srcVectorData[srcIndexSave], sizeof(T)*counter);
    destIndex += (dir ? bigNumRows : counter);
    srcIndexSave += (dir ? counter : bigNumRows);
    counter++;
  }
}

template<typename SrcType, typename DestType>
void serialize<rect,lowertri>::invoke(SrcType& src, DestType& dest){
  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();
  U destNumRows = dest.num_rows_local();
  U destNumColumns = dest.num_columns_local();

  T* srcVectorData = src.data();
  T* destVectorData = dest.data();

  U counter{srcNumRows};
  U srcOffset{0};
  U destOffset{0};
  U counter2{srcNumRows+1};
  for (U i=0; i<srcNumColumns; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], counter*sizeof(T));
    srcOffset += counter2;
    destOffset += counter;
    counter--;
  }
  return;
}

template<typename BigType, typename SmallType>
void serialize<rect,lowertri>::invoke(BigType& src, SmallType& dest, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){
  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  U bigNumRows = src.num_rows_local();
  U bigNumColumns = src.num_columns_local();

  T* bigVectorData = src.data();
  T* smallVectorData = dest.data();

  U counter{rangeY};
  U srcOffset = (dir ? 0 : cutDimensionYstart+bigNumRows*cutDimensionXstart);
  U destOffset = (dir ? cutDimensionYstart+bigNumRows*cutDimensionXstart  : 0);
  T* destVectorData = dir ? bigVectorData : smallVectorData;
  T* srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*counter);
    destOffset += (dir ? bigNumRows+1 : counter);
    srcOffset += (dir ? counter : bigNumRows+1);
    counter--;
  }
}


template<typename SrcType>
void serialize<uppertri,rect>::invoke(SrcType& src){

  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  // But now, I am going to have this call the serialize from UT to square, because thats what this will actually be doing
  // I tried a simple static_cast, but it didn't work, so now I will just copy code. Ugh! Fix later.
  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();

  T* srcVectorData = src.scratch();
  T* destVectorData = src.pad();

  U counter{1};
  U srcOffset{0};
  U destOffset{0};
  U counter2{srcNumRows};
  for (U i=0; i<srcNumRows; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], counter*sizeof(T));
    U fillZeros = srcNumRows-counter;
    fillZerosContig(&destVectorData[destOffset+counter], fillZeros);
    srcOffset += counter;
    destOffset += counter2;
    counter++;
  }
  return;
}

template<typename SrcType, typename DestType>
void serialize<uppertri,rect>::invoke(SrcType& src, DestType& dest){

  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  // But now, I am going to have this call the serialize from UT to square, because thats what this will actually be doing
  // I tried a simple static_cast, but it didn't work, so now I will just copy code. Ugh! Fix later.
  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();

  T* srcVectorData = src.data();
  T* destVectorData = dest.data();

  U counter{1};
  U srcOffset{0};
  U destOffset{0};
  U counter2{srcNumRows};
  for (U i=0; i<srcNumRows; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], counter*sizeof(T));
    U fillZeros = srcNumRows-counter;
    fillZerosContig(&destVectorData[destOffset+counter], fillZeros);
    srcOffset += counter;
    destOffset += counter2;
    counter++;
  }
  return;
}


template<typename BigType, typename SmallType>
void serialize<uppertri,rect>::invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                      typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){

  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  // But now, I am going to have this call the serialize from UT to square, because thats what this will actually be doing
  // I tried a simple static_cast, but it didn't work, so now I will just copy code. Ugh! Fix later.
  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  U bigNumColumns = big.num_columns_local();

  T* bigVectorData = big.data();
  T* smallVectorData = small.data();

  U bigMatOffset = ((cutDimensionXstart*(cutDimensionXstart+1))>>1);
  bigMatOffset += cutDimensionYstart;
  U bigMatCounter = cutDimensionXstart;
  U srcOffset = (dir ? 0 : bigMatOffset);
  U destOffset = (dir ? bigMatOffset : 0);
  T* destVectorData = dir ? bigVectorData : smallVectorData;
  T* srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*rangeY);
    destOffset += (dir ? (bigMatCounter+1) : rangeY);
    srcOffset += (dir ? rangeY : (bigMatCounter+1));
    bigMatCounter++;
  }
}


template<typename SrcType, typename DestType>
void serialize<uppertri,uppertri>::invoke(SrcType& src, DestType& dest){

  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();
  U srcNumElems = srcNumRows*srcNumColumns;

  T* srcVectorData = src.data();
  T* destVectorData = dest.data();

  // direction doesn't matter here since no indexing here
  memcpy(&destVectorData[0], &srcVectorData[0], sizeof(T)*srcNumElems);
  return;
}


template<typename BigType, typename SmallType>
void serialize<uppertri,uppertri>::invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                            typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){

  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  U bigNumColumns = big.num_columns_local();

  T* bigVectorData = big.data();
  T* smallVectorData = small.data();

  U bigMatOffset = ((cutDimensionXstart*(cutDimensionXstart+1))>>1);
  bigMatOffset += cutDimensionYstart;
  U bigMatCounter = cutDimensionXstart;
  U smallMatOffset = 0;
  U smallMatCounter{1};
  U srcOffset = (dir ? smallMatOffset : bigMatOffset);
  U destOffset = (dir ? bigMatOffset : smallMatOffset);
  T* destVectorData = dir ? bigVectorData : smallVectorData;
  T* srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*smallMatCounter);
    destOffset += (dir ? (bigMatCounter+1) : smallMatCounter);
    srcOffset += (dir ? smallMatCounter : (bigMatCounter+1));
    bigMatCounter++;
    smallMatCounter++;
  }
  return;
}


template<typename SrcType>
void serialize<lowertri,rect>::invoke(SrcType& src){

  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  // But now, I am going to have this call the serialize from UT to square, because thats what this will actually be doing
  // I tried a simple static_cast, but it didn't work, so now I will just copy code. Ugh! Fix later.
  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();

  T* srcVectorData = src.scratch();
  T* destVectorData = src.pad();

  U counter{srcNumRows};
  U srcOffset{0};
  U destOffset{0};
  for (U i=0; i<srcNumColumns; i++){
    fillZerosContig(&destVectorData[destOffset], i);
    memcpy(&destVectorData[destOffset+i], &srcVectorData[srcOffset], counter*sizeof(T));
    srcOffset += counter;
    destOffset += srcNumRows;
    counter--;
  }
}

template<typename SrcType, typename DestType>
void serialize<lowertri,rect>::invoke(SrcType& src, DestType& dest){

  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  // But now, I am going to have this call the serialize from UT to square, because thats what this will actually be doing
  // I tried a simple static_cast, but it didn't work, so now I will just copy code. Ugh! Fix later.
  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();

  T* srcVectorData = src.data();
  T* destVectorData = dest.data();

  U counter{srcNumRows};
  U srcOffset{0};
  U destOffset{0};
  for (U i=0; i<srcNumColumns; i++){
    fillZerosContig(&destVectorData[destOffset], i);
    memcpy(&destVectorData[destOffset+i], &srcVectorData[srcOffset], counter*sizeof(T));
    srcOffset += counter;
    destOffset += srcNumRows;
    counter--;
  }
  return;
}

template<typename BigType, typename SmallType>
void serialize<lowertri,rect>::invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                        typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){

  // Only written as one way to quiet compiler errors when adding rectangle matrix compatibility with MM3D
  // But now, I am going to have this call the serialize from UT to square, because thats what this will actually be doing
  // I tried a simple static_cast, but it didn't work, so now I will just copy code. Ugh! Fix later.
  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;

  U bigNumRows = big.num_rows_local();
  U bigNumColumns = big.num_columns_local();

  T* bigVectorData = big.data();
  T* smallVectorData = small.data();

  U smallMatOffset = 0;
  U smallMatCounter = rangeY;
  U bigMatOffset = ((bigNumColumns*(bigNumColumns+1))>>1);
  U helper = bigNumColumns - cutDimensionXstart;
  bigMatOffset -= ((helper*(helper+1))>>1);
  bigMatOffset += (cutDimensionYstart-cutDimensionXstart);
  U bigMatCounter = bigNumRows-cutDimensionXstart;
  U srcOffset = (dir ? smallMatOffset : bigMatOffset);
  U destOffset = (dir ? bigMatOffset : smallMatOffset);
  T* destVectorData = dir ? bigVectorData : smallVectorData;
  T* srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeX; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*rangeY);
    destOffset += (dir ? (bigMatCounter-1) : smallMatCounter);
    srcOffset += (dir ? smallMatCounter : (bigMatCounter-1));
    bigMatCounter--;
  }
  return;
}


template<typename SrcType, typename DestType>
void serialize<lowertri,lowertri>::invoke(SrcType& src, DestType& dest){

  using T = typename SrcType::ScalarType;
  using U = typename SrcType::DimensionType;
  U srcNumRows = src.num_rows_local();
  U srcNumColumns = src.num_columns_local();
  U srcNumElems = srcNumRows*srcNumColumns;

  T* srcVectorData = src.data();
  T* destVectorData = dest.data();

  // direction doesn't matter here since no indexing here
  memcpy(&destVectorData[0], &srcVectorData[0], sizeof(T)*srcNumElems);
}

template<typename BigType, typename SmallType>
void serialize<lowertri,lowertri>::invoke(BigType& big, SmallType& small, typename BigType::DimensionType cutDimensionXstart, typename BigType::DimensionType cutDimensionXend,
                                                            typename BigType::DimensionType cutDimensionYstart, typename BigType::DimensionType cutDimensionYend, bool dir){

  using T = typename BigType::ScalarType;
  using U = typename BigType::DimensionType;
  U rangeX = cutDimensionXend-cutDimensionXstart;
  U rangeY = cutDimensionYend-cutDimensionYstart;
  assert(rangeX == rangeY);

  U bigNumRows = big.num_rows_local();
  U bigNumColumns = big.num_columns_local();

  T* bigVectorData = big.data();
  T* smallVectorData = small.data();

  U smallMatOffset = 0;
  U smallMatCounter = rangeY;
  U bigMatOffset = ((bigNumColumns*(bigNumColumns+1))>>1);
  U helper = bigNumColumns - cutDimensionXstart;
  bigMatOffset -= ((helper*(helper+1))>>1);
  bigMatOffset += (cutDimensionYstart-cutDimensionXstart);
  U bigMatCounter = bigNumRows-cutDimensionXstart;
  U srcOffset = (dir ? smallMatOffset : bigMatOffset);
  U destOffset = (dir ? bigMatOffset : smallMatOffset);
  T* destVectorData = dir ? bigVectorData : smallVectorData;
  T* srcVectorData = dir ? smallVectorData : bigVectorData;
  for (U i=0; i<rangeY; i++){
    memcpy(&destVectorData[destOffset], &srcVectorData[srcOffset], sizeof(T)*smallMatCounter);
    destOffset += (dir ? bigMatCounter : smallMatCounter);
    srcOffset += (dir ? smallMatCounter : bigMatCounter);
    bigMatCounter--;
    smallMatCounter--;
  }
  return;
}
