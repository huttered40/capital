/* Author: Edward Hutter */

template<typename T, typename U>
void square::_assemble(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY){
  // dimensionX must be equal to dimensionY, but I can't check this at compile time.
  //assert(dimensionX == dimensionY);

  matrix.resize(dimensionX);
  matrixNumElems = dimensionX * dimensionY;
  data = new T[matrixNumElems];
  _assemble_matrix(data, scratch, pad, matrix, dimensionX, dimensionY);
}

template<typename T, typename U>
void square::_assemble_matrix(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, U dimensionX, U dimensionY){
  U matrixNumElems = dimensionX * dimensionY;
  scratch = new T[matrixNumElems];
  pad = nullptr;
  U offset{0};
  for (auto& ptr : matrix){
    ptr = &data[offset];
    offset += dimensionY;
  }
}

template<typename T, typename U>
void square::_copy(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, T* const & source, U dimensionX, U dimensionY){
  U numElems = 0;		// Just choose one dimension.
  _assemble(data, scratch, pad, matrix, numElems, dimensionX, dimensionY);	// Just choose one dimension.
  std::memcpy(&data[0], &source[0], numElems*sizeof(T));
}

template<typename T, typename U>
void square::_print(const std::vector<T*>& matrix, U dimensionX, U dimensionY){
  for (U i=0; i<dimensionY; i++){
    for (U j=0; j<dimensionX; j++){
      std::cout << " " << matrix[j][i];
    }
    std::cout << std::endl;
  }
}


template<typename T, typename U>
void rect::_assemble(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY){
  matrix.resize(dimensionX);
  matrixNumElems = dimensionX * dimensionY;
  data = new T[matrixNumElems];
  _assemble_matrix(data, scratch, pad, matrix, dimensionX, dimensionY);
}

template<typename T, typename U>
void rect::_assemble_matrix(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, U dimensionX, U dimensionY){
  U matrixNumElems = dimensionX * dimensionY;
  scratch = new T[matrixNumElems];
  pad = nullptr;
  U offset{0};
  for (auto& ptr : matrix){
    ptr = &data[offset];
    offset += dimensionY;
  }
}

template<typename T, typename U>
void rect::_copy(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, T* const & source, U dimensionX, U dimensionY){
  U numElems = 0;
  _assemble(data, scratch, pad, matrix, numElems, dimensionX, dimensionY);
  std::memcpy(&data[0], &source[0], numElems*sizeof(T));
}

template<typename T, typename U>
void rect::_print(const std::vector<T*>& matrix, U dimensionX, U dimensionY){
  for (U i=0; i<dimensionY; i++){
    for (U j=0; j<dimensionX; j++){
      std::cout << " " << matrix[j][i];
    }
    std::cout << std::endl;
  }
}


template<typename T, typename U>
void uppertri::_assemble(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY){
  // dimensionY must be equal to dimensionX
  assert(dimensionX == dimensionY);

  matrix.resize(dimensionY);
  matrixNumElems = ((dimensionY*(dimensionY+1))>>1);		// dimensionX == dimensionY
  data = new T[matrixNumElems];
  _assemble_matrix(data, scratch, pad, matrix, dimensionX, dimensionY);
}

template<typename T, typename U>
void uppertri::_assemble_matrix(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, U dimensionX, U dimensionY){
  U nonPackedNumElems = dimensionX*dimensionY;
  U matrixNumElems = ((dimensionY*(dimensionY+1))>>1);		// dimensionX == dimensionY
  scratch = new T[matrixNumElems];
  pad = new T[nonPackedNumElems];	// we give full non-packed size here to account for need for summa to use nonpacked layout
  U offset{0};
  U counter{1};
  for (auto& ptr : matrix){
    ptr = &data[offset];
    offset += counter;				// Hopefully this doesn't have 64-bit overflow problems :(
    counter++;
  }
}

template<typename T, typename U>
void uppertri::_copy(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, T* const & source, U dimensionX, U dimensionY){
  U numElems = 0;
  _assemble(data, scratch, pad, matrix, numElems, dimensionX, dimensionY);
  std::memcpy(&data[0], &source[0], numElems*sizeof(T));
}

template<typename T, typename U>
void uppertri::_print(const std::vector<T*>& matrix, U dimensionX, U dimensionY){
  U startIter = 0;
  for (U i=0; i<dimensionY; i++){
    // print spaces to represent the lower triangular zeros
    for (U j=0; j<i; j++){
      std::cout << "    ";
    }

    for (U j=startIter; j<dimensionX; j++){
      std::cout << " " << matrix[j][i];
    }
    startIter++;
    std::cout << std::endl;
  }
}


template<typename T, typename U>
void lowertri::_assemble(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY){
  // dimensionY must be equal to dimensionX
  assert(dimensionX == dimensionY);

  matrix.resize(dimensionX);
  matrixNumElems = ((dimensionY*(dimensionY+1))>>1);
  data = new T[matrixNumElems];
  _assemble_matrix(data, scratch, pad, matrix, dimensionX, dimensionY);
}

template<typename T, typename U>
void lowertri::_assemble_matrix(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, U dimensionX, U dimensionY){
  U nonPackedNumElems = dimensionX*dimensionY;
  U matrixNumElems = ((dimensionY*(dimensionY+1))>>1);		// dimensionX == dimensionY
  scratch = new T[matrixNumElems];
  pad = new T[nonPackedNumElems];	// we give full non-packed size here to account for need for summa to use nonpacked layout
  U offset{0};
  U counter{dimensionY};
  for (auto& ptr : matrix){
    ptr = &data[offset];
    offset += counter;				// Hopefully this doesn't have 64-bit overflow problems :(
    counter--;
  }
}

template<typename T, typename U>
void lowertri::_copy(T*& data, T*& scratch, T*& pad, std::vector<T*>& matrix, T* const & source, U dimensionX, U dimensionY){
  U numElems = 0;
  _assemble(data, scratch, pad, matrix, numElems, dimensionX, dimensionY);
  std::memcpy(&data[0], &source[0], numElems*sizeof(T));
}

template<typename T, typename U>
void lowertri::_print(const std::vector<T*>& matrix, U dimensionX, U dimensionY){
  for (U i=0; i<dimensionY; i++){
    for (U j=0; j<=i; j++){
      std::cout << matrix[j][i-j] << " ";
    }
    std::cout << "\n";
  }
}
