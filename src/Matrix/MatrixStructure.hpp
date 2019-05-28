/* Author: Edward Hutter */

template<typename T, typename U>
void Square::_Assemble(std::vector<T>& data, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY){
  // dimensionX must be equal to dimensionY, but I can't check this at compile time.
  //assert(dimensionX == dimensionY);

  matrix.resize(dimensionX);
  matrixNumElems = dimensionX * dimensionY;
  data.resize(matrixNumElems);

  _AssembleMatrix(data, matrix, dimensionX, dimensionY);
}

template<typename T, typename U>
void Square::_AssembleMatrix(std::vector<T>& data, std::vector<T*>& matrix, U dimensionX, U dimensionY){
  U offset{0};
  for (auto& ptr : matrix){
    ptr = &data[offset];
    offset += dimensionY;
  }
}

template<typename T>
void Square::_Dissamble(std::vector<T*>& matrix){
  if ((matrix.size() > 0) && (matrix[0] != nullptr)){
    delete[] matrix[0];
  }
}

template<typename T, typename U>
void Square::_Copy(std::vector<T>& data, std::vector<T*>& matrix, const std::vector<T>& source, U dimensionX, U dimensionY){
  U numElems = 0;		// Just choose one dimension.
  _Assemble(data, matrix, numElems, dimensionX, dimensionY);	// Just choose one dimension.
  std::memcpy(&data[0], &source[0], numElems*sizeof(T));
}

template<typename T, typename U>
void Square::_Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY){
  for (U i=0; i<dimensionY; i++){
    for (U j=0; j<dimensionX; j++){
      std::cout << " " << matrix[j][i];
    }
    std::cout << std::endl;
  }
}


template<typename T, typename U>
void Rectangular::_Assemble(std::vector<T>& data, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY){
  matrix.resize(dimensionX);
  matrixNumElems = dimensionX * dimensionY;
  data.resize(matrixNumElems);
  
  _AssembleMatrix(data, matrix, dimensionX, dimensionY);
}

template<typename T, typename U>
void Rectangular::_AssembleMatrix(std::vector<T>& data, std::vector<T*>& matrix, U dimensionX, U dimensionY){
  U offset{0};
  for (auto& ptr : matrix){
    ptr = &data[offset];
    offset += dimensionY;
  }
}

template<typename T>
void Rectangular::_Dissamble(std::vector<T*>& matrix){
  if ((matrix.size() > 0) && (matrix[0] != nullptr)){
    delete[] matrix[0];
  }
}

template<typename T, typename U>
void Rectangular::_Copy(std::vector<T>& data, std::vector<T*>& matrix, const std::vector<T>& source, U dimensionX, U dimensionY){
  U numElems = 0;
  _Assemble(data, matrix, numElems, dimensionX, dimensionY);
  std::memcpy(&data[0], &source[0], numElems*sizeof(T));
}

template<typename T, typename U>
void Rectangular::_Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY){
  for (U i=0; i<dimensionY; i++){
    for (U j=0; j<dimensionX; j++){
      std::cout << " " << matrix[j][i];
    }
    std::cout << std::endl;
  }
}


template<typename T, typename U>
void UpperTriangular::_Assemble(std::vector<T>& data, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY){
  // dimensionY must be equal to dimensionX
  assert(dimensionX == dimensionY);

  matrix.resize(dimensionY);
  matrixNumElems = ((dimensionY*(dimensionY+1))>>1);		// dimensionX == dimensionY
  data.resize(matrixNumElems);

  _AssembleMatrix(data, matrix, dimensionX, dimensionY);
}

template<typename T, typename U>
void UpperTriangular::_AssembleMatrix(std::vector<T>& data, std::vector<T*>& matrix, U dimensionX, U dimensionY){
  U offset{0};
  U counter{1};
  for (auto& ptr : matrix){
    ptr = &data[offset];
    offset += counter;				// Hopefully this doesn't have 64-bit overflow problems :(
    counter++;
  }
}


template<typename T>
void UpperTriangular::_Dissamble(std::vector<T*>& matrix){
  if ((matrix.size() > 0) && (matrix[0] != nullptr)){
    delete[] matrix[0];
  }
}

template<typename T, typename U>
void UpperTriangular::_Copy(std::vector<T>& data, std::vector<T*>& matrix, const std::vector<T>& source, U dimensionX, U dimensionY){
  U numElems = 0;
  _Assemble(data, matrix, numElems, dimensionX, dimensionY);
  std::memcpy(&data[0], &source[0], numElems*sizeof(T));
}

template<typename T, typename U>
void UpperTriangular::_Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY){
  U startIter = 0;
  for (U i=0; i<dimensionY; i++){
    // Print spaces to represent the lower triangular zeros
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
void LowerTriangular::_Assemble(std::vector<T>& data, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY){
  // dimensionY must be equal to dimensionX
  assert(dimensionX == dimensionY);

  matrix.resize(dimensionX);
  matrixNumElems = ((dimensionY*(dimensionY+1))>>1);
  data.resize(matrixNumElems);

  _AssembleMatrix(data, matrix, dimensionX, dimensionY);
}

template<typename T, typename U>
void LowerTriangular::_AssembleMatrix(std::vector<T>& data, std::vector<T*>& matrix, U dimensionX, U dimensionY){
  U offset{0};
  U counter{dimensionY};
  for (auto& ptr : matrix){
    ptr = &data[offset];
    offset += counter;				// Hopefully this doesn't have 64-bit overflow problems :(
    counter--;
  }
}

template<typename T>
void LowerTriangular::_Dissamble(std::vector<T*>& matrix){
  if ((matrix.size() > 0) && (matrix[0] != nullptr)){
    delete[] matrix[0];
  }
}

template<typename T, typename U>
void LowerTriangular::_Copy(std::vector<T>& data, std::vector<T*>& matrix, const std::vector<T>& source, U dimensionX, U dimensionY){
  U numElems = 0;
  _Assemble(data, matrix, numElems, dimensionX, dimensionY);
  std::memcpy(&data[0], &source[0], numElems*sizeof(T));
}

template<typename T, typename U>
void LowerTriangular::_Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY){
  for (U i=0; i<dimensionY; i++){
    for (U j=0; j<=i; j++){
      std::cout << matrix[j][i-j] << " ";
    }
    std::cout << "\n";
  }
}
