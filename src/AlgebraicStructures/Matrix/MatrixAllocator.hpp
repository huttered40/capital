/* Author: Edward Hutter */


template<typename T, typename U>
void MatrixAllocatorSquare<T,U>::Allocate()
{
  // Nothing yet
}

template<typename T, typename U>
void MatrixAllocatorSquare<T,U>::Construct()
{
  // Nothing yet
}

template<typename T, typename U>
void MatrixAllocatorSquare<T,U>::Assemble(std::vector<T*>& matrix, U dimensionX, U dimensionY)
{
  // dimensionX must be equal to dimensionY, but I can't check this at compile time.
  assert(dimensionX == dimensionY);

  matrix.resize(dimensionX);
  matrix[0] = new T[dimensionX * dimensionX];
  
  U offset{0};
  for (auto& ptr : matrix)
  {
    ptr = &matrix[0][offset];
    offset += dimensionX;
  }
}

template<typename T, typename U>
void MatrixAllocatorSquare<T,U>::Deallocate()
{
  // Nothing yet
}

template<typename T, typename U>
void MatrixAllocatorSquare<T,U>::Destroy()
{
  // Nothing yet
}

template<typename T, typename U>
void MatrixAllocatorSquare<T,U>::Dissamble(std::vector<T*>& matrix)
{
  if ((matrix.size() > 0) && (matrix[0] != nullptr))
  {
    delete[] matrix[0];
  }
}

template<typename T, typename U>
void MatrixAllocatorSquare<T,U>::Copy(std::vector<T*>& matrix, const std::vector<T*>& source, U dimensionX, U dimensionY)
{
  Assemble(matrix, dimensionX, dimensionX);	// Just choose one dimension.
  U numElems = dimensionX*dimensionX;		// Just choose one dimension.
  std::memcpy(matrix[0], source[0], numElems);
}

template<typename T, typename U>
void MatrixAllocatorSquare<T,U>::Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY)
{
  for (const auto& rows : matrix)
  {
    for (U i=0; i<dimensionX; i++)
    {
      std::cout << " " << rows[i];
    }
    std::cout << std::endl;
  }
}


template<typename T, typename U>
void MatrixAllocatorRectangle<T,U>::Allocate()
{
  // Nothing yet
}

template<typename T, typename U>
void MatrixAllocatorRectangle<T,U>::Construct()
{
  // Nothing yet
}

template<typename T, typename U>
void MatrixAllocatorRectangle<T,U>::Assemble(std::vector<T*>& matrix, U dimensionX, U dimensionY)
{
  matrix.resize(dimensionX);
  matrix[0] = new T[dimensionX * dimensionY];
  
  U offset{0};
  for (auto& ptr : matrix)
  {
    ptr = &matrix[0][offset];
    offset += dimensionY;
  }
}

template<typename T, typename U>
void MatrixAllocatorRectangle<T,U>::Deallocate()
{
  // Nothing yet
}

template<typename T, typename U>
void MatrixAllocatorRectangle<T,U>::Destroy()
{
  // Nothing yet
}

template<typename T, typename U>
void MatrixAllocatorRectangle<T,U>::Dissamble(std::vector<T*>& matrix)
{
  if ((matrix.size() > 0) && (matrix[0] != nullptr))
  {
    delete[] matrix[0];
  }
}

template<typename T, typename U>
void MatrixAllocatorRectangle<T,U>::Copy(std::vector<T*>& matrix, const std::vector<T*>& source, U dimensionX, U dimensionY)
{
  Assemble(matrix, dimensionX, dimensionY);
  U numElems = dimensionX*dimensionY;
  std::memcpy(matrix[0], source[0], numElems);
}

template<typename T, typename U>
void MatrixAllocatorRectangle<T,U>::Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY)
{
  for (const auto& rows : matrix)
  {
    for (U i=0; i<dimensionY; i++)
    {
      std::cout << " " << rows[i];
    }
    std::cout << std::endl;
  }
}


template<typename T, typename U>
void MatrixAllocatorUpperTriangular<T,U>::Allocate()
{
  // Nothing yet
}

template<typename T, typename U>
void MatrixAllocatorUpperTriangular<T,U>::Construct()
{
  // Nothing yet
}

template<typename T, typename U>
void MatrixAllocatorUpperTriangular<T,U>::Assemble(std::vector<T*>& matrix, U dimensionX, U dimensionY)
{
  // dimensionY must be equal to dimensionX
  assert(dimensionX == dimensionY);

  matrix.resize(dimensionX);
  U numElems = ((dimensionY*(dimensionY+1))>>1);
  matrix[0] = new T[numElems];
  
  U offset{0};
  U counter{dimensionY};
  for (auto& ptr : matrix)
  {
    ptr = &matrix[0][offset];
    offset += counter;				// Hopefully this doesn't have 64-bit overflow problems :(
    counter--;
  }
}

template<typename T, typename U>
void MatrixAllocatorUpperTriangular<T,U>::Deallocate()
{
  // Nothing yet
}

template<typename T, typename U>
void MatrixAllocatorUpperTriangular<T,U>::Destroy()
{
  // Nothing yet
}

template<typename T, typename U>
void MatrixAllocatorUpperTriangular<T,U>::Dissamble(std::vector<T*>& matrix)
{
  if ((matrix.size() > 0) && (matrix[0] != nullptr))
  {
    delete[] matrix[0];
  }
}

template<typename T, typename U>
void MatrixAllocatorUpperTriangular<T,U>::Copy(std::vector<T*>& matrix, const std::vector<T*>& source, U dimensionX, U dimensionY)
{
  Assemble(matrix, dimensionX, dimensionY);
  U numElems = ((dimensionY*(dimensionY+1))>>1);
  std::memcpy(matrix[0], source[0], numElems);
}

template<typename T, typename U>
void MatrixAllocatorUpperTriangular<T,U>::Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY)
{
  U counter{dimensionY};
  for (const auto& rows : matrix)
  {
    for (U i=0; i<counter; i++)
    {
      std::cout << " " << rows[i];
    }
    counter--;
    std::cout << std::endl;
  }
}


template<typename T, typename U>
void MatrixAllocatorLowerTriangular<T,U>::Allocate()
{
  // Nothing yet
}

template<typename T, typename U>
void MatrixAllocatorLowerTriangular<T,U>::Construct()
{
  // Nothing yet
}

template<typename T, typename U>
void MatrixAllocatorLowerTriangular<T,U>::Assemble(std::vector<T*>& matrix, U dimensionX, U dimensionY)
{
  // dimensionY must be equal to dimensionX
  assert(dimensionX == dimensionY);

  matrix.resize(dimensionX);
  U numElems = ((dimensionX*(dimensionX+1))>>1);
  matrix[0] = new T[numElems];
  
  U offset{0};
  U counter{1};
  for (auto& ptr : matrix)
  {
    ptr = &matrix[0][offset];
    offset += counter;				// Hopefully this doesn't have 64-bit overflow problems :(
    counter++;
  }
}

template<typename T, typename U>
void MatrixAllocatorLowerTriangular<T,U>::Deallocate()
{
  // Nothing yet
}

template<typename T, typename U>
void MatrixAllocatorLowerTriangular<T,U>::Destroy()
{
  // Nothing yet
}

template<typename T, typename U>
void MatrixAllocatorLowerTriangular<T,U>::Dissamble(std::vector<T*>& matrix)
{
  if ((matrix.size() > 0) && (matrix[0] != nullptr))
  {
    delete[] matrix[0];
  }
}

template<typename T, typename U>
void MatrixAllocatorLowerTriangular<T,U>::Copy(std::vector<T*>& matrix, const std::vector<T*>& source, U dimensionX, U dimensionY)
{
  Assemble(matrix, dimensionX, dimensionY);
  U numElems = ((dimensionX*(dimensionX+1))>>1);
  std::memcpy(matrix[0], source[0], numElems);
}

template<typename T, typename U>
void MatrixAllocatorLowerTriangular<T,U>::Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY)
{
  U counter{1};
  for (const auto& rows : matrix)
  {
    for (U i=0; i<counter; i++)
    {
      std::cout << " " << rows[i];
    }
    counter++;
    std::cout << std::endl;
  }
}
