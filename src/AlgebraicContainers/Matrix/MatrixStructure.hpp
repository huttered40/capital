/* Author: Edward Hutter */


template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureSquare<T,U,Distributer>::Allocate()
{
  // Nothing yet
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureSquare<T,U,Distributer>::Construct()
{
  // Nothing yet
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureSquare<T,U,Distributer>::Assemble(std::vector<T>& data, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY)
{
  // dimensionX must be equal to dimensionY, but I can't check this at compile time.
  assert(dimensionX == dimensionY);

  matrix.resize(dimensionX);
  matrixNumElems = dimensionX * dimensionX;
  data.resize(matrixNumElems);
  
  MatrixStructureSquare<T,U,Distributer>::AssembleMatrix(data, matrix, dimensionX, dimensionY);
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureSquare<T,U,Distributer>::AssembleMatrix(std::vector<T>& data, std::vector<T*>& matrix, U dimensionX, U dimensionY)
{
  U offset{0};
  for (auto& ptr : matrix)
  {
    ptr = &data[offset];
    offset += dimensionX;
  }
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureSquare<T,U,Distributer>::Deallocate()
{
  // Nothing yet
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureSquare<T,U,Distributer>::Destroy()
{
  // Nothing yet
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureSquare<T,U,Distributer>::Dissamble(std::vector<T*>& matrix)
{
  if ((matrix.size() > 0) && (matrix[0] != nullptr))
  {
    delete[] matrix[0];
  }
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureSquare<T,U,Distributer>::Copy(std::vector<T>& data, std::vector<T*>& matrix, const std::vector<T>& source, U dimensionX, U dimensionY)
{
  U numElems = 0;		// Just choose one dimension.
  Assemble(data, matrix, numElems, dimensionX, dimensionX);	// Just choose one dimension.
  std::memcpy(&data[0], &source[0], numElems*sizeof(T));
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureSquare<T,U,Distributer>::DistributeRandom(std::vector<T*>& matrix, U localDimensionX, U localDimensionY, U globalDimensionX, U globalDimensionY, U localPgridDimX, U localPgridDimY, U globalPgridDimX, U globalPgridDimDimY)
{
  Distributer<T,U,0>::DistributeRandom(matrix, localDimensionX, localDimensionY, globalDimensionX, globalDimensionY, localPgridDimX, localPgridDimY, globalPgridDimX, globalPgridDimDimY);
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureSquare<T,U,Distributer>::DistributeSymmetric(std::vector<T*>& matrix, U localDimensionX, U localDimensionY, U globalDimensionX, U globalDimensionY, U localPgridDimX, U localPgridDimY, U globalPgridDimX, U globalPgridDimDimY, bool diagonallyDominant)
{
  Distributer<T,U,0>::DistributeSymmetric(matrix, localDimensionX, localDimensionY, globalDimensionX, globalDimensionY, localPgridDimX, localPgridDimY, globalPgridDimX, globalPgridDimDimY, diagonallyDominant);
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureSquare<T,U,Distributer>::Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY)
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



template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureRectangle<T,U,Distributer>::Allocate()
{
  // Nothing yet
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureRectangle<T,U,Distributer>::Construct()
{
  // Nothing yet
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureRectangle<T,U,Distributer>::Assemble(std::vector<T>& data, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY)
{
  matrix.resize(dimensionY);
  matrixNumElems = dimensionX * dimensionY;
  data.resize(matrixNumElems);
  
  MatrixStructureRectangle<T,U,Distributer>::AssembleMatrix(data, matrix, dimensionX, dimensionY);
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureRectangle<T,U,Distributer>::AssembleMatrix(std::vector<T>& data, std::vector<T*>& matrix, U dimensionX, U dimensionY)
{
  U offset{0};
  for (auto& ptr : matrix)
  {
    ptr = &data[offset];
    offset += dimensionX;
  }
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureRectangle<T,U,Distributer>::Deallocate()
{
  // Nothing yet
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureRectangle<T,U,Distributer>::Destroy()
{
  // Nothing yet
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureRectangle<T,U,Distributer>::Dissamble(std::vector<T*>& matrix)
{
  if ((matrix.size() > 0) && (matrix[0] != nullptr))
  {
    delete[] matrix[0];
  }
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureRectangle<T,U,Distributer>::Copy(std::vector<T>& data, std::vector<T*>& matrix, const std::vector<T>& source, U dimensionX, U dimensionY)
{
  int numElems = 0;
  Assemble(data, matrix, numElems, dimensionX, dimensionY);
  std::memcpy(&data[0], &source[0], numElems*sizeof(T));
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureRectangle<T,U,Distributer>::DistributeRandom(std::vector<T*>& matrix, U localDimensionX, U localDimensionY, U globalDimensionX, U globalDimensionY, U localPgridDimX, U localPgridDimY, U globalPgridDimX, U globalPgridDimDimY)
{
  Distributer<T,U,1>::DistributeRandom(matrix, localDimensionX, localDimensionY, globalDimensionX, globalDimensionY, localPgridDimX, localPgridDimY, globalPgridDimX, globalPgridDimDimY);
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureRectangle<T,U,Distributer>::Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY)
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


template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureUpperTriangular<T,U,Distributer>::Allocate()
{
  // Nothing yet
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureUpperTriangular<T,U,Distributer>::Construct()
{
  // Nothing yet
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureUpperTriangular<T,U,Distributer>::Assemble(std::vector<T>& data, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY)
{
  // dimensionY must be equal to dimensionX
  assert(dimensionX == dimensionY);

  matrix.resize(dimensionY);
  matrixNumElems = ((dimensionY*(dimensionY+1))>>1);
  data.resize(matrixNumElems);

  MatrixStructureUpperTriangular<T,U,Distributer>::AssembleMatrix(data, matrix, dimensionX, dimensionY);
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureUpperTriangular<T,U,Distributer>::AssembleMatrix(std::vector<T>& data, std::vector<T*>& matrix, U dimensionX, U dimensionY)
{
  U offset{0};
  U counter{dimensionX};
  for (auto& ptr : matrix)
  {
    ptr = &data[offset];
    offset += counter;				// Hopefully this doesn't have 64-bit overflow problems :(
    counter--;
  }
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureUpperTriangular<T,U,Distributer>::Deallocate()
{
  // Nothing yet
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureUpperTriangular<T,U,Distributer>::Destroy()
{
  // Nothing yet
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureUpperTriangular<T,U,Distributer>::Dissamble(std::vector<T*>& matrix)
{
  if ((matrix.size() > 0) && (matrix[0] != nullptr))
  {
    delete[] matrix[0];
  }
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureUpperTriangular<T,U,Distributer>::Copy(std::vector<T>& data, std::vector<T*>& matrix, const std::vector<T>& source, U dimensionX, U dimensionY)
{
  U numElems = 0;
  Assemble(data, matrix, numElems, dimensionX, dimensionY);
  std::memcpy(&data[0], &source[0], numElems*sizeof(T));
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureUpperTriangular<T,U,Distributer>::DistributeRandom(std::vector<T*>& matrix, U localDimensionX, U localDimensionY, U globalDimensionX, U globalDimensionY, U localPgridDimX, U localPgridDimY, U globalPgridDimX, U globalPgridDimY)
{
  Distributer<T,U,2>::DistributeRandom(matrix, localDimensionX, localDimensionY, globalDimensionX, globalDimensionY, localPgridDimX, localPgridDimY, globalPgridDimX, globalPgridDimY);
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureUpperTriangular<T,U,Distributer>::Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY)
{
  U counter{dimensionX};
  for (const auto& rows : matrix)
  {
    U iter{dimensionX-counter};
    for (U i=0; i<iter; i++)
    {
      std::cout << "  ";
    }
    for (U i=0; i<counter; i++)
    {
      std::cout << " " << rows[i];
    }
    counter--;
    std::cout << std::endl;
  }
}


template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureLowerTriangular<T,U,Distributer>::Allocate()
{
  // Nothing yet
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureLowerTriangular<T,U,Distributer>::Construct()
{
  // Nothing yet
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureLowerTriangular<T,U,Distributer>::Assemble(std::vector<T>& data, std::vector<T*>& matrix, U& matrixNumElems, U dimensionX, U dimensionY)
{
  // dimensionY must be equal to dimensionX
  assert(dimensionX == dimensionY);

  matrix.resize(dimensionY);
  matrixNumElems = ((dimensionY*(dimensionY+1))>>1);
  data.resize(matrixNumElems);

  MatrixStructureLowerTriangular<T,U,Distributer>::AssembleMatrix(data, matrix, dimensionX, dimensionY);
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureLowerTriangular<T,U,Distributer>::AssembleMatrix(std::vector<T>& data, std::vector<T*>& matrix, U dimensionX, U dimensionY)
{
  U offset{0};
  U counter{1};
  for (auto& ptr : matrix)
  {
    ptr = &data[offset];
    offset += counter;				// Hopefully this doesn't have 64-bit overflow problems :(
    counter++;
  }
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureLowerTriangular<T,U,Distributer>::Deallocate()
{
  // Nothing yet
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureLowerTriangular<T,U,Distributer>::Destroy()
{
  // Nothing yet
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureLowerTriangular<T,U,Distributer>::Dissamble(std::vector<T*>& matrix)
{
  if ((matrix.size() > 0) && (matrix[0] != nullptr))
  {
    delete[] matrix[0];
  }
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureLowerTriangular<T,U,Distributer>::Copy(std::vector<T>& data, std::vector<T*>& matrix, const std::vector<T>& source, U dimensionX, U dimensionY)
{
  U numElems = 0;
  Assemble(data, matrix, numElems, dimensionX, dimensionY);
  std::memcpy(&data[0], &source[0], numElems*sizeof(T));
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureLowerTriangular<T,U,Distributer>::DistributeRandom(std::vector<T*>& matrix, U localDimensionX, U localDimensionY, U globalDimensionX, U globalDimensionY, U localPgridDimX, U localPgridDimY, U globalPgridDimX, U globalPgridDimDimY)
{
  Distributer<T,U,3>::DistributeRandom(matrix, localDimensionX, localDimensionY, globalDimensionX, globalDimensionY, localPgridDimX, localPgridDimY, globalPgridDimX, globalPgridDimDimY);
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureLowerTriangular<T,U,Distributer>::Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY)
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
