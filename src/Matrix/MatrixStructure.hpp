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
  //assert(dimensionX == dimensionY);

  matrix.resize(dimensionX);
  matrixNumElems = dimensionX * dimensionY;
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
    offset += dimensionY;
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
  Assemble(data, matrix, numElems, dimensionX, dimensionY);	// Just choose one dimension.
  std::memcpy(&data[0], &source[0], numElems*sizeof(T));
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureSquare<T,U,Distributer>::DistributeRandom(std::vector<T*>& matrix, U localDimensionX, U localDimensionY, U globalDimensionX, U globalDimensionY, U localPgridDimX, U localPgridDimY, U globalPgridDimX, U globalPgridDimDimY, U key)
{
  Distributer<T,U,0>::DistributeRandom(matrix, localDimensionX, localDimensionY, globalDimensionX, globalDimensionY, localPgridDimX, localPgridDimY, globalPgridDimX, globalPgridDimDimY, key);
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureSquare<T,U,Distributer>::DistributeSymmetric(std::vector<T*>& matrix, U localDimensionX, U localDimensionY, U globalDimensionX, U globalDimensionY, U localPgridDimX, U localPgridDimY, U globalPgridDimX, U globalPgridDimDimY, U key, bool diagonallyDominant)
{
  Distributer<T,U,0>::DistributeSymmetric(matrix, localDimensionX, localDimensionY, globalDimensionX, globalDimensionY, localPgridDimX, localPgridDimY, globalPgridDimX, globalPgridDimDimY, key, diagonallyDominant);
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureSquare<T,U,Distributer>::Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY)
{
  for (U i=0; i<dimensionY; i++)
  {
    for (U j=0; j<dimensionX; j++)
    {
      std::cout << " " << matrix[j][i];
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
  matrix.resize(dimensionX);
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
    offset += dimensionY;
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
void MatrixStructureRectangle<T,U,Distributer>::DistributeRandom(std::vector<T*>& matrix, U localDimensionX, U localDimensionY, U globalDimensionX, U globalDimensionY, U localPgridDimX, U localPgridDimY, U globalPgridDimX, U globalPgridDimDimY, U key)
{
  Distributer<T,U,1>::DistributeRandom(matrix, localDimensionX, localDimensionY, globalDimensionX, globalDimensionY, localPgridDimX, localPgridDimY, globalPgridDimX, globalPgridDimDimY, key);
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureRectangle<T,U,Distributer>::Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY)
{
  for (U i=0; i<dimensionY; i++)
  {
    for (U j=0; j<dimensionX; j++)
    {
      std::cout << " " << matrix[j][i];
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
  matrixNumElems = ((dimensionY*(dimensionY+1))>>1);		// dimensionX == dimensionY
  data.resize(matrixNumElems);

  MatrixStructureUpperTriangular<T,U,Distributer>::AssembleMatrix(data, matrix, dimensionX, dimensionY);
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureUpperTriangular<T,U,Distributer>::AssembleMatrix(std::vector<T>& data, std::vector<T*>& matrix, U dimensionX, U dimensionY)
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
void MatrixStructureUpperTriangular<T,U,Distributer>::DistributeRandom(std::vector<T*>& matrix, U localDimensionX, U localDimensionY, U globalDimensionX, U globalDimensionY, U localPgridDimX, U localPgridDimY, U globalPgridDimX, U globalPgridDimY, U key)
{
  Distributer<T,U,2>::DistributeRandom(matrix, localDimensionX, localDimensionY, globalDimensionX, globalDimensionY, localPgridDimX, localPgridDimY, globalPgridDimX, globalPgridDimY, key);
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureUpperTriangular<T,U,Distributer>::Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY)
{
  U startIter = 0;
  for (U i=0; i<dimensionY; i++)
  {
    // Print spaces to represent the lower triangular zeros
    for (U j=0; j<i; j++)
    {
      std::cout << "    ";
    }

    for (U j=startIter; j<dimensionX; j++)
    {
      std::cout << " " << matrix[j][i];
    }
    startIter++;
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

  matrix.resize(dimensionX);
  matrixNumElems = ((dimensionY*(dimensionY+1))>>1);
  data.resize(matrixNumElems);

  MatrixStructureLowerTriangular<T,U,Distributer>::AssembleMatrix(data, matrix, dimensionX, dimensionY);
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureLowerTriangular<T,U,Distributer>::AssembleMatrix(std::vector<T>& data, std::vector<T*>& matrix, U dimensionX, U dimensionY)
{
  U offset{0};
  U counter{dimensionY};
  for (auto& ptr : matrix)
  {
    ptr = &data[offset];
    offset += counter;				// Hopefully this doesn't have 64-bit overflow problems :(
    counter--;
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
void MatrixStructureLowerTriangular<T,U,Distributer>::DistributeRandom(std::vector<T*>& matrix, U localDimensionX, U localDimensionY, U globalDimensionX, U globalDimensionY, U localPgridDimX, U localPgridDimY, U globalPgridDimX, U globalPgridDimDimY, U key)
{
  Distributer<T,U,3>::DistributeRandom(matrix, localDimensionX, localDimensionY, globalDimensionX, globalDimensionY, localPgridDimX, localPgridDimY, globalPgridDimX, globalPgridDimDimY, key);
}

template<typename T, typename U, template<typename,typename,int> class Distributer>
void MatrixStructureLowerTriangular<T,U,Distributer>::Print(const std::vector<T*>& matrix, U dimensionX, U dimensionY)
{
  for (U i=0; i<dimensionY; i++)
  {
    for (U j=0; j<=i; j++)
    {
      std::cout << matrix[j][i-j] << " ";
    }
    std::cout << "\n";
  }
}
