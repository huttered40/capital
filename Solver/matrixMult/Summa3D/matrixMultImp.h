/*
	Author: Edward Hutter
*/

/*
  Iterative Parallel Matrix Multiplication - Broadcasts and AllReduces along the layers

  These routines require that matrix data is duplicated across all layers of the 3D Processor Grid
  Many of these routines also have specific data access patterns based on the way the data is stored

  The current implementation is based on the Cholesky Matrix Multiplication, where the entire matrix is passed in by reference
    and we use parameters as cookie cutters to pick at the piece of the matrix that we want to multiply.

  Further development is up in the air
*/

//#include "matrixMult.h"

#define DEBUGGING_MATRIX_MULTIPLICATION


/*
	Simple constructor, requires a 3D grid to be set up before multiply is called
*/
template<typename T>
matrixMult<T>::matrixMult
			(
			MPI_Comm grid,		// Existing processor grid that will get split
			uint32_t dim		// dimension of the processor grid cube (only use case so far is CholeskyQR)
			)
{
  // I need to set up a grid for the Summ3D MM operation. Take from Cholesky.

  constructGridMM(grid, dim);
}

/*
	Constructor for an existing 3D grid (will be called by Cholesy)
*/
template <typename T>
matrixMult<T>::matrixMult
				(
				MPI_Comm rowCommunicator,
				MPI_Comm columnCommunicator,
				MPI_Comm layerCommunicator,
				MPI_Comm grid3DCommunicator,
				MPI_Comm depthCommunicator,
				uint32_t rowCommunicatorRank,
				uint32_t columnCommunicatorRank,
				uint32_t layerCommunicatorRank,
				uint32_t grid3DCommunicatorRank,
				uint32_t depthCommunicatorRank,
				uint32_t rowCommunicatorSize,
				uint32_t columnCommunicatorSize,
				uint32_t layerCommunicatorSize,
				uint32_t grid3DCommunicatorSize,
				uint32_t depthCommunicatorSize,
				std::vector<int> gridCoordinates
			 	)
{
  // set up the 4 member communicators for use in multiply operations
  this->rowComm = rowCommunicator;
  this->columnComm = columnCommunicator;
  this->layerComm = layerCommunicator;
  this->grid3DComm = grid3DCommunicator;
  this->depthComm = depthCommunicator;
  this->rowCommRank = rowCommunicatorRank;
  this->columnCommRank = columnCommunicatorRank;
  this->layerCommRank = layerCommunicatorRank;
  this->grid3DCommRank = grid3DCommunicatorRank;
  this->depthCommRank = depthCommunicatorRank;
  this->rowCommSize = rowCommunicatorSize;
  this->columnCommSize = columnCommunicatorSize;
  this->layerCommSize = layerCommunicatorSize;
  this->grid3DCommSize = grid3DCommunicatorSize;
  this->depthCommSize = depthCommunicatorSize;
  this->gridCoords = gridCoordinates;
  // Set up anything else?
}

template<typename T>
void matrixMult<T>::constructGridMM(MPI_Comm grid, uint32_t dim)
{
  this->processorGridDimSize = dim;
  
  this->gridDims.resize(3,this->processorGridDimSize);
  this->gridCoords.resize(this->nDims); 
  
  /*
    The 3D Cartesian Communicator is used to distribute the random data in a cyclic fashion
    The other communicators are created for specific communication patterns involved in the algorithm.
  */
  std::vector<int> boolVec(3,0);
  MPI_Cart_create(this->worldComm, this->nDims, &this->gridDims[0], &boolVec[0], false, &this->grid3D);
  MPI_Comm_rank(this->grid3D, &this->grid3DRank);
  MPI_Comm_size(this->grid3D, &this->grid3DSize);
  MPI_Cart_coords(this->grid3D, this->grid3DRank, this->nDims, &this->gridCoords[0]);

  /*
    Before creating row and column sub-communicators, grid3D must be split into 2D Layer communicators.
  */

  // 2D (xy) Layer Communicator (split by z coordinate)
  MPI_Comm_split(this->grid3D, this->gridCoords[2],this->grid3DRank,&this->layerComm);
  MPI_Comm_rank(this->layerComm,&this->layerCommRank);
  MPI_Comm_size(this->layerComm,&this->layerCommSize);
  // Row Communicator
  MPI_Comm_split(this->layerComm, this->gridCoords[0],this->gridCoords[1],&this->rowComm);
  MPI_Comm_rank(this->rowComm,&this->rowCommRank);
  MPI_Comm_size(this->rowComm,&this->rowCommSize);
  // column Communicator
  MPI_Comm_split(this->layerComm, this->gridCoords[1],this->gridCoords[0],&this->colComm);
  MPI_Comm_rank(this->colComm,&this->colCommRank);
  MPI_Comm_size(this->colComm,&this->colCommSize);
  // Depth Communicator
  MPI_Comm_split(this->grid3D,this->gridCoords[0]*this->processorGridDimSize+this->gridCoords[1],this->gridCoords[2],&this->depthComm);
  MPI_Comm_rank(this->depthComm,&this->depthCommRank);
  MPI_Comm_size(this->depthComm,&this->depthCommSize);

}

template<typename T>
void matrixMult<T>::multiply
				(
				std::vector<T> &matA,
				std::vector<T> &matB,
				std::vector<T> &matC,			// Should I pass these in as vectors or pointers?
         			uint32_t dimXstartA,
				uint32_t dimXendA,
				uint32_t dimYstartA,
				uint32_t dimYendA,
				uint32_t dimXstartB,
				uint32_t dimXendB,
				uint32_t dimYstartB,
				uint32_t dimYendB,
				uint32_t dimXstartC,
				uint32_t dimXendC,
				uint32_t dimYstartC,
				uint32_t dimYendC,
				uint32_t matrixWindowRow,
				uint32_t matrixWindowCol,
				uint32_t matrixSizeRow,
				uint32_t matrixSizeCol,
				uint32_t key,
				uint32_t matrixCutSize
			    	)
{
  // Make sure that the arrays have enough memory, so do a resize
  // do the std::moves

  this->matrixA = std::move(matA);
  this->matrixB = std::move(matB);
  //Special circumstances warrant this change. This is from a bug that i had where I was "moving" the same reference twice.
  if (key != 4)
  {
    this->matrixC = std::move(matC);
  }

  switch (key)
  {
    case 1:
    {
    /*
	Matrix Multiplication of a square matrix and a lower triangular matrix

	matrixA - square cut of a square matrix that is packed in a rectangular manner
	matrixB - full (no cut) triangular matrix that is packed in a triangular manner
	matrixC - square cut of a triangular matrix that is packed in a triangular manner
    */
      multiply1
		(
		dimXstartA,
		dimXendA,
		dimYstartA,
		dimYendA,
		dimXstartB,
		dimXendB,
		dimYstartB,			// this was dimYstartA before. I have no idea how this did not cause errors.
		dimYendB,
		dimXstartC,
		dimXendC,
		dimYstartC,
		dimYendC,
		matrixWindowRow,		// arbitrary choice between row and col
		matrixSizeRow,			// arbitrary choice between row and col
		matrixCutSize
		);
      break;
    }
    case 2:
    {
    /*
	Matrix Multiplication of a square matrix with a square matrix
	
	matrixA - square cut of a triangular matrix that is packed in a triangular manner
	matrixB - full (no cut) square matrix that is packed in a square manner
	matrixC - full (no cut) square matrix that is packed in a square manner
    */
      multiply2
		(
		dimXstartA,
		dimXendA,
		dimYstartA,
		dimYendA,
		dimXstartB,
		dimXendB,
		dimYstartB,			// this was dimYstartA before. I have no idea how this did not cause errors.
		dimYendB,
		dimXstartC,
		dimXendC,
		dimYstartC,
		dimYendC,
		matrixWindowRow,		// arbitrary choice between row and col
		matrixSizeRow			// arbitrary choice between row and col
		);
      break;
    }
    case 3:
    {
    /*
	Matrix Multiplication of a Square matrix and a lower triangular matrix
	
	matrixA - square cut of a triangular matrix that is packed in a triangular manner
	matrixB - triangular cut of a triangular matrix that is packed in a triangular manner
	matrixC - full (no cut) square matrix that is packed in a square manner
    */
      multiply3
		(
		dimXstartA,
		dimXendA,
		dimYstartA,
		dimYendA,
		dimXstartB,
		dimXendB,
		dimYstartB,			// this was dimYstartA before. I have no idea how this did not cause errors.
		dimYendB,
		dimXstartC,
		dimXendC,
		dimYstartC,
		dimYendC,
		matrixWindowRow,		// arbitrary choice between row and col
		matrixSizeRow			// arbitrary choice between row and col
		);
      break;
    }
    case 4:
    {
    /*
	Matrix Multiplication of a Lower triangular and a square matrix
	
	matrixA - triangular cut of a triangular matrix that is packed in a triangular manner
	matrixB - full (no cut) square matrix that is packed in a square manner
	matrixC - square cut of a triangular matrix that is packed in a triangular manner
    */
      multiply4
		(
		dimXstartA,
		dimXendA,
		dimYstartA,
		dimYendA,
		dimXstartB,
		dimXendB,
		dimYstartB,			// this was dimYstartA before. I have no idea how this did not cause errors.
		dimYendB,
		dimXstartC,
		dimXendC,
		dimYstartC,
		dimYendC,
		matrixWindowRow,		// arbitrary choice between row and col
		matrixSizeRow			// arbitrary choice between row and col
		);
      break;
    }
    case 5:
    {
      multiply5
		(
		matrixWindowRow,
		matrixWindowCol,
		matrixSizeRow,
		matrixSizeCol
		);
      break;
    }
  }

  // reverse the std::moves to fill back up the buffers that were passed in
  matA = std::move(this->matrixA);
  matB = std::move(this->matrixB);
  if (key != 4)
  {
    matC = std::move(this->matrixC);
  }
}

/*
	Matrix Multiplication of a square matrix and a lower triangular matrix

	matrixA - square cut of a square matrix that is packed in a rectangular manner
	matrixB - full (no cut) triangular matrix that is packed in a triangular manner
	matrixC - square cut of a triangular matrix that is packed in a triangular manner
*/
template<typename T>
void matrixMult<T>::multiply1
				(
         			uint32_t dimXstartA,
				uint32_t dimXendA,
				uint32_t dimYstartA,
				uint32_t dimYendA,
				uint32_t dimXstartB,
				uint32_t dimXendB,
				uint32_t dimYstartB,
				uint32_t dimYendB,
				uint32_t dimXstartC,
				uint32_t dimXendC,
				uint32_t dimYstartC,
				uint32_t dimYendC,
				uint32_t matrixWindow,
				uint32_t matrixSize,
				uint32_t matrixCutSize
				)
{
  std::vector<T> buffer1;  // use capacity() or reserve() methods here?
  std::vector<T> buffer2;

  if (this->rowCommRank == this->gridCoords[2])     // different matches on each of the P^(1/3) layers
  {
    // (square,lower) -> Note for further optimization.
    // -> This copy loop may not be needed. I might be able to purely send in &this->matrixLInverse[0] into broadcast
    uint64_t buffer1Size = matrixWindow;
    buffer1Size *= matrixWindow;
    buffer1.resize(buffer1Size,0.);
    uint64_t startOffset = matrixCutSize;
    startOffset *= matrixWindow;
    uint64_t index1 = 0;
    uint64_t index2 = startOffset;
    for (uint32_t i=0; i<matrixWindow; i++)
    {
      for (uint32_t j=0; j<matrixWindow; j++)
      {
        uint64_t temp = index2;
        temp += j;
        buffer1[index1++] = this->matrixA[temp];	//this->matrixA[layer][temp];
        #if DEBUGGING
        if ((this->gridCoords[0] == PROCESSOR_X_) && (this->gridCoords[1] == PROCESSOR_Y_) && (this->gridCoords[2] == PROCESSOR_Z_))
        {
          std::cout << "index - " << temp << " and val - " << buffer1[index1-1] << " and matWindow - " << matrixWindow << " and matCutSize - " << matrixCutSize << std::endl;
        }
        #endif
      }
    index2 += matrixCutSize;
    }
    // Now we broadcast..
    // Note that this broadcast will broadcast different sizes of buffer1, so on the receiving end, we will need another case statement
    // so that a properly-sized buffer is used.
    MPI_Bcast(&buffer1[0],buffer1.size(),MPI_DOUBLE,this->gridCoords[2],this->rowComm);
  }
  else
  {
    // receive the broadcast
    uint64_t buffer1Size = matrixWindow;
    buffer1Size *= matrixWindow;
    buffer1.resize(buffer1Size);
    // Note that depending on the key, the broadcast received here will be of different sizes. Need care when unpacking it later.
    MPI_Bcast(&buffer1[0],buffer1.size(),MPI_DOUBLE,this->gridCoords[2],this->rowComm);
  }

  if (this->columnCommRank == this->gridCoords[2])    // May want to recheck this later if bugs occur.
  {
    // Triangular
    buffer2 = std::move(this->matrixB); //std::move(this->holdTransposeL);			// Change the copy to a move, basically just changing names without copying
    // Note that this broadcast will broadcast different sizes of buffer1, so on the receiving end, we will need another case statement
    // so that a properly-sized buffer is used.
    MPI_Bcast(&buffer2[0],buffer2.size(),MPI_DOUBLE,this->gridCoords[2],this->columnComm);
  }
  else
  {
    // What I can do here is receive with a buffer of size matrixWindow*(matrixWindow+1)/2 and then iterate over the last part to see what kind it is
    // i was trying to do it with gridCoords, but then I would have needed this->rowRank, stuff like that so this is a bit easier
    // Maybe I can fix this later to make it more stable
    uint64_t buffer2Size = matrixWindow;
    buffer2Size *= (matrixWindow+1);
    buffer2Size >>= 1;
    buffer2.resize(buffer2Size);	// this is a special trick
    // Note that depending on the key, the broadcast received here will be of different sizes. Need care when unpacking it later.
    MPI_Bcast(&buffer2[0],buffer2.size(),MPI_DOUBLE,this->gridCoords[2],this->columnComm);
  }
  
  /*
    Now once here, I have the row data and the column data residing in buffer1 and buffer2
    I need to use the data in these buffers correctly to calculate partial sums for each place that I own. And store them together in order to
      properly reduce the sums. So use another temporary buffer. Only update this->matrixU or this->matrixL after the reduction if root or after the final
        broadcast if non-root.
  */


  // So iterate over everything that we own and calculate its partial sum that will be used in reduction
  // buffer3 will be used to hold the partial matrix sum result that will contribute to the reduction.
  uint64_t buffer4Size = matrixWindow;
  buffer4Size *= matrixWindow;
  std::vector<T> buffer4(buffer4Size,0.);	// need to give this outside scope

  // Triangular Multiplicaton -> (lower,square)

  uint64_t updatedVectorSize = matrixWindow;
  updatedVectorSize *= matrixWindow;
  std::vector<T> updatedVector(updatedVectorSize,0.);
  uint64_t index = 0;
  for (uint32_t i = 0; i<matrixWindow; i++)
  {
    uint64_t temp = i;
    temp *= matrixWindow;
    for (uint32_t j = 0; j<=i; j++)
    {
      uint64_t updatedVectorIndex = temp+j;
      updatedVector[updatedVectorIndex] = buffer2[index]; 
      index++;
    }
  }
  
  cblas_dtrmm(CblasRowMajor,CblasRight,CblasLower,CblasTrans,CblasNonUnit,matrixWindow,matrixWindow,1.,&updatedVector[0], matrixWindow, &buffer1[0], matrixWindow);
  MPI_Allreduce(&buffer1[0],&buffer4[0],buffer1.size(),MPI_DOUBLE,MPI_SUM,this->depthComm);
  
  uint64_t holdMatrixSize = matrixWindow;
  holdMatrixSize *= matrixWindow;
  //this->holdMatrix.resize(holdMatrixSize,0.);
      
  // Square
  uint64_t startOffset = dimXstartC;
  startOffset *= (dimXstartC+1);
  startOffset >>= 1;
  //int startOffset = (dimXstartC*(dimXstartC+1))>>1;
  uint64_t index1 = 0;
  uint64_t index2 = startOffset;
  index2 += dimYstartC;
  for (uint32_t i=0; i<matrixWindow; i++)
  {
    for (uint32_t j=0; j<matrixWindow; j++)
    {
      //this->matrixL[index2++] = buffer4[index1++];                    // Transpose has poor spatial locality. Could fix later if need be
      this->matrixC[index2++] = buffer4[index1++];
    }
    uint64_t temp = dimYstartC;
    temp += i;
    temp++;
    //index2 += (dimYstartC+i+1);
    index2 += temp;
  }
}

/*
	Matrix Multiplication of a square matrix with a square matrix
	
	matrixA - square cut of a triangular matrix that is packed in a triangular manner
	matrixB - full (no cut) square matrix that is packed in a square manner
	matrixC - full (no cut) square matrix that is packed in a square manner
*/
template<typename T>
void matrixMult<T>::multiply2
				(
         			uint32_t dimXstartA,
				uint32_t dimXendA,
				uint32_t dimYstartA,
				uint32_t dimYendA,
				uint32_t dimXstartB,
				uint32_t dimXendB,
				uint32_t dimYstartB,
				uint32_t dimYendB,
				uint32_t dimXstartC,
				uint32_t dimXendC,
				uint32_t dimYstartC,
				uint32_t dimYendC,
				uint32_t matrixWindow,
				uint32_t matrixSize
				)
{
  std::vector<T> buffer1;  // use capacity() or reserve() methods here?
  std::vector<T> buffer2;

  if (this->rowCommRank == this->gridCoords[2])     // different matches on each of the P^(1/3) layers
  {
    // Not triangular
    uint64_t buffer1Size = matrixWindow;
    buffer1Size *= matrixWindow;
    buffer1.resize(buffer1Size);
    uint64_t startOffset = dimXstartA;
    startOffset *= (dimXstartA+1);
    startOffset >>= 1;
    //int startOffset = (dimXstartA*(dimXstartA+1))>>1;
    uint64_t index1 = 0;
    uint64_t index2 = startOffset;
    index2 += dimYstartA;
    for (uint32_t i=0; i<matrixWindow; i++)
    {
      for (uint32_t j=0; j<matrixWindow; j++)
      {
        buffer1[index1++] = this->matrixA[index2++]; //this->matrixL[index2++];
      }
      uint64_t temp = dimYstartA;
      temp += i;
      temp++;
      //index2 += (dimYstartA+i+1);
      index2 += temp;
    }
    // Broadcast here?
    // Note that this broadcast will broadcast different sizes of buffer1, so on the receiving end, we will need another case statement
    // so that a properly-sized buffer is used.
    MPI_Bcast(&buffer1[0],buffer1.size(),MPI_DOUBLE,this->gridCoords[2],this->rowComm);
  }
  else
  {
    // receive broadcast
    uint64_t buffer1Size = matrixWindow;
    buffer1Size *= matrixWindow;
    buffer1.resize(buffer1Size);
    // Note that depending on the key, the broadcast received here will be of different sizes. Need care when unpacking it later.
    MPI_Bcast(&buffer1[0],buffer1.size(),MPI_DOUBLE,this->gridCoords[2],this->rowComm);
  }

  if (this->columnCommRank == this->gridCoords[2])    // May want to recheck this later if bugs occur.
  {
    // Not Triangular, but transpose
    buffer2 = std::move(this->matrixB);			// Changed the copy to a move
    // Note that this broadcast will broadcast different sizes of buffer1, so on the receiving end, we will need another case statement
    // so that a properly-sized buffer is used.
    MPI_Bcast(&buffer2[0],buffer2.size(),MPI_DOUBLE,this->gridCoords[2],this->columnComm);
  }
  else
  {
    uint64_t buffer2Size = matrixWindow;
    buffer2Size *= matrixWindow;
    buffer2.resize(buffer2Size);				// resize? Maybe reserve? Not sure what is more efficient?
    // Note that depending on the key, the broadcast received here will be of different sizes. Need care when unpacking it later.
    MPI_Bcast(&buffer2[0],buffer2.size(),MPI_DOUBLE,this->gridCoords[2],this->columnComm);
  }
  
  /*
    Now once here, I have the row data and the column data residing in buffer1 and buffer2
    I need to use the data in these buffers correctly to calculate partial sums for each place that I own. And store them together in order to
      properly reduce the sums. So use another temporary buffer. Only update this->matrixU or this->matrixL after the reduction if root or after the final
        broadcast if non-root.
  */


  // So iterate over everything that we own and calculate its partial sum that will be used in reduction
  // buffer3 will be used to hold the partial matrix sum result that will contribute to the reduction.
  uint64_t buffer4Size = matrixWindow;
  buffer4Size *= matrixWindow;
  std::vector<T> buffer4(buffer4Size,0.);	// need to give this outside scope
      
  // Matrix Multiplication -> (square,square)

  uint64_t buffer3Size = matrixWindow;
  buffer3Size *= matrixWindow;
  std::vector<T> buffer3(buffer3Size);

  cblas_dgemm(CblasRowMajor,CblasNoTrans,CblasTrans,matrixWindow,matrixWindow,matrixWindow,1.,&buffer1[0],matrixWindow,&buffer2[0],matrixWindow,1.,&buffer3[0],matrixWindow);

  // Lets use this->holdMatrix as my matrix C and lets right to it. Then we can perform a move operation on this->holdMatrix into this->matrixA[this->matrixA.size()-1]
  MPI_Allreduce(&buffer3[0],&buffer4[0],buffer3.size(),MPI_DOUBLE,MPI_SUM,this->depthComm);


/*      
      uint64_t tempSize = matrixWindow;		// should be the same as shift was in CholeskyRecurse function
      tempSize *= matrixWindow;
      this->matrixA.push_back(std::vector<T>(tempSize),0.);
      // Now I need a copy loop to grab the Lower-right quarter square from the current matrixA (sort of a waste of time. Think of ways to reduce this?)
      std::vector<T> holdSyrkData(matrixWindow*matrixWindow);
      uint64_t index1 = 0;
      uint64_t index2 = matrixWindow;
      index2 *= matrixCutSize;
      index2 += matrixWindow;
      for (uint32_t i=0; i<matrixWindow; i++)
      {
        for (uint32_t j=0; j<matrixWindow; j++)
        {
          uint64_t hold2 = index2;
          hold2 += j;
          holdSyrkData[index1++] = this->matrixA[this->matrixA.size()-2][hold2];
        }
        index2 += matrixCutSize;
      }
      cblas_ssyrk(CBlasRowMajor, CBlasLower, CBlasUpper, matrixWindow, matrixWindow, -1., );
      MPI_Allreduce(&holdMatrix[0],&this->matrixA[this->matrixA.size()-1][0],this->holdMatrix.size(),MPI_DOUBLE,MPI_SUM,this->depthComm);
*/
  
  uint64_t holdMatrixSize = matrixWindow;
  holdMatrixSize *= matrixWindow;
  //this->holdMatrix.resize(holdMatrixSize,0.);
      
  uint64_t index1 = 0;
  for (uint32_t i=0; i<matrixWindow; i++)
  {
    uint64_t temp = i;
    temp *= matrixWindow;
    for (uint32_t j=0; j<matrixWindow; j++)
    {
      uint64_t holdMatrixIndex = temp;
      holdMatrixIndex += j;
      //this->holdMatrix[holdMatrixIndex] = buffer4[index1++];
      this->matrixC[holdMatrixIndex] = buffer4[index1++];
    }
  }
}

/*
	Matrix Multiplication of a Square matrix and a lower triangular matrix
	
	matrixA - square cut of a triangular matrix that is packed in a triangular manner
	matrixB - triangular cut of a triangular matrix that is packed in a triangular manner
	matrixC - full (no cut) square matrix that is packed in a square manner
*/
template<typename T>
void matrixMult<T>::multiply3
				(
         			uint32_t dimXstartA,
				uint32_t dimXendA,
				uint32_t dimYstartA,
				uint32_t dimYendA,
				uint32_t dimXstartB,
				uint32_t dimXendB,
				uint32_t dimYstartB,
				uint32_t dimYendB,
				uint32_t dimXstartC,
				uint32_t dimXendC,
				uint32_t dimYstartC,
				uint32_t dimYendC,
				uint32_t matrixWindow,
				uint32_t matrixSize
				)
{
  std::vector<T> buffer1;  // use capacity() or reserve() methods here?
  std::vector<T> buffer2;

  if (this->rowCommRank == this->gridCoords[2])     // different matches on each of the P^(1/3) layers
  {
    // Not Triangular
    uint64_t buffer1Size = matrixWindow;
    buffer1Size *= matrixWindow;
    buffer1.resize(buffer1Size);
    uint64_t startOffset = dimXstartA;
    startOffset *= (dimXstartA+1);
    startOffset >>= 1;
    //int startOffset = (dimXstartA*(dimXstartA+1))>>1;
    uint64_t index1 = 0;
    uint64_t index2 = startOffset;
    index2 += dimYstartA;
    for (uint64_t i=0; i<matrixWindow; i++)
    {
      for (uint64_t j=0; j<matrixWindow; j++)
      {
        buffer1[index1++] = this->matrixA[index2++]; //this->matrixL[index2++];
      }
      uint64_t temp = dimYstartA;
      temp += i;
      temp++;
      index2 += temp;
      //index2 += (dimYstartA+i+1);
    }
    // broadcast here?
    // Note that this broadcast will broadcast different sizes of buffer1, so on the receiving end, we will need another case statement
    // so that a properly-sized buffer is used.
    MPI_Bcast(&buffer1[0],buffer1.size(),MPI_DOUBLE,this->gridCoords[2],this->rowComm);
  }
  else
  {
    // receive broadcast
    uint64_t buffer1Size = matrixWindow;
    buffer1Size *= matrixWindow;
    buffer1.resize(buffer1Size);
    // Note that depending on the key, the broadcast received here will be of different sizes. Need care when unpacking it later.
    MPI_Bcast(&buffer1[0],buffer1.size(),MPI_DOUBLE,this->gridCoords[2],this->rowComm);
  }

  if (this->columnCommRank == this->gridCoords[2])    // May want to recheck this later if bugs occur.
  {
    // Triangular -> Lower
    // As noted above, size depends on whether or not the gridCoords lie in the lower-triangular portion of first block
    uint64_t buffer2Size = matrixWindow;
    buffer2Size *= (matrixWindow+1);
    buffer2Size >>= 1;
    buffer2.resize(buffer2Size);	// change to bitshift later
    uint64_t startOffset = dimXstartB;
    startOffset *= (dimXstartB+1);
    startOffset >>= 1;
    //int startOffset = (dimXstartB*(dimXstartB+1))>>1;
    uint64_t index1 = 0;
    uint64_t index2 = startOffset;
    index2 += dimYstartB;
    for (uint32_t i=0; i<matrixWindow; i++)
    {
      for (uint32_t j=0; j<=i; j++)
      {
        buffer2[index1++] = this->matrixB[index2++]; //this->matrixLInverse[index2++];
      }
      index2 += dimYstartB;
    }
    // Note that this broadcast will broadcast different sizes of buffer1, so on the receiving end, we will need another case statement
    // so that a properly-sized buffer is used.
    MPI_Bcast(&buffer2[0],buffer2.size(),MPI_DOUBLE,this->gridCoords[2],this->columnComm);
  }
  else
  {
    uint64_t buffer2Size = matrixWindow;
    buffer2Size *= (matrixWindow+1);
    buffer2Size >>= 1;
    buffer2.resize(buffer2Size);	// this is a special trick
    // Note that depending on the key, the broadcast received here will be of different sizes. Need care when unpacking it later.
    MPI_Bcast(&buffer2[0],buffer2.size(),MPI_DOUBLE,this->gridCoords[2],this->columnComm);
  }
  
  /*
    Now once here, I have the row data and the column data residing in buffer1 and buffer2
    I need to use the data in these buffers correctly to calculate partial sums for each place that I own. And store them together in order to
      properly reduce the sums. So use another temporary buffer. Only update this->matrixU or this->matrixL after the reduction if root or after the final
        broadcast if non-root.
  */


  // So iterate over everything that we own and calculate its partial sum that will be used in reduction
  // buffer3 will be used to hold the partial matrix sum result that will contribute to the reduction.
  uint64_t buffer4Size = matrixWindow;
  buffer4Size *= matrixWindow;
  std::vector<T> buffer4(buffer4Size,0.);	// need to give this outside scope
      
  // Triangular Multiplication -> (square,lower)
  uint64_t updatedVectorSize = matrixWindow;
  updatedVectorSize *= matrixWindow;
  std::vector<T> updatedVector(updatedVectorSize,0.);
  uint64_t index = 0;
  for (uint32_t i = 0; i<matrixWindow; i++)
  {
    uint64_t temp = i;
    temp *= matrixWindow;
    //int temp = i*matrixWindow;
    for (uint32_t j = 0; j <= i; j++)	// this could be wrong ???
    {
      uint64_t updatedVectorIndex = temp;
      updatedVectorIndex += j;
      updatedVector[updatedVectorIndex] = buffer2[index++];
    }
  }
      
  cblas_dtrmm(CblasRowMajor,CblasRight,CblasLower,CblasNoTrans,CblasNonUnit,matrixWindow,matrixWindow,1.,&updatedVector[0],matrixWindow,&buffer1[0],matrixWindow);
  MPI_Allreduce(&buffer1[0],&buffer4[0],buffer1.size(),MPI_DOUBLE,MPI_SUM,this->depthComm);	// use buffer4
  
  uint64_t holdMatrixSize = matrixWindow;
  holdMatrixSize *= matrixWindow;
  //this->holdMatrix.resize(holdMatrixSize,0.);
      
  uint64_t index1 = 0;
  uint64_t index2 = 0;
  for (uint32_t i=0; i<matrixWindow; i++)
  {
    for (uint32_t j=0; j<matrixWindow; j++)
    {
      //this->holdMatrix[index2++] = buffer4[index1++];
      this->matrixC[index2++] = buffer4[index1++];
    }
  }
}

/*
	Matrix Multiplication of a Lower triangular and a square matrix
	
	matrixA - triangular cut of a triangular matrix that is packed in a triangular manner
	matrixB - full (no cut) square matrix that is packed in a square manner
	matrixC - square cut of a triangular matrix that is packed in a triangular manner
*/
template<typename T>
void matrixMult<T>::multiply4
				(
         			uint32_t dimXstartA,
				uint32_t dimXendA,
				uint32_t dimYstartA,
				uint32_t dimYendA,
				uint32_t dimXstartB,
				uint32_t dimXendB,
				uint32_t dimYstartB,
				uint32_t dimYendB,
				uint32_t dimXstartC,
				uint32_t dimXendC,
				uint32_t dimYstartC,
				uint32_t dimYendC,
				uint32_t matrixWindow,
				uint32_t matrixSize
				)
{
  std::vector<T> buffer1;  // use capacity() or reserve() methods here?
  std::vector<T> buffer2;

  if (this->rowCommRank == this->gridCoords[2])     // different matches on each of the P^(1/3) layers
  {
    // Triangular -> Lower
    // As noted above, size depends on whether or not the gridCoords lie in the lower-triangular portion of first block
    uint64_t buffer1Size = matrixWindow;
    buffer1Size *= (matrixWindow+1);
    buffer1Size >>= 1;
    buffer1.resize(buffer1Size);
    uint64_t startOffset = dimXstartA;
    startOffset *= (dimXstartA+1);
    startOffset >>= 1;
    //int startOffset = (dimXstartA*(dimXstartA+1))>>1;
    uint64_t index1 = 0;
    uint64_t index2 = startOffset;
    index2 += dimYstartA;
    for (uint32_t i=0; i<matrixWindow; i++)		// start can take on 2 values corresponding to the situation above: 1 or 0
    {
      for (uint32_t j=0; j<=i; j++)
      {
        buffer1[index1++] = this->matrixA[index2++];//this->matrixLInverse[index2++];
      }
      index2 += dimYstartA;
    }
    // Broadcast here?
    // Note that this broadcast will broadcast different sizes of buffer1, so on the receiving end, we will need another case statement
    // so that a properly-sized buffer is used.
    MPI_Bcast(&buffer1[0],buffer1.size(),MPI_DOUBLE,this->gridCoords[2],this->rowComm);
  }
  else
  {
    // receive broadcast here
    uint64_t buffer1Size = matrixWindow;
    buffer1Size *= (matrixWindow+1);
    buffer1Size >>= 1;
    buffer1.resize(buffer1Size);
    // Note that depending on the key, the broadcast received here will be of different sizes. Need care when unpacking it later.
    MPI_Bcast(&buffer1[0],buffer1.size(),MPI_DOUBLE,this->gridCoords[2],this->rowComm);
  }

  if (this->columnCommRank == this->gridCoords[2])    // May want to recheck this later if bugs occur.
  {
    // Not Triangular, but this requires a special traversal because we are using this->holdMatrix
    uint64_t buffer2Size = matrixWindow;
    buffer2Size *= matrixWindow;
    buffer2.resize(buffer2Size);
    //int startOffset = (dimXstartB*(dimXstartB-1))>>1;
    uint64_t index1 = 0;
    uint64_t index2 = 0;
    //int index2 = startOffset + dimYstartB;
    for (uint32_t i=0; i<matrixWindow; i++)
    {
      for (uint32_t j=0; j<matrixWindow; j++)
      {
        buffer2[index1++] = (-1)*this->matrixB[index2++];//(-1)*this->holdMatrix[index2++];
      }
      //index2 += dimYstartB;
    }
    // Note that this broadcast will broadcast different sizes of buffer1, so on the receiving end, we will need another case statement
    // so that a properly-sized buffer is used.
    MPI_Bcast(&buffer2[0],buffer2.size(),MPI_DOUBLE,this->gridCoords[2],this->columnComm);
  }
  else
  {
    uint64_t buffer2Size = matrixWindow;
    buffer2Size *= matrixWindow;
    buffer2.resize(buffer2Size);
    // Note that depending on the key, the broadcast received here will be of different sizes. Need care when unpacking it later.
    MPI_Bcast(&buffer2[0],buffer2.size(),MPI_DOUBLE,this->gridCoords[2],this->columnComm);
  }
  
  /*
    Now once here, I have the row data and the column data residing in buffer1 and buffer2
    I need to use the data in these buffers correctly to calculate partial sums for each place that I own. And store them together in order to
      properly reduce the sums. So use another temporary buffer. Only update this->matrixU or this->matrixL after the reduction if root or after the final
        broadcast if non-root.
  */


  // So iterate over everything that we own and calculate its partial sum that will be used in reduction
  // buffer3 will be used to hold the partial matrix sum result that will contribute to the reduction.
  uint64_t buffer4Size = matrixWindow;
  buffer4Size *= matrixWindow;
  std::vector<T> buffer4(buffer4Size,0.);	// need to give this outside scope
      
  // Triangular Multiplication -> (lower,square)
  uint64_t updatedVectorSize = matrixWindow;
  updatedVectorSize *= matrixWindow;
  std::vector<T> updatedVector(updatedVectorSize,0.);
  uint64_t index = 0;
  for (uint32_t i = 0; i<matrixWindow; i++)
  {
    uint64_t temp = i;
    temp *= matrixWindow;
    //int temp = i*matrixWindow;
    for (uint32_t j = 0; j <= i; j++)	// This could be wrong?
    {
      uint64_t updatedVectorSize = temp;
      updatedVectorSize += j;
      updatedVector[updatedVectorSize] = buffer1[index++];
    }
  }

  cblas_dtrmm(CblasRowMajor,CblasLeft,CblasLower,CblasNoTrans,CblasNonUnit,matrixWindow,matrixWindow,1.,&updatedVector[0],matrixWindow,&buffer2[0],matrixWindow);
  MPI_Allreduce(&buffer2[0],&buffer4[0],buffer2.size(),MPI_DOUBLE,MPI_SUM,this->depthComm);
  
  uint64_t holdMatrixSize = matrixWindow;
  holdMatrixSize *= matrixWindow;
  //this->holdMatrix.resize(holdMatrixSize,0.);
      
  uint64_t startOffset = dimXstartC;
  startOffset *= (dimXstartC+1);
  startOffset >>= 1;
  //int startOffset = (dimXstartC*(dimXstartC+1))>>1;
  uint64_t index1 = 0;
  uint64_t index2 = startOffset;
  index2 += dimYstartC;
  for (uint32_t i=0; i<matrixWindow; i++)
  {
    for (uint32_t j=0; j<matrixWindow; j++)
    {
      //this->matrixLInverse[index2++] = buffer4[index1++];
      this->matrixA[index2++] = buffer4[index1++];			// Special case where I write to A and not C 
    }
    uint64_t temp = dimYstartC;
    temp += i;
    temp++;
    index2 += temp;
  }
}

/*
	Matrix Multiplication of a rectangular matrix with a rectangular matrix

	Currently used for Q=AR^{-1}
	
	matrixA - full (no cut) rectangular matrix that is packed in a rectangular manner
	matrixB - full (no cut) triangular matrix that is packed in a rectangular manner
	matrixC - full (no cut) rectangular matrix that is packed in a rectangular manner
*/
template<typename T>
void matrixMult<T>::multiply5
				(
				uint32_t matrixWindowRow,
				uint32_t matrixWindowCol,
				uint32_t matrixSizeRow,
				uint32_t matrixSizeCol
				)
{
  std::vector<T> buffer1;  // use capacity() or reserve() methods here?
  std::vector<T> buffer2;

  if (this->rowCommRank == this->gridCoords[2])     // different matches on each of the P^(1/3) layers
  {
  }
}
