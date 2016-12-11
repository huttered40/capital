//#include "solver.h" careful here

template<typename T>
solver<T>::solver(int rank, int size, int nDims, int matrixDimSize)
{
  /*
	save arguments as member variables
  */
  this->worldRank = rank;
  this->worldSize = size;
  this->nDims = nDims;						// should always be 3
  this->matrixDimSize = matrixDimSize;

  /*
	Lets precompute a list of cubes for quick lookUp of cube dimensions based on processor size
  */

  for (int i=1; i<500; i++)					// I can change the bounds [1,1000] later
  {
    this->gridSizeLookUp[i*i*i] = i; 				// quick lookup for (grid size, 2d sheet size)
  }
}

template <typename T>
void solver<T>::startUp(bool &flag)
{
  /*
	Look up to see if the total processor size is a cube. If not, then return.
	We can reduce this restriction after we have it working for a perfect cubic processor grid
  
	If found, this->processorGridDimSize will represent the number of processors along a row or column (P^(1/3))
  */

  if (this->gridSizeLookUp.find(this->worldSize) == gridSizeLookUp.end())
  {
    flag = true;
    return;
  }

  this->processorGridDimSize = this->gridSizeLookUp[this->worldSize];	// Identifies number of processors in each dimension. This will tell use how many processes are in rowComm and colComm

  /*
	this->baseCaseSize gives us a size in which to stop recursing in the LURecurse method
	= n/P^(2/3). The math behind it is written about in my report/other papers
  
	HUGE NOTE: I changed this to be n/P, instead of n/P^(2/3), since we are doing it with only P^{2/3} processors
		instead of all P.

	Just watch out. We need to make sure that n/P > 0, or else weird stuff could happen
  */
  this->baseCaseSize = this->matrixDimSize/(this->processorGridDimSize*this->processorGridDimSize); // could be changed to P^(2/3) later if problems occur

  /*
    For setting this->gridDims in the future, I may want to use MPI_Dims_create
  */
  this->gridDims.resize(this->nDims,this->processorGridDimSize);
  this->gridCoords.resize(this->nDims); 
  
  /*
	Prepare to create some communicators that will be needed later
	The Cart Communicator is mainly used to distribute the random data in a block-cyclic fashion
	The other communicators are created for communication reasons. Will come back to this later
  */
  std::vector<int> boolVec(3,0);
  MPI_Cart_create(MPI_COMM_WORLD, this->nDims, &this->gridDims[0], &boolVec[0], false, &this->grid3D);

  /*  
	I don't know if I even really need this->gridRank. I'll keep it in here but I may scrap it later.
	It may also be the source of bugs if I use this->worldRank instead of this->gridRank if I am using the Cart Communicator for something
  */

  MPI_Comm_rank(this->grid3D, &this->grid3DRank);
  MPI_Comm_size(this->grid3D, &this->grid3DSize);

  /*
	Remember that this call, like the 2 above it, are collective, meaning that every rank present in the communicator must call it. So MPI_Cart_coords gives a different 3-tuple location to each rank/processor in the cartesian communicator.
  */
  MPI_Cart_coords(this->grid3D, this->grid3DRank, this->nDims, &this->gridCoords[0]);

  /*
    Secret: Before I create row and column communicators, I need to split grid3D into a separate communicator for its layers.
      This is key because it is the only way to get 
  */

  // 2D (xy) Layer Communicator (split by z coordinate)
  MPI_Comm_split(this->grid3D, this->gridCoords[2],this->grid3DRank,&this->layerComm);
  MPI_Comm_rank(this->layerComm,&layerCommRank);
  MPI_Comm_size(this->layerComm,&layerCommSize);
  // Row Communicator
  MPI_Comm_split(this->layerComm, this->gridCoords[0],this->gridCoords[1],&this->rowComm);
  MPI_Comm_rank(this->rowComm,&this->rowCommRank);
  MPI_Comm_size(this->rowComm,&this->rowCommSize);
  // column Communicator
  MPI_Comm_split(this->layerComm, this->gridCoords[0],this->gridCoords[1],&this->colComm);
  MPI_Comm_rank(this->colComm,&this->colCommRank);
  MPI_Comm_size(this->colComm,&this->colCommSize);
  // Depth Communicator
  MPI_Comm_split(this->grid3D,this->gridCoords[0]*this->processorGridDimSize+this->gridCoords[1],this->gridCoords[2],&this->depthComm);
  MPI_Comm_rank(this->depthComm,&this->depthCommRank);
  MPI_Comm_rank(this->depthComm,&this->depthCommSize);

}


/*
  Lets focus only on a cyclic distribution for now. I can try to add block-cyclic or blocked distributions later.
*/

template <typename T>
void solver<T>::collectDataCyclic()
{
  /*
	There is still a problem here. I need a way to differentiate between the different layers.
	The same matrix should be laid on top of each.
  
	I am using the gridCoords[i] array that I attained through my call to MPI_Cart_create.
  	Notice how the gridCoords[2] value is forgotten, because we treat each z-layer the same. The same matrix A is embedded there.
  
        I also wanted to initialize the results matrices to 0 so that I can just build them up and not have to worry about garbage values

	When writing this code to distribute the values, you have to think of who lives there. Well if we think using the Cartesian rank coordinates,
		then we dont care what layer (in the z-direction) we are in. We care about the (x,y,?) coordinates. Sublayers exist in a block-cyclic distribution, so that in the first sublayer of the matrix, all processors are used, therefore, we cycle them from (0,1,2,0,1,2,0,1,2,....) in the x-dimension of te cube (when traveling down from the top-front-left corner) to the bottom-front-left corner. And then for each of those, there are the processors in the y-direction as well, which are representing via the nested loop.

*/

  for (int i=this->gridCoords[0]; i<this->matrixDimSize; i+=this->processorGridDimSize)
  {
    std::vector<T> temp;
    for (int j=this->gridCoords[1]; j<this->matrixDimSize; j+=this->processorGridDimSize)
    {
      srand(i*this->matrixDimSize + j);			// this is very important. Needs to be the same for all processors
    							// completely based on where we are in entire matrix
      temp.push_back((rand()%100)*1./100.);
      if (i==j)
      {
        temp[temp.size()-1]+=10.;		// we want all diagonals to be dominant.
      }
    }

    std::vector<T> fillZeros(temp.size(),temp[0]*0.);	// This may throw a compiler error
    matrixL.push_back(fillZeros);
    matrixU.push_back(fillZeros);
    matrixLInverse.push_back(fillZeros);
    matrixUInverse.push_back(fillZeros);
    matrixA.push_back(temp);
    matrixB.push_back(fillZeros);		// I could fill this up with A or just start it off at zeros. I reset it anyway 
  }
  
  return;
}


/********************************************************************************************************************************************/
// This divides the program. If anything above this point is incorrect, then everything below will be incorrect as well.
/********************************************************************************************************************************************/




/*
  The solve method will initiate the solving of this LU Factorization algorithm
  All ranks better be ready to go in order to solve this problem.
*/
template<typename T>
void solver<T>::solve()
{
  /*
	Note: I changed the arguments from this->matrixDimSize, which doesnt help given the fact that we are dealing only with the 2-d vectors
		that each process holds, to this->matrixA.size(), which does represent what we are cutting up
  
	Note: There may be bugs in how I pass in these 5 arguments to LURecurse. Look for this if weird things occur.

	Check the final argument. For safety reasons, I changed it. Be careful!
  */

  LURecurse(0,this->matrixA.size(),0,this->matrixA.size(),this->matrixA.size(),this->matrixDimSize,0);  // start on track 1
}

/*
  This function is where I can recurse within each of the P^{1/3} layers separately
  But after that, we perform matrix multiplication with all P processors in the 3-d grid.
*/
template<typename T>
void solver<T>::LURecurse(int dimXstart, int dimXend, int dimYstart, int dimYend, int matrixWindow, int matrixSize, int matrixTrack)
{

  if (matrixSize == this->baseCaseSize) 	// recursive base case
  {
    LURecurseBaseCase(dimXstart,dimXend,dimYstart,dimYend,matrixWindow,matrixSize,matrixTrack);   // big decision here...
    return;
  }

  /*
	Recursive case -> Keep recursing on the top-left half
	Then I need to solve the recursive matrix multiplications. This is non-trivial because I need to spread out the data.
	Then I recurse again.
  */

  LURecurse(dimXstart,dimXend-matrixWindow/2,dimYstart,dimYend-matrixWindow/2,matrixWindow/2,matrixSize/2,matrixTrack);
  
  MM(dimXstart+matrixWindow/2,dimXend,dimYstart,dimYend-matrixWindow/2,dimXstart,dimXend-matrixWindow/2,dimYstart,dimYend-matrixWindow/2,dimXstart+matrixWindow/2,dimXend,dimYstart,dimYend-matrixWindow/2,matrixWindow/2,matrixSize/2,0,matrixTrack);
  
  MM(dimXstart,dimXend-matrixWindow/2,dimYstart,dimYend-matrixWindow/2,dimXstart,dimXend-matrixWindow/2,dimYstart+matrixWindow/2,dimYend,dimXstart,dimXend-matrixWindow/2,dimYstart+matrixWindow/2,dimYend,matrixWindow/2,matrixSize/2,1,matrixTrack);

  //Below is the tricky one. We need to create a vector via MM(..), then subtract it from A via some sort easy method...
  MM(dimXstart+matrixWindow/2,dimXend,dimYstart,dimYend-matrixWindow/2,dimXstart,dimXend-matrixWindow/2,dimYstart+matrixWindow/2,dimYend,-1,-1,-1,-1,matrixWindow/2,matrixSize/2,2,matrixTrack);

  // Big question: LURecurseBaseCase relies on taking from matrixA, but in this case, we would have to take from a modified matrix via the TRSM. How should the program access this and deal with it????

  for (int i=0; i<matrixWindow/2; i++)
  {
    for (int j=0; j<matrixWindow/2; j++)
    {
      this->matrixB[dimXstart+matrixWindow/2+i][dimYstart+matrixWindow/2+j] = this->matrixA[dimXstart+matrixWindow/2+i][dimYstart+matrixWindow/2+j] - this->holdMatrix[i][j];
    }
  }
  LURecurse(dimXstart+matrixWindow/2,dimXend,dimYstart+matrixWindow/2,dimYend,matrixWindow/2,matrixSize/2,1);  // HALT! Change the track!

  // These last 4 matrix multiplications are for building up our LInverse and UInverse matrices
  MM(dimXstart+matrixWindow/2,dimXend,dimYstart,dimYend-matrixWindow/2,dimXstart,dimXend-matrixWindow/2,dimYstart,dimYend-matrixWindow/2,-1,-1,-1,-1,matrixWindow/2,matrixSize/2,3,matrixTrack);

  MM(dimXstart+matrixWindow/2,dimXend,dimYstart+matrixWindow/2,dimYend,-1,-1,-1,-1,dimXstart+matrixWindow/2,dimXend,dimYstart,dimYend-matrixWindow/2,matrixWindow/2,matrixSize/2,3,matrixTrack);

  MM(dimXstart,dimXend-matrixWindow/2,dimYstart+matrixWindow/2,dimYend,dimXstart+matrixWindow/2,dimXend,dimYstart+matrixWindow/2,dimYend,-1,-1,-1,-1,matrixWindow/2,matrixSize/2,3,matrixTrack);

  MM(dimXstart,dimXend-matrixWindow/2,dimYstart,dimYend-matrixWindow/2,-1,-1,-1,-1,dimXstart,dimXend-matrixWindow/2,dimYstart+matrixWindow/2,dimYend,matrixWindow/2,matrixSize/2,3,matrixTrack);
}

/*
  Recursive Matrix Multiplication -> needs arguments that makes sense.

  This routine requires that matrix data is duplicated across all layers of the 3D Processor Grid
*/
template<typename T>
void solver<T>::MM(int dimXstartA,int dimXendA,int dimYstartA,int dimYendA,int dimXstartB,int dimXendB,int dimYstartB,int dimYendB,int dimXstartC, int dimXendC, int dimYstartC, int dimYendC, int matrixWindow,int matrixSize, int key, int matrixTrack)
{
  /*
    I need two broadcasts, then a reduction, then a broadcast
  */

  std::vector<T> buffer1(matrixWindow*matrixWindow);
  std::vector<T> buffer2(matrixWindow*matrixWindow);
  if (this->gridCoords[0] == this->gridCoords[2])     // different matches on each of the P^(1/3) layers
  {
    for (int i=dimXstartA; i<dimXendA; i++)
    {
      for (int j=dimYstartA; j<dimYendA; j++)
      {
        int index = (i-dimXstartA)*matrixWindow+(j-dimYstartA);
        switch (key)
        {
          case 0:
          {
            buffer1[index] = (matrixTrack ? this->matrixB[i][j] : this->matrixA[i][j]);
            break;
          }
          case 1:
          {
            buffer1[index] = this->matrixLInverse[i][j];
            break;
          }
          case 2:
          {
            buffer1[index] = this->matrixL[i][j];
            break;
          }
          case 3:
          {
            buffer1[index] = this->matrixL[i][j];
            break;
          }
          case 4:
          {
            // We need to make buffer1 negative by multiplying each guy by (-1)
            buffer1[index] = (-1)*this->matrixLInverse[i][j];
            break;
          }
          case 5:
          {
            buffer1[index] = this->matrixU[i][j];
            break;
          }
          case 6:
          {
            // We need to make buffer1 negative by multiplying each guy by (-1)
            buffer1[index] = (-1)*this->matrixUInverse[i][j];
            break;
          }
        }
      }
    }
    MPI_Bcast(&buffer1[0],buffer1.size(),MPI_DOUBLE,this->gridCoords[2] /*I may want to change this*/ ,this->rowComm);
  }
  else
  {
    MPI_Bcast(&buffer1[0],buffer1.size(),MPI_DOUBLE,this->gridCoords[2] /*I may want to change this*/ ,this->rowComm);
  }
  
  if (this->gridCoords[1] == this->gridCoords[2])    // May want to recheck this later if bugs occur.
  {
    for (int i=dimXstartB; i<dimXendB; i++)
    {
      for (int j=dimYstartB; j<dimYendB; j++)
      {
        int index = (i-dimXstartB)*matrixWindow+(j-dimYstartB);
        switch (key)
        {
          case 0:
          {
            buffer2[index] = this->matrixUInverse[i][j];
            break;
          }
          case 1:
          {
            buffer2[index] = (matrixTrack ? this->matrixB[i][j] : this->matrixA[i][j]);
            break;
          }
          case 2:
          {
            buffer2[index] = this->matrixU[i][j];
            break;
          }
          case 3:
          {
            buffer2[index] = this->matrixLInverse[i][j];
            break;
          }
          case 4:
          {
            // Tricky part
            buffer2[index] = this->holdMatrix[i-dimXstartB][j-dimYstartB];
            break;
          }
          case 5:
          {
            buffer2[index] = this->matrixUInverse[i][j];
          }
          case 6:
          {
            // Tricky part
            buffer2[index] = this->holdMatrix[i-dimXstartB][j-dimYstartB];
            break;
          }
        }
      }
    }
    MPI_Bcast(&buffer2[0],buffer2.size(),MPI_DOUBLE,this->gridCoords[2] /*I may want to change this*/ ,this->colComm);
  }
  else
  {
    MPI_Bcast(&buffer2[0],buffer2.size(),MPI_DOUBLE,this->gridCoords[2] /*I may want to change this*/ ,this->colComm);
  }

  /*
    Now once here, I have the row data and the column data residing in buffer1 and buffer2
    I need to use the data in these buffers correctly to calculate partial sums for each place that I own. And store them together in order to
      properly reduce the sums. So use another temporary buffer. Only update this->matrixU or this->matrixL after the reduction if root or after the final
        broadcast if non-root.
  */

  // So iterate over everything that we own and calculate its partial sum that will be used in reduction
  std::vector<T> buffer3(matrixWindow*matrixWindow,0.);
  for (int i=0; i<matrixWindow; i++)
  {
    for (int j=0; j<matrixWindow; j++)
    {
      // Now for each guy, I need to calculate its partial sum
      for (int k=0; k<matrixWindow; k++)
      {
        buffer3[i*matrixWindow+j] += (buffer1[i*matrixWindow+k]*buffer2[j+k*matrixWindow]);	// Watch the indexing on these!! Tricky!
      }
    }
  }

  // Now we can reduce these values via a reduction onto a root (1st layer)
  std::vector<T> buffer4(matrixWindow*matrixWindow,0.);
  MPI_Reduce(&buffer3[0],&buffer4[0],buffer3.size(),MPI_DOUBLE,MPI_SUM,0,this->depthComm);  // every depth subCommunicator has a 0 rank

  MPI_Bcast(&buffer4[0],buffer4.size(),MPI_DOUBLE,0,this->depthComm);

  /*
    Now I need a case statement of where to put the guy that was just broadasted...
  */

  this->holdMatrix.clear();
  this->holdMatrix.resize(matrixWindow*matrixWindow);
  for (int i=dimXstartC; i<dimXendC; i++)
  {
    for (int j=dimYstartC; j<dimYendC; j++)
    {
      int index = (i-dimXstartC)*matrixWindow+(j-dimYstartC);
      switch(key)
      {
        case 0:
        {
          this->matrixL[i][j] = buffer4[index];
          break;
        }
        case 1:
        {
          this->matrixU[i][j] = buffer4[index];
          break;
        }
        case 2:
        {
          // tricky. Figure this out
          this->holdMatrix[i-dimXstartC][j-dimYstartC] = buffer4[index];
          break;
        }
        case 3:
        {
          // tricky. Figure this out
          this->holdMatrix[i-dimXstartC][dimYstartC] = buffer4[index];
          break;
        }
        case 4:
        {
          this->matrixLInverse[i][j] = buffer4[index];
          break;
        }
        case 5:
        {
          // tricky. Figure this out
          this->holdMatrix[i-dimXstartC][j-dimYstartC] = buffer4[index];
          break;
        }
        case 6:
        {
          this->matrixUInverse[i][j] = buffer4[index];
        }
      }
    }
  }
}

/*
  Base case of LURecurse
*/
template<typename T>
void solver<T>::LURecurseBaseCase(int dimXstart, int dimXend, int dimYstart, int dimYend, int matrixWindow, int matrixSize, int matrixTrack)
{
  /*
	1) AllGather onto a single processor, which has to be chosen carefully
	2) Sequential BLAS-3 routines to solve for LU and inverses

	For now, I can just create a 1-d buffer of the data that I want to send first.
	Later, I can try to optimize this to avoid needless copying just for a collective

	Also remember the importance of using the corect communicator here. We need the sheet communicator of size P^{2/3}
	This avoids traffic and is the only way it could work

	I think that I will need to receive the data in a buffer, and then use another buffer to line up the data properly
	This is extra computational cost, but for the moment, I don't see any other option

	This data will be used to solve for L,U,L-I, and U-I. So it is very important that the input buffer be correct for BLAS-3
	Then we will work on communicating the correct data to the correct places so that RMM operates correctly.
 
	So below, I need to transfer a 2-d vector into a 1-d vector
  */
  
  std::vector<T> sendBuffer;

  if (matrixTrack == 0)
  {
    for (int i=dimXstart; i<dimXend; i++)
    {
      for (int j=dimYstart; j<dimYend; j++)
      {
        sendBuffer.push_back(this->matrixA[i][j]);
      }
    }
  }
  else /* matrixTrack == 1 */
  {
    for (int i=dimXstart; i<dimXend; i++)
    {
      for (int j=dimYstart; j<dimYend; j++)
      {
        sendBuffer.push_back(this->matrixB[i][j]);
      }
    }
  }

  std::vector<T> recvBuffer(sendBuffer.size()*this->processorGridDimSize*this->processorGridDimSize,0);

  MPI_Allgather(&sendBuffer[0], sendBuffer.size(),MPI_DOUBLE,&recvBuffer[0],sendBuffer.size(),MPI_DOUBLE,this->layerComm);
  /*
	After the all-gather, I need to format recvBuffer so that it works with OpenBLAS routines!
  */

  int count = 0;
  std::vector<T> realMatrix(matrixSize*matrixSize);		// make this 1-D to feed into BLAS??
  for (int i=0; i<(this->processorGridDimSize*this->processorGridDimSize); i++)  // MACRO loop over all processes' data (stored contiguously)
  {
    for (int j=0; j<matrixWindow; j++)
    {
      for (int k=0; k<matrixWindow; k++)
      {
        int index = j*this->processorGridDimSize*matrixSize+(i/this->processorGridDimSize)*matrixSize + k*this->processorGridDimSize+(i%this->processorGridDimSize);  // remember that recvBuffer is stored as P^(2/3) consecutive blocks of matrix data pertaining to each p
        realMatrix[index] = recvBuffer[count++];
      }
    }
  }

  // As of right here, realMatrix is a 1-d row-order matrix. It is correct
  /* Use BLAS to find L and U matrices */
  std::vector<int> pivotVector(matrixSize);
  LAPACKE_dgetrf(LAPACK_ROW_MAJOR,matrixSize,matrixSize,&realMatrix[0],matrixSize,&pivotVector[0]); // column major???
  /*
	Now right now, realMatrix contains both L and U. I need to now get L-Inverse and U-inverse via dtrtri

        only need a select part of L,U,L-I, and U-I. Pick out the data cyclically like we did initialliy
	Then put this data in the correct spot dictated by the arguments into the respective 2-d vectors
  */
  std::vector<T> matrix_U = realMatrix;		// big copy here. Maybe do a std::move operation here, then a copy, so 1 and not 2 copies?
  std::vector<T> matrix_L = realMatrix;
  for (int i=0; i<matrixSize; i++)
  {
    for (int j=0; j<matrixSize; j++)
    {
      if (i==j)
      {
        matrix_L[i*matrixSize+j] = 1.;
      }
      if (i<j)
      {
        matrix_U[i*matrixSize+j] = 0.;
      }
      else
      {
        matrix_L[i*matrixSize+j] = 0.;
      }
    }
  }

  std::vector<T> matrixU_inverse = matrix_U;		// big copy here
  std::vector<T> matrixL_inverse = matrix_L;
  
  LAPACKE_dtrtri(LAPACK_ROW_MAJOR,'U','N',matrixSize,&matrixU_inverse[0],matrixSize);
  LAPACKE_dtrtri(LAPACK_ROW_MAJOR,'L','N',matrixSize,&matrixL_inverse[0],matrixSize);


  /*
    Now I need to pick out values from matrix_U, matrix_L, matrixU_inverse, and matrixL_inverse
      and I want to place them in my vectors matrixL, matrixU, matrixLInverse, and matrixUInverse

    Remember that I need to cyclically traverse the matrix to put the right values in the right places
  
    Basically I am converting these temporary 1d vectors into our 2d member variable vectors

    Remember here that I am picking out the data that belongs to the local process
  */

  int pIndex = this->gridCoords[0]*this->processorGridDimSize+this->gridCoords[1];
  for (int i=dimXstart; i<dimXend; i++)
  {
    for (int j=dimYstart; j<dimYend; j++)
    {
      // I need to use dimXstart and dimXend and dimYstart and dimYend ...
      int index = (i-dimXstart)*this->processorGridDimSize*matrixSize+(pIndex/this->processorGridDimSize)*matrixSize + (j-dimYstart)*this->processorGridDimSize+(pIndex%this->processorGridDimSize);
      this->matrixU[i][j] = matrix_U[index];
      this->matrixL[i][j] = matrix_L[index];
      this->matrixUInverse[i][j] = matrixU_inverse[index];
      this->matrixLInverse[i][j] = matrixL_inverse[index];
    }
  }
}

