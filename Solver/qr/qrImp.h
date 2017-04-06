//#include "solver.h" -> Not done here because of template issues

/*
  Turn on debugging statements when necessary by flipping the 0 to 1
*/
#define DEBUGGING 0
#define INFO_OUTPUT 1
#define PROCESSOR_X_ 0
#define PROCESSOR_Y_ 0
#define PROCESSOR_Z_ 0

template<typename T>
qr<T>::qr(uint32_t rank, uint32_t size, int argc, char **argv)
{
  this->worldRank = rank;
  this->worldSize = size;
  this->nDims = 3;			// Might want to make this a parameter of argv later, especially with QR and tuning parameter c
  this->matrixSizeRow = atoi(argv[1]);		// N
  this->matrixSizeCol = atoi(argv[2]);		// k
  this->processorGridDimTune = atoi(argv[3]);
  this->processorGridDimReact = this->worldSize/(this->processorGridDimTune*this->processorGridDimTune);	// 64-bit trick here?
  this->localRowSize = this->matrixSizeRow/this->processorGridDimReact; // d==P for now
  this->localColSize = this->matrixSizeCol/this->processorGridDimTune;  // c==1 for now
  this->argc = argc;
  this->argv = argv;
  this->matrixANorm = 0.;
  this->matrixQNorm = 0.;
  this->matrixRNorm = 0.;

/*
  Precompute a list of cubes for quick lookUp of cube dimensions based on processor size
  Maps number of processors involved in the computation to its cubic root to avoid expensive cubic root routine
  Table should reside in the L1 cache for quick lookUp, but only needed once
*/
  // This might need to be redone for the tuning parameter c, but leave it for now.
  for (uint32_t i=1; i<500; i++)
  {
    uint64_t size = i;
    size *= i;
    size *= i;
    this->gridSizeLookUp[size] = i;
  }

  this->constructGridCholesky();
//  this->distributeData(true);

  #if INFO_OUTPUT
  if (this->worldRank == 0)
  {
    std::cout << "Program - QR\n"; 
    std::cout << "Size of matrix ->                                                 " << this->matrixDimSize << std::endl;
    std::cout << "Size of MPI_COMM_WORLD ->                                         " << this->worldSize << std::endl;
    std::cout << "Rank of my processor in MPI_COMM_WORLD ->                         " << this->worldRank << std::endl;
    std::cout << "Number of dimensions of processor grid ->                         " << this->nDims << std::endl;
    std::cout << "Number of processors along one dimension of 3-Dimensional grid -> " << this->processorGridDimSize << std::endl;
    std::cout << "Grid coordinates in 3D Processor Grid for my processor ->        (" << this->gridCoords[0] << "," << this->gridCoords[1] << "," << this->gridCoords[2] << ")" << std::endl;
    std::cout << "Size of 2D Layer Communicator ->                                  " << this->layerCommSize << std::endl;
    std::cout << "Rank of my processor in 2D Layer Communicator ->                  " << this->layerCommRank << std::endl;
    std::cout << "Size of Row Communicator ->                                       " << this->colCommSize << std::endl;
    std::cout << "Rank of my processor in Row Communicator ->                       " << this->rowCommRank << std::endl;
    std::cout << "Size of Column Communicator ->                                    " << this->colCommSize << std::endl;
    std::cout << "Rank of my processor in Column Communicator ->                    " << this->colCommRank << std::endl;
    std::cout << "Size of Depth Communicator Communicator ->                        " << this->depthCommSize << std::endl;
    std::cout << "Rank of my processor in Depth Communicator ->                     " << this->depthCommRank << std::endl;
  }
  #endif
}

template <typename T>
void qr<T>::constructGridCholesky(void)
{
  /*
	Look up to see if the number of processors in startup phase is a cubic. If not, then return.
	We can reduce this restriction after we have it working for a perfect cubic processor grid
	If found, this->processorGridDimSize will represent the number of processors along one dimension, such as along a row or column (P^(1/3))
  */

/*
  if (this->gridSizeLookUp.find(this->worldSize) == gridSizeLookUp.end())
  {
    #if DEBUGGING
    std::cout << "Requested number of processors is not valid for a 3-Dimensional processor grid. Program is ending." << std::endl;
    #endif
    return;
  }

  this->processorGridDimSize = this->gridSizeLookUp[this->worldSize];
*/
  
  /*
    this->baseCaseSize gives us a size in which to stop recursing in the CholeskyRecurse method
    = n/P^(2/3). The math behind it is written about in my report and other papers
  */

/*  
  this->gridDims.resize(this->nDims,this->processorGridDimSize);
  this->gridCoords.resize(this->nDims); 
*/
  
  /*
    The 3D Cartesian Communicator is used to distribute the random data in a cyclic fashion
    The other communicators are created for specific communication patterns involved in the algorithm.
  */

/*  
  std::vector<int> boolVec(3,0);
  MPI_Cart_create(MPI_COMM_WORLD, this->nDims, &this->gridDims[0], &boolVec[0], false, &this->grid3D);
  MPI_Comm_rank(this->grid3D, &this->grid3DRank);
  MPI_Comm_size(this->grid3D, &this->grid3DSize);
  MPI_Cart_coords(this->grid3D, this->grid3DRank, this->nDims, &this->gridCoords[0]);
*/
  /*
    Before creating row and column sub-communicators, grid3D must be split into 2D Layer communicators.
  */

/*
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
*/
}

/*
  Cyclic distribution of data on one layer, then broadcasting the data to the other P^{1/3}-1 layers, similar to how Scalapack does it
*/
template <typename T>
void qr<T>::distributeData(void)
{
  /*
    If we think about the Cartesian rank coordinates, then we dont care what layer (in the z-direction) we are in. We care only about the (x,y) coordinates.
    Sublayers exist in a cyclic distribution, so that in the first sublayer of the matrix, all processors are used.
    Therefore, we cycle them from (0,1,2,0,1,2,0,1,2,....) in the x-dimension of the cube (when traveling down from the top-front-left corner) to the bottom-front-left corner.
    Then for each of those, there are the processors in the y-direction as well, which are representing via the nested loop.
  */

  /*
    Note that because matrix A is supposed to be symmetric, I will only give out the lower-triangular portion of A
    Note that this would require a change in my algorithm, because I do reference uper part of A.
    Instead, I will only give values for the upper-portion of A. At a later date, I can change the code to only
    allocate a triangular portion instead of storing half as zeros.
  */

  uint64_t size = this->processorGridDimSize;			// Expensive division? Lots of compute cycles? Worse than pushing back?
  size *= size;
  uint64_t matSize = this->matrixDimSize;
  matSize *= matSize;
  size = matSize/size;						// Watch for overflow, N^2 / P^{2/3} is the initial size of the data that each p owns

  // Note: above, I am not sure if the vector can hold more than 2^32 elements. Maybe this isnt too big of a deal, but ...

  // Only distribute the data sequentially on the first layer. Then we broadcast down the 3D processor cube using the depth communicator
  if (((inParallel) && (this->gridCoords[2] == 0)) || (!inParallel))
  {
    uint64_t counter = 0;
    for (uint32_t i=this->gridCoords[0]; i<this->matrixDimSize; i+=this->processorGridDimSize)
    {
      for (uint32_t j=this->gridCoords[1]; j<this->matrixDimSize; j+=this->processorGridDimSize)
      {
        if (i > j)
        {
          uint64_t seed = i;
          seed *= this->matrixDimSize;
          seed += j;
          srand48(seed);
        }
        else
        {
          uint64_t seed = j;
          seed *= this->matrixDimSize;
          seed += i;
          srand48(seed);
        }
      
        //this->matrixA[this->matrixA.size()-1].push_back((rand()%100)*1./100.);
        this->matrixA[0][counter++] = drand48();
        if (i==j)
        {
          //matrixA[this->matrixA.size()-1][matrixA[this->matrixA.size()-1].size()-1] += 10.;		// All diagonals will be dominant for now.
          this->matrixA[0][counter-1] += this->matrixDimSize;
        }
      }
    }

    if (inParallel)
    {
      MPI_Bcast(&this->matrixA[0][0], size, MPI_DOUBLE, this->depthCommRank, this->depthComm);  
    }
  }
  else			// All other processor not on that 1st layer
  {
    if (inParallel)
    {
      // I am assuming that the root has rank 0 in the depth communicator
      MPI_Bcast(&this->matrixA[0][0], size, MPI_DOUBLE, 0, this->depthComm);
    }
  }

  #if DEBUGGING
  if ((this->gridCoords[0] == PROCESSOR_X_) && (this->gridCoords[1] == PROCESSOR_Y_) && (this->gridCoords[2] == PROCESSOR_Z_))
  {
    std::cout << "About to print what processor (" << this->gridCoords[0] << "," << this->gridCoords[1] << "," << this->gridCoords[2] << ") owns.\n";
    printMatrixSequential(this->matrixA[0], this->localSize, false);
  }
  #endif

}

/*
  The solve method will initiate the solving of this Cholesky Factorization algorithm
*/
template<typename T>
void qr<T>::qrSolve(std::vector<T> &matA, std::vector<T> &matL, std::vector<T> &matLI, bool isData)
{
  // use resize or reserve here for matrixA, matrixL, matrixLI to make sure that enuf memory is used.
  matA.resize(this->localRowSize*this->localColSize);		// Makes sure that the user's data structures are of proper size.
  matQ.resize(matA.size());
  matR.resize(this->localColSize*this->localColSize);    // k*k for now, each processor owns this
  //allocateLayers();				// Should I pass in a pointer here? I dont think so since its only for this->matrixA
  //this->matrixA[0] = std::move(matA);		// because this->matrixA holds layers
  //this->matrixL = std::move(matL);
  //this->matrixLInverse = std::move(matLI);

  if (!isData)	// so from QR, isData will be true, but for a true Cholesky, isData = false
  {
    this->distributeDataCyclic(();
  }
  
  // For now, lets just support either numP == 1 (for say QR c==1, but later we could support a tuning parameter that could do a mix?
//  if (this->worldSize == 1)
//  {
//    qrLAPack(this->matrixA[0], this->matrixL, this->matrixLInverse, this->localSize, false); // localSize == matrixDimSize
    // Might be a more efficient way to do this, but I will cut the size now
//    trimMatrix(this->matrixL, this->localSize);
//    trimMatrix(this->matrixLInverse, this->localSize);		// trimMatrix will cut size from n*n back to n*(n+1)/2
//  }
//  else
//  {
    // Start at 0th layer
//    qrEngine(0,this->localSize,0,this->localSize,this->localSize,this->matrixDimSize, this->localSize, 0);
//  }
  /*
    Now I need to re-validate the arguments so that the user has the data he needs.
    Need to do another std::move into matrixA, matrixL, and matrixLI again.
    Basically I invalidate the input matrices at first, then re-validate them before the solve routine returns
    I can think of it as "sucking" out the data from my member variables back into the user's data structures
  */

//  matA = std::move(this->matrixA[0]);		// get the first layer again
//  matL = std::move(this->matrixL);
//  matLI = std::move(this->matrixLInverse);

  choleskyQR(matA,matQ,matR);

}

/*
  Write function description
*/
template<typename T>
void qr<T>::choleskyQR(std::vector<T> &matrixA, std::vector<T> &matrixQ, std::vector<T> &matrixR)
{
  // do the A^{T}*A matrix multiplication
  // call the cholesky
  // Perform the TRSM - Q = A*R^{-1}
  // Remember that for now, I am only doing the c==1 version. This will have to be adjusted substantially to account for c>1

  // for c==1 case, call a lapack syrk, an allreduce, another multiplication, etc..
}

/*
  Transpose Communication Helper Function
*/
template<typename T>
void qr<T>::fillTranspose(uint32_t dimXstart, uint32_t dimXend, uint32_t dimYstart, uint32_t dimYend, uint32_t matrixWindow, uint32_t dir)
{
  switch (dir)
  {
    case 0:
    {
      // copy L11-inverse data into holdTransposeInverse to be send/received
      this->holdTransposeL.clear();
      uint64_t holdTransposeLSize = matrixWindow;
      holdTransposeLSize *= (matrixWindow+1);
      holdTransposeLSize >>= 1;
      this->holdTransposeL.resize(holdTransposeLSize, 0.);
      uint64_t startOffset = dimXstart;
      startOffset *= (dimXstart+1);
      startOffset >>= 1;
      //int startOffset = ((dimXstart*(dimXstart+1))>>1);
      uint64_t index1 = 0;
      uint64_t index2 = startOffset + dimXstart;
      for (uint32_t i=0; i<matrixWindow; i++)
      {
        for (uint32_t j=0; j<=i; j++)
        {
          this->holdTransposeL[index1++] = this->matrixLInverse[index2++];
        }
        index2 += dimYstart;
      }
      if ((this->gridCoords[0] != this->gridCoords[1]))
      {
        // perform MPI_SendRecv_replace
        int destRank = -1;
        int rankArray[3] = {this->gridCoords[1], this->gridCoords[0], this->gridCoords[2]};
        MPI_Cart_rank(this->grid3D, &rankArray[0], &destRank);
        MPI_Status stat;
        MPI_Sendrecv_replace(&this->holdTransposeL[0], this->holdTransposeL.size(), MPI_DOUBLE,
          destRank, 0, destRank, MPI_ANY_TAG, MPI_COMM_WORLD, &stat);
      }
      break;
    }
    case 1:
    {
      // copy L11-inverse data into holdTransposeInverse to be send/received
      this->holdTransposeL.clear();
      uint64_t holdTransposeLSize = matrixWindow;
      holdTransposeLSize *= matrixWindow;
      this->holdTransposeL.resize(holdTransposeLSize, 0.);
      uint64_t startOffset = dimXstart;
      startOffset *= (dimXstart+1);
      startOffset >>= 1;
      //int startOffset = ((dimXstart*(dimXstart+1))>>1);
      uint64_t index1 = 0;
      uint64_t index2 = startOffset + dimYstart;
      for (uint32_t i=0; i<matrixWindow; i++)
      {
        for (uint32_t j=0; j<matrixWindow; j++)
        {
          this->holdTransposeL[index1++] = this->matrixL[index2++];
        }
        uint64_t temp = dimYstart;
        temp += i;
        temp++;
        index2 += temp;
      }

      if ((this->gridCoords[0] != this->gridCoords[1]))
      {
        // perform MPI_SendRecv_replace
        MPI_Status stat;
        int destRank = -1;
        int rankArray[3] = {this->gridCoords[1], this->gridCoords[0], this->gridCoords[2]};
        MPI_Cart_rank(this->grid3D, &rankArray[0], &destRank);
        MPI_Sendrecv_replace(&this->holdTransposeL[0], this->holdTransposeL.size(), MPI_DOUBLE,
          destRank, 0, destRank, 0, MPI_COMM_WORLD, &stat);

      }    
      break;
    }
  }
}


/*
  Might be a better way to do this, as in using only a giant 1D vector to hold all layers, but this should be good for now
  Or possibely allocating a huge 1-D vector and then indexing into it using that 2D. Hmmmmm
*/
template<typename T>
void qr<T>::allocateLayers(void)
{
  // Fill up the 2D matrixA so that no push_backs are needed that would cause massive reallocations when size >= capacity
  // This is a one-time call so expensive divides are ok
  this->matrixA.resize(this->processorGridDimSize*this->processorGridDimSize);	// changes capacity once so that it is never changed again
  // We can skip the allocation of the 1st layer because it would be wasted to do move operation
  // Or maybe I should??? What if not enough space is allocated for it and it causes a giant re-allocation of memory behind the scenes?
  // So I will leave as starting at i=0 for now, but could change it to i=1 later if that helps
  uint64_t matSize = this->localSize;			// Will have been given correct value before this
  matSize *= matSize;
  
// Try this, can erase if there are problems
/*
  uint64_t fullSize = matSize;
  fullSize <<= 2;
  fullSize -= (4*(matSize >> (2*this->processorGridDimSize*this->processorGridDimSize)));
  fullSize /= 3;
  this->matrixA[0] = new T[fullSize];		// new feature. Must do this instead of using resize. This is the only way I can see to do what I want using vectors.
  uint64_t holdSize = matSize;
  for (uint32_t i=1; i<this->matrixA.size(); i++)
  {
    //this->matrixA[i].resize(matSize);
    this->matrixA[i] = &this->matrixA[0][holdSize];
    matSize >>= 2;					// divide by 4
    holdSize += matSize;
  }
*/
  for (uint32_t i=0; i<this->matrixA.size(); i++)
  {
    this->matrixA[i].resize(matSize);
    matSize >>= 2;					// divide by 4
  }
}

template<typename T>
void qr<T>::trimMatrix(std::vector<T> &data, uint32_t n)
{
  // Use overwriting trick
  uint64_t tracker = 0;
  for (uint32_t i=0; i<n; i++)
  {
    uint64_t index1 = i*n;
    for (uint32_t j=0; j<=i; j++)
    {
      data[tracker++] = data[index1+j];
    }
  }

  data.resize(tracker);		// this should cut off the garbage pieces and leave a n*(n+1)/2 size matrix
}

/*
  This function was taken from cholesky code and needs to be completely changed
*/
template<typename T>
void qr<T>::qrLAPack(std::vector<T> &data, std::vector<T> &dataL, std::vector<T> &dataInverse, uint32_t n, bool needData)
{
  
  // hold on. Why does this need to distribute the data when we can just use my distributedataCyclic function?

  if (needData)
  {
    for (uint32_t i=0; i<n; i++)
    {
      for (uint32_t j=0; j<n; j++)
      {
        if (i > j)
        {
          uint64_t seed = i;
          seed *= n;
          seed += j;
          srand48(seed);
        }
        else
        {
          uint64_t seed = j;
          seed *= n;
          seed += i;
          srand48(seed);
        }
        uint64_t seed = i;
        seed *= n;
        seed += j;
        data[seed] = drand48();
        //std::cout << "hoogie - " << i*n+j << " " << data[i*n+j] << std::endl;
        if (i==j)
        {
          data[seed] += this->matrixDimSize;
        }
      }
    }
  }

  dataL = data;			// big copy. Cant use a std::move operation here because we actually need data, we can't suck out its data
  LAPACKE_dpotrf(LAPACK_ROW_MAJOR,'L',n,&dataL[0],n);
  dataInverse = dataL;				// expensive copy
  LAPACKE_dtrtri(LAPACK_ROW_MAJOR,'L','N',n,&dataInverse[0],n);

  #if DEBUGGING
  std::cout << "Cholesky Solution is below *************************************************************************************************************\n";

  for (uint32_t i=0; i<n; i++)
  {
    for (uint32_t j=0; j<n; j++)
    {
      uint64_t index = i;
      index *= n;
      index += j;
      std::cout << dataL[index] << " ";
    }
    std::cout << "\n";
  }
  #endif

  return;
}

template<typename T>
void qr<T>::getResidualLayer(std::vector<T> &matA, std::vector<T> &matL, std::vector<T> &matLI)
{
  /*
	We want to perform a reduction on the data on one of the P^{1/3} layers, then call lapackTest with a single
	processor. Then we can, in the right order, compare these solutions for correctness.
  */

  // Initialize
  this->matrixLNorm = 0.;
  this->matrixANorm = 0.;
  this->matrixLInverseNorm = 0.;

  if (this->gridCoords[2] == 0)		// 1st layer
  {
    //std::vector<T> sendData(this->matrixDimSize * this->matrixDimSize);
    uint64_t recvDataSize = this->processorGridDimSize;
    recvDataSize *= recvDataSize;
    recvDataSize *= matA.size();
    std::vector<T> recvData(recvDataSize);	// only the bottom half, remember?
    MPI_Gather(&matA[0],matA.size(), MPI_DOUBLE, &recvData[0],matA.size(), MPI_DOUBLE, 0, this->layerComm);		// use 0 as root rank, as it needs to be the same for all calls
    std::vector<T> recvDataL(this->processorGridDimSize*this->processorGridDimSize*matL.size());	// only the bottom half, remember?
    MPI_Gather(&matL[0],matL.size(), MPI_DOUBLE, &recvDataL[0],matL.size(), MPI_DOUBLE, 0, this->layerComm);		// use 0 as root rank, as it needs to be the same for all calls
    std::vector<T> recvDataLInverse(this->processorGridDimSize*this->processorGridDimSize*matLI.size());	// only the bottom half, remember?
    MPI_Gather(&matLI[0],matLI.size(), MPI_DOUBLE, &recvDataLInverse[0],matLI.size(), MPI_DOUBLE, 0, this->layerComm);		// use 0 as root rank, as it needs to be the same for all calls

    if (this->layerCommRank == 0)
    {
      /*
	Now on this specific rank, we currently have all the algorithm data that we need to compare, but now we must call the sequential
        lapackTest method in order to get what we want to compare the Gathered data against.
      */

      uint64_t matSize = this->matrixDimSize;
      matSize *= matSize;
      std::vector<T> data(matSize);
      std::vector<T> lapackData(matSize);
      std::vector<T> lapackDataInverse(matSize);
      qrLAPack(data, lapackData, lapackDataInverse, this->matrixDimSize, true);		// pass this vector in by reference and have it get filled up

      // Now we can start comparing this->matrixL with data
      // Lets just first start out by printing everything separately too the screen to make sure its correct up till here
      
      {
        uint64_t index = 0;
        uint64_t pCounterSize = this->processorGridDimSize;
        pCounterSize *= pCounterSize;
        std::vector<int> pCounters(pCounterSize,0);		// start each off at zero
        for (uint32_t i=0; i<this->matrixDimSize; i++)
        {
          #if DEBUGGING
          std::cout << "Row - " << i << std::endl;
          #endif
          for (uint32_t j=0; j<=i; j++)
	  {
            uint64_t PE = this->processorGridDimSize;
            PE *= (i%this->processorGridDimSize);
            PE += (j%this->processorGridDimSize);
            //int PE = (j%this->processorGridDimSize) + (i%this->processorGridDimSize)*this->processorGridDimSize;
            uint64_t recvDataIndex = PE;
            recvDataIndex *= matL.size();
            recvDataIndex += pCounters[PE];
	    #if DEBUGGING
            std::cout << lapackData[index] << " " << recvDataL[recvDataIndex] << " " << index << std::endl;
	    std::cout << lapackData[index] - recvDataL[recvDataIndex] << std::endl;
            #endif
            pCounters[PE]++;
            double diff = lapackData[index++] - recvDataL[recvDataIndex];
            if (diff > 1e-12) { std::cout << "Bad - " << i << "," << j << " , diff - " << diff << " lapack - " << lapackData[index-1] << " and real data - " << recvDataL[recvDataIndex] << " index1 - " << index-1 << " index2 - " << recvDataIndex << std::endl; }
            //double diff = lapackData[index++] - recvDataL[PE*this->matrixL.size() + pCounters[PE]++];
            //diff = abs(diff);
            this->matrixLNorm += (diff*diff);
          }
          if (i%2==0)
          {
            pCounters[1]++;		// this is a serious edge case due to the way I handled the actual code
          }
          
          index = this->matrixDimSize;
          index *= (i+1);			// try this. We skip the non lower triangular elements
          #if DEBUGGING
          std::cout << std::endl;
          #endif
        }
        this->matrixLNorm = sqrt(this->matrixLNorm);
      }
      {
        uint64_t index = 0;
        uint64_t pCounterSize = this->processorGridDimSize;
        pCounterSize *= pCounterSize;
        std::vector<int> pCounters(pCounterSize,0);		// start each off at zero
        std::cout << "\n";
        for (uint32_t i=0; i<this->matrixDimSize; i++)
        {
          #if DEBUGGING
          std::cout << "Row - " << i << std::endl;
          #endif
          for (uint32_t j=0; j<this->matrixDimSize; j++)
	  {
            uint64_t PE = this->processorGridDimSize;
            PE *= (i%this->processorGridDimSize);
            PE += (j%this->processorGridDimSize);
	    uint64_t recvDataIndex = PE;
            recvDataIndex *= matA.size();
            recvDataIndex += pCounters[PE];
            #if DEBUGGING
            std::cout << data[index] << " " << recvData[recvDataIndex] << " " << index << std::endl;
	    std::cout << data[index] - recvData[recvDataIndex] << std::endl;
            #endif
            pCounters[PE]++;
            double diff = data[index++] - recvData[recvDataIndex];
            if (diff > 1e-12) { std::cout << "Bad - " << i << "," << j << " , diff - " << diff << " lapack - " << data[index-1] << " and real data - " << recvData[recvDataIndex] << std::endl; }
            this->matrixANorm += (diff*diff);
          }
//          if (i%2==0)
//          {
//            pCounters[1]++;		// this is a serious edge case due to the way I handled the actual code
//          }
          index = this->matrixDimSize;
          index *= (i+1);			// try this. We skip the non lower triangular elements
          #if DEBUGGING
          std::cout << std::endl;
          #endif
        }
        this->matrixANorm = sqrt(this->matrixANorm);
      }
      {
        uint64_t index = 0;
        uint64_t pCounterSize = this->processorGridDimSize;
        pCounterSize *= pCounterSize;
        std::vector<int> pCounters(pCounterSize,0);		// start each off at zero
        for (uint32_t i=0; i<this->matrixDimSize; i++)
        {
          #if DEBUGGING
          std::cout << "Row - " << i << std::endl;
          #endif
          for (uint32_t j=0; j<=i; j++)
	  {
            uint64_t PE = this->processorGridDimSize;
            PE *= (i%this->processorGridDimSize);
            PE += (j%this->processorGridDimSize);
            //int PE = (j%this->processorGridDimSize) + (i%this->processorGridDimSize)*this->processorGridDimSize;
            uint64_t recvDataIndex = PE;
            recvDataIndex *= matLI.size();
            recvDataIndex += pCounters[PE];
            #if DEBUGGING
	    std::cout << lapackDataInverse[index] << " " << recvDataLInverse[recvDataIndex] << " " << index << std::endl;
	    std::cout << lapackDataInverse[index] - recvDataLInverse[recvDataIndex] << std::endl;
            #endif
            pCounters[PE]++;
            double diff = lapackDataInverse[index++] - recvDataLInverse[recvDataIndex];
            if (diff > 1e-12) { std::cout << "Bad - " << i << "," << j << " , diff - " << diff << " lapack - " << lapackDataInverse[index-1] << " and real data - " << recvDataLInverse[recvDataIndex] << std::endl; }
            this->matrixLInverseNorm += (diff*diff);
          }
          if (i%2==0)
          {
            pCounters[1]++;		// this is a serious edge case due to the way I handled the code
          }
          index = this->matrixDimSize;
          index *= (i+1);			// try this. We skip the non lower triangular elements
          #if DEBUGGING
          std::cout << std::endl;
          #endif
        }
        this->matrixLInverseNorm = sqrt(this->matrixLInverseNorm);
      }
      
      std::cout << "matrix A Norm - " << this->matrixANorm << std::endl;
      std::cout << "matrix L Norm - " << this->matrixLNorm << std::endl;
      std::cout << "matrix L Inverse Norm - " << this->matrixLInverseNorm << std::endl;
    }
  }
}


template <typename T>
void qr<T>::getResidualParallel()
{
  // Waiting on scalapack support. No other way to this
}

/*
  printMatrixSequential assumes that it is only called by a single processor
  Note: matrix could be triangular or square. So I should add a new function or something to print a triangular matrix without segfaulting
*/
template<typename T>
void qr<T>::printMatrixSequential(std::vector<T> &matrix, uint32_t n, bool isTriangle)
{
  if (isTriangle)
  {
    uint64_t tracker = 0;
    for (uint32_t i=0; i<n; i++)
    {
      for (uint32_t j=0; j<n; j++)
      {
        if (i >= j)
        {
          std::cout << matrix[tracker++] << " ";
        }
        else
        {
          std::cout << 0 << " ";
        }
      }
      std::cout << "\n";
    }
  }
  else  // square
  {
    uint64_t tracker = 0;
    for (uint32_t i=0; i<n; i++)
    {
      for (uint32_t j=0; j<n; j++)
      {
        std::cout << matrix[tracker++] << " ";
      }
      std::cout << "\n";
    }

  }
}

/*
  printMatrixParallel needs to be fixed. Want to assume its being called by P processors or something
*/
template<typename T>
void qr<T>::printMatrixParallel(std::vector<T> &matrix, uint32_t n)
{
  if (this->gridCoords[2] == 0)		// 1st layer
  {
    //std::vector<T> sendData(this->matrixDimSize * this->matrixDimSize);
    std::vector<T> recvData(this->processorGridDimSize*this->processorGridDimSize*this->matrixA[this->matrixA.size()-1].size());	// only the bottom half, remember?
    MPI_Gather(&this->matrixA[this->matrixA.size()-1][0],this->matrixA[this->matrixA.size()-1].size(), MPI_DOUBLE, &recvData[0],this->matrixA[this->matrixA.size()-1].size(), MPI_DOUBLE,
	0, this->layerComm);		// use 0 as root rank, as it needs to be the same for all calls

    if (this->layerCommRank == 0)
    {
      /*
	Now on this specific rank, we currently have all the algorithm data that we need to compare, but now we must call the sequential
        lapackTest method in order to get what we want to compare the Gathered data against.
      */
      
      for (int i=0; i<recvData.size(); i++)
      {
        if (i%this->matrixDimSize == 0)
        { std::cout << "\n"; }
        std::cout << recvData[i] << " ";
      }

      std::cout << "\n\n";

      std::vector<int> pCounters(this->processorGridDimSize*this->processorGridDimSize,0);		// start each off at zero
      for (int i=0; i<this->matrixDimSize; i++)
      {
        for (int j=0; j < this->matrixDimSize; j++)
	{
          int PE = (j%this->processorGridDimSize) + (i%this->processorGridDimSize)*this->processorGridDimSize;
          std::cout << recvData[PE*this->matrixA[this->matrixA.size()-1].size() + pCounters[PE]++] << " ";
        }
//        if (i%2==0)
//        {
//          pCounters[1]++;		// this is a serious edge case due to the way I handled the actual code
//        }
        std::cout << std::endl;
      }
    }
  }
}
