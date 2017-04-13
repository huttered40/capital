//#include "solver.h" -> Not done here because of template issues

/*
  Turn on debugging statements when necessary by flipping the 0 to 1
*/

//#include "./../cholesky/cholesky.h"

#define DEBUGGING_QR 0
#define INFO_OUTPUT 0
#define PROCESSOR_X_ 0
#define PROCESSOR_Y_ 0
#define PROCESSOR_Z_ 0

#include "./../cholesky/cholesky.h"				// Is this the best place for it?

template<typename T>
qr<T>::qr(uint32_t rank, uint32_t size, int argc, char **argv, MPI_Comm comm)
{
  this->worldComm = comm;
  this->worldRank = rank;
  this->worldSize = size;
  this->nDims = 3;			// Might want to make this a parameter of argv later, especially with QR and tuning parameter c
  this->matrixRowSize = atoi(argv[1]);		// N
  this->matrixColSize = atoi(argv[2]);		// k
  this->processorGridDimTune = atoi(argv[3]);		// For now, this could be c=1 or c=P^{1/3}
  this->processorGridDimReact = this->worldSize/(this->processorGridDimTune*this->processorGridDimTune);	// 64-bit trick here?
  this->localRowSize = this->matrixRowSize/this->processorGridDimReact;
  this->localColSize = this->matrixColSize/this->processorGridDimTune;
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

  this->constructGridQR();					// Question: Do I add Matrix Multiplication code in here, or do add another case in the Cholesky
								//	MatMul? Or do I separate out those 4 MM cases into separate private methods?
								//	I would like to have all Matrix Multiplication methods in one place, but I need
								// 	to make sure that the processor grid is set up in certain way for switching from
								// 	QR code to Cholesky code.
//  this->distributeData(true);


  #if INFO_OUTPUT
  if (this->worldRank == 0)
  {
    std::cout << "Program - QR\n"; 
    std::cout << "Size of matrix ->                                                 " << this->matrixRowSize << " x " << this->matrixColSize << std::endl;
    std::cout << "Size of MPI_COMM_WORLD ->                                         " << this->worldSize << std::endl;
    std::cout << "Rank of my processor in MPI_COMM_WORLD ->                         " << this->worldRank << std::endl;
    std::cout << "Number of dimensions of processor grid ->                         " << 3 << std::endl;
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
void qr<T>::constructGridQR(void)
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
void qr<T>::distributeData(std::vector<T> &matA)
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

  // Note: above, I am not sure if the vector can hold more than 2^32 elements. Maybe this isnt too big of a deal, but ...

  // Only distribute the data sequentially on the first layer. Then we broadcast down the 3D processor cube using the depth communicator


/*
  if (this->gridCoords[2] == 0)
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
*/


  uint32_t start = this->worldRank*this->localRowSize;      // start acts as the offset into each processor's local data too
  uint32_t end = std::min(start+this->localRowSize, this->localColSize);

  // Allocate the diagonals
  for (uint32_t i=start; i<end; i++)
  {
    uint64_t seed = i*this->matrixColSize;
    seed += i;
    srand48(seed);
    matA[seed-start*this->matrixColSize] = drand48() + this->matrixRowSize;                  // Diagonals should be greater than the sum of its column elements for stability reasons
  }

  // Allocate the strictly lower triangular part
  for (uint32_t i=0; i<this->localRowSize; i++)
  {
    uint64_t temp = (i+start)*this->matrixColSize;					// 64-bit expansion trick here? Do later
    uint32_t iterMax = std::min(i+start,this->localColSize);			// remember that matrixColSize == localColSize when c==1, but not otherwise
    for (uint32_t j=0; j<iterMax; j++)
    {
      uint64_t seed = temp+j;						// Again, this should work for c==1, but many changes for c==P^{1/3}
      srand48(seed);
      matA[i*this->localColSize+j] = drand48();
    }
  }

  // Allocate the strictly upper triangular part
  for (uint32_t i=0; i<this->localRowSize; i++)
  {
    uint64_t temp = (i+start)*this->matrixColSize;					// 64-bit expansion trick here? Do later
    uint32_t iterMax = std::min(i+start,this->localColSize);			// remember that matrixColSize == localColSize when c==1, but not otherwise
    for (uint32_t j=iterMax+1; j<this->localColSize; j++)
    {
      uint64_t seed = temp+j;						// Again, this should work for c==1, but many changes for c==P^{1/3}
      srand48(seed);
      matA[i*this->localColSize+j] = drand48();
    }
  }
}

/*
  The solve method will initiate the solving of this Cholesky Factorization algorithm
*/
template<typename T>
void qr<T>::qrSolve(std::vector<T> &mat1, std::vector<T> &mat2, std::vector<T> &mat3, bool isData)
{
  // use resize or reserve here for matrixA, matrixL, matrixLI to make sure that enough memory is used. Might be a better way to do this later
  mat1.resize(this->localRowSize*this->localColSize);		// Makes sure that the user's data structures are of proper size.
  mat2.resize(mat1.size(),0.);
  mat3.resize(this->localColSize*this->localColSize,0.);    // k*k for now, each processor owns this
  //allocateLayers();				// Should I pass in a pointer here? I dont think so since its only for this->matrixA
  //this->matrixA[0] = std::move(matA);		// because this->matrixA holds layers
  //this->matrixL = std::move(matL);
  //this->matrixLInverse = std::move(matLI);

  if (!isData)	// so from QR, isData will be true, but for a true Cholesky, isData = false
  {
    this->distributeData(mat1);		// for now, must pass in matA since I am not using member vectors
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

  std::vector<T> tempR1(mat3.size(),0.);
  std::vector<T> tempR2(mat3.size(),0.);
  std::vector<T> tempQ(mat2.size(), 0.);		// Try to think of a way to get the memory footprint down here. Can I re-use anything?
  choleskyQR(mat1,tempQ,tempR1);
  choleskyQR(tempQ,mat2,tempR2);
  
  // One more multiplication step, mat3 = tempR2*tempR1, via MM for now, may be able to exploit some structure in it for cheaper?
  cblas_dgemm(CblasRowMajor, CblasTrans, CblasTrans, this->localColSize, this->localColSize,
    this->localColSize, 1., &tempR2[0], this->localColSize, &tempR1[0], this->localColSize, 0., &mat3[0], this->localColSize);
}

/*
  Write function description
*/
template<typename T>
void qr<T>::choleskyQR(std::vector<T> &matrix1, std::vector<T> &matrix2, std::vector<T> &matrix3)
{
  // do the A^{T}*A matrix multiplication
  // call the cholesky
  // Perform the TRSM - Q = A*R^{-1}
  // Remember that for now, I am only doing the c==1 version. This will have to be adjusted substantially to account for c>1

  // for c==1 case, call a lapack syrk, an allreduce, another multiplication, etc..
  // Note that ssyrk writes to either the upper triangular part or the lower triangular part, since the resulting matrix is diagonal.
  std::vector<T> tempC(matrix3.size());		// will be filled up by the SYRK and will be fed into cholesky at matrix A
  std::vector<T> recvC(tempC.size(),0.);
  std::vector<T> tempInverse(matrix3.size());		// is this size right with the cholesky? Doesnt cholesky take triangular size??????
  cblas_dsyrk(CblasRowMajor, CblasLower, CblasTrans, this->localColSize, this->localRowSize, 1., &matrix1[0], this->localColSize, 0, &tempC[0], this->localColSize);
  MPI_Allreduce(&tempC[0], &recvC[0], tempC.size(), MPI_DOUBLE, MPI_SUM, this->worldComm);
  MPI_Comm tempComm;							// change name later
  MPI_Comm_split(MPI_COMM_WORLD, this->worldRank, this->worldRank, &tempComm);
  cholesky<double> myCholesky(0, 1, 3, this->localColSize, tempComm);		// Note that each processor that calls thinks its the only one
  myCholesky.choleskySolve(recvC, matrix3, tempInverse, true);
  expandMatrix(tempInverse,this->localColSize);
  cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, this->localRowSize, this->localColSize,
    this->localColSize, 1., &matrix1[0], this->localRowSize, &tempInverse[0], this->localColSize, 0., &matrix2[0], this->localRowSize);
}


/*
  This function was taken from cholesky code and needs to be completely changed

  m rows, n cols for whoever calls this
*/
template<typename T>
void qr<T>::qrLAPack(std::vector<T> &data, std::vector<T> &dataQ, std::vector<T> &dataR, uint32_t m, uint32_t n, bool needData)
{
  
  // hold on. Why does this need to distribute the data when we can just use my distributedataCyclic function?

  if (needData)
  {
    // Allocate the diagonals
    uint32_t endDiag = std::min(m,n);
    for (uint32_t i=0; i<endDiag; i++)
    {
      uint64_t seed = i*n;
      seed += i;
      srand48(seed);
      data[seed] = drand48() + std::max(m,n);                  // Diagonals should be greater than the sum of its column elements for stability reasons
    }

    // Allocate the strictly lower triangular part
    for (uint32_t i=0; i<m; i++)
    {
      uint64_t temp = i*n;					// 64-bit expansion trick here? Do later
      uint32_t iterMax = std::min(i,n);			// remember that matrixColSize == localColSize when c==1, but not otherwise
      for (uint32_t j=0; j<iterMax; j++)
      {
        uint64_t seed = temp+j;						// Again, this should work for c==1, but many changes for c==P^{1/3}
        srand48(seed);
        data[seed] = drand48();
      }
    }

    // Allocate the strictly upper triangular part
    for (uint32_t i=0; i<m; i++)
    {
      uint64_t temp = i*n;					// 64-bit expansion trick here? Do later
      for (uint32_t j=i+1; j<n; j++)
      {
        uint64_t seed = temp+j;						// Again, this should work for c==1, but many changes for c==P^{1/3}
        srand48(seed);
        data[seed] = drand48();
      }
    }
  }

  std::vector<T> helperQR(this->matrixColSize, 0.);		// size - min(m,n)
  dataQ = data;
  LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, this->matrixRowSize, this->matrixColSize, &dataQ[0], this->matrixColSize, &helperQR[0]);

  // Now to get the actual matrix Q back into row-order form, we need to call another LAPACKE routine
  // Note: dataQ that was passed in MUST be of size m*n
  LAPACKE_dorgqr(LAPACK_ROW_MAJOR, this->matrixRowSize, this->matrixColSize, this->matrixColSize, &dataQ[0], this->matrixColSize, &helperQR[0]);  

  return;
}

/*
  Note: I am just trying to get this to work in the c==1 case. I can generalize it to any c later.

  Also, I can't seem to find how to get matrix R, so I will not check for that right now
*/
template<typename T>
void qr<T>::getResidual(std::vector<T> &matA, std::vector<T> &matQ, std::vector<T> &matR)
{
  /*
	We want to perform a reduction on the data on one of the P^{1/3} layers, then call lapackTest with a single
	processor. Then we can, in the right order, compare these solutions for correctness.
  */


  // Initialize
  this->matrixQNorm = 0.;
  this->matrixANorm = 0.;
  this->matrixRNorm = 0.;

  if (this->worldRank != 0)		// Not layer anymore, lets use a root of 0 for the MPI_Gather
  {
    MPI_Gather(&matA[0],matA.size(), MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, this->worldComm);		// use 0 as root rank, as it needs to be the same for all calls
    MPI_Gather(&matQ[0],matQ.size(), MPI_DOUBLE, NULL, 0, MPI_DOUBLE, 0, this->worldComm);		// use 0 as root rank, as it needs to be the same for all calls
  }
  else		// If not root 0, contribute to the two gathers
  {
    //std::vector<T> sendData(this->matrixDimSize * this->matrixDimSize);
    uint64_t recvDataSize = this->matrixRowSize*this->matrixColSize;
    std::vector<T> recvData(recvDataSize);	// only the bottom half, remember?
    MPI_Gather(&matA[0],matA.size(), MPI_DOUBLE, &recvData[0],matA.size(), MPI_DOUBLE, 0, this->worldComm);		// use 0 as root rank, as it needs to be the same for all calls
    std::vector<T> recvDataQ(recvDataSize);	// only the bottom half, remember?
    MPI_Gather(&matQ[0], matQ.size(), MPI_DOUBLE, &recvDataQ[0],matQ.size(), MPI_DOUBLE, 0, this->worldComm);		// use 0 as root rank, as it needs to be the same for all calls


    uint64_t matSize = this->matrixRowSize;
    matSize *= this->matrixColSize;
    std::vector<T> data(matSize, 0.);
    std::vector<T> lapackDataQ(matSize, 0.);
    //std::vector<T> lapackDataR(matSize);						// wont check for R now
    std::vector<T> tempLaPack;
    qrLAPack(data, lapackDataQ, tempLaPack, this->matrixRowSize, this->matrixColSize, true);		// pass this vector in by reference and have it get filled up
    // Now we can start comparing this->matrixL with data
    // Lets just first start out by printing everything separately too the screen to make sure its correct up till here

 
    for (uint64_t i=0; i<data.size(); i++)
    {
      recvDataQ[i] *= (-1);		//I DONT KNOW WHY IM DOING THIS, BUT ALL OF MY Q IS NEGATIEV
      #if DEBUGGING_QR
      std::cout << lapackDataQ[i] << " " << recvDataQ[i] << " " << i << std::endl;
      std::cout << lapackDataQ[i] - recvDataQ[i] << std::endl;
      #endif
      double diff = lapackDataQ[i] - recvDataQ[i];
      if (diff > 1e-12) { std::cout << "Bad - " << i << ", diff - " << diff << " lapack - " << lapackDataQ[i] << " and real data - " << recvDataQ[i] << std::endl; }
      //double diff = lapackData[index++] - recvDataL[PE*this->matrixL.size() + pCounters[PE]++];
      //diff = abs(diff);
      this->matrixQNorm += (diff*diff);
    }
    this->matrixQNorm = sqrt(this->matrixQNorm);


    for (uint64_t i=0; i<lapackDataQ.size(); i++)
    {
      #if DEBUGGING_QR
      std::cout << data[i] << " " << recvData[i] << " " << std::endl;
      std::cout << data[i] - recvData[i] << std::endl;
      #endif
      double diff = data[i] - recvData[i];
      if (diff > 1e-12) { std::cout << "Bad - " << i << ", diff - " << diff << " lapack - " << data[i] << " and real data - " << recvData[i] << std::endl; }
      this->matrixANorm += (diff*diff);
    }  
    this->matrixANorm = sqrt(this->matrixANorm);
/*
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
*/      
    std::cout << "matrix A Norm - " << this->matrixANorm << std::endl;
    std::cout << "matrix Q Norm - " << this->matrixQNorm << std::endl;
    std::cout << "matrix R Norm - " << this->matrixRNorm << std::endl;
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

/*
  Might be a more efficient way to do this later
*/
template<typename T>
void qr<T>::expandMatrix(std::vector<T> &data, uint32_t n)
{
  data.resize(n*n);					// change from lower triangular to square with half zeros
  std::vector<T> temp(n*n,0.);
  uint64_t tracker = 0;
  for (uint32_t i=0; i<n; i++)
  {
    uint64_t index1 = i*n;
    for (uint32_t j=0; j<=i; j++)
    {
      temp[index1+j] = data[tracker++];
    }
  }

  data = std::move(temp);				// No copy from temp buffer necessary here, just change the internal pointer via a move operation
}