// Need this to use fortran scalapack function
extern "C"
{
  void pdpotrf_(char*, int *m, double *A, int *iA, int *jA, int *desca, int *info);
  void Cblacs_get(int, int, int*);
  void Cblacs_gridinit(int *ConTxt, char *order, int nprow, int npcol);
}

int log2(int n)
{
  return log(n)/log(2);		// shoudl return a cut-off integer
}


#define NUM_ITER 5

//proper modulus for 'a' in the range of [-b inf]
#define WRAP(a,b)       ((a + b)%b)
#define MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )
/*
void Cblacs_pinfo(int*,int*);

void Cblacs_get(int,int,int*);

int Cblacs_gridinit(int*,char*,int,int);
*/
/*
void descinit(int *,  int *,
                 int *,  int *,
                 int *,  int *,
                 int *,  int *,
                 int *, int *);

static void cdesc_init(int * desc, 
                       int m,       int n,
                       int mb,      int nb,
                       int irsrc,   int icsrc,
                       int ictxt,   int LLD,
                                        int * info){
  descinit(desc,&m,&n,&mb,&nb,&irsrc,&icsrc,
             &ictxt, &LLD, info);
}
*/
/*
static void  pdotrf(char *,     int *,
                     double *,  int *,
                     int *,     int *,
                     int *);

static void cpdgetrf(char ch,     int n,
                     double *A, int ia,
                     int ja,    int * desca,
                     int * info){
  pdgetrf(&ch,&n,A,&ia,&ja,desca,info);
}
*/

template<typename T>
void cholesky<T>::choleskyScalapack()
{
/*
  int myRank, numPes;

  MPI_Init(&argc, &argv);
  MPI_Comm_size(MPI_COMM_WORLD, &numPes);
  MPI_Comm_rank(MPI_COMM_WORLD, &myRank);
*/

  std::cout << "Hello\n";
  int numPes = this->worldSize;
  int myRank = this->worldRank;

  int log_numPes = log2(numPes);
  printf("log number of numPes - %d\n",log_numPes);

  return;
  if (argc < 4 || argc > 5) {
    if (myRank == 0) 
      printf("%s [log2_mat_dim] [log2_pe_mat_lda] [log2_blk_dim] [number of iterations]\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  int log_matrixDim = atoi(argv[1]);
  int log_blockDim = atoi(argv[2]);
  int log_sbDim = atoi(argv[3]);
  int matrixDim = 1<<log_matrixDim;
  int blockDim = 1<<log_blockDim;
  int sbDim = 1<<log_sbDim;

  int num_iter;
  if (argc > 4) num_iter = atoi(argv[4]);
  else num_iter = NUM_ITER;

  if (myRank == 0){ 
    printf("PDGETRFOF SQUARE MATRIX\n");
    printf("MATRIX DIMENSION IS %d\n", matrixDim);
    printf("BLOCK DIMENSION IS %d\n", sbDim);
    printf("PERFORMING %d ITERATIONS\n", num_iter);
#ifdef RAND
    printf("WITH RANDOM DATA\n");
#else
    printf("WITH DATA=INDEX\n");
#endif
  }

  if (matrixDim < blockDim || matrixDim % blockDim != 0) {
    if (myRank == 0) printf("array_size_X \% block_size_X != 0!\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }
  if (matrixDim < blockDim || matrixDim % blockDim != 0) {
    if (myRank == 0) printf("array_size_Y \% block_size_Y != 0!\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  int log_num_blocks_dim = log_matrixDim - log_blockDim;
  int num_blocks_dim = 1<<log_num_blocks_dim;

  if (myRank == 0){
    printf("NUM X BLOCKS IS %d\n", num_blocks_dim);
    printf("NUM Y BLOCKS IS %d\n", num_blocks_dim);
  }

  printf("numPes - %d, num_blocks_dim - %d, num_blocks_dim - %d\n", numPes, num_blocks_dim,num_blocks_dim);
  if (num_blocks_dim*num_blocks_dim != numPes){
    if (myRank == 0) printf("NUMBER OF BLOCKS MUST BE EQUAL TO NUMBER OF PROCESSORS\n");
    MPI_Abort(MPI_COMM_WORLD, -1);
  }

  int  myRow = myRank / num_blocks_dim;
  int  myCol = myRank % num_blocks_dim;
  int iter, i, j, k, l, blk_row, blk_col, blk_row_offset, write_offset, offset;

  double * mat_A = (double*)malloc(blockDim*blockDim*sizeof(double));
  int * mat_P = (int*)malloc(2*matrixDim*sizeof(int));
  
  double * temp;
  double ans_verify;


  int icontxt, info;
  int iam, inprocs;
  char order = 'R';

  //Cblacs_pinfo(&iam,&inprocs);
  Cblacs_get(0, 0, &icontxt);	// if doesnt work, try first argument being -1
  Cblacs_gridinit(&icontxt, &order, num_blocks_dim, num_blocks_dim);

  int desc_a[9];
  desc_a[2] = matrixDim;
  desc_a[3] = matrixDim;
  desc_a[4] = sbDim;
  desc_a[5] = sbDim;
  desc_a[6] = 0;
  desc_a[7] = 0;
  desc_a[8] = blockDim;
  desc_a[0] = 1;			// ?
  desc_a[1] = icontxt;
/*
  cdesc_init(desc_a, matrixDim, matrixDim,
                    sbDim, sbDim,
                    0,  0,
                    icontxt, blockDim, 
                                 &info);
*/
  assert(info==0);

                                    
  double startTime, endTime, totalTime;
  int temp2 = 1;
  char lower = 'L';
  totalTime = 0.0;
  for (iter=0; iter < num_iter; iter++){
    srand48(1234*myRank);
    for (i=0; i < blockDim; i++){
      for (j=0; j < blockDim; j++){
        mat_A[i*blockDim+j] = drand48();
      }
    }
    startTime = MPI_Wtime();
    pdpotrf_(&lower,&matrixDim, mat_A, &temp2, &temp2, desc_a, &info);
    endTime = MPI_Wtime();
    totalTime += endTime -startTime;
  }

  if(myRank == 0) {
    printf("Completed %u iterations\n", iter);
    printf("Time elapsed per iteration: %f\n", totalTime/num_iter);
    printf("Gigaflops: %f\n", ((2./3.)*matrixDim*matrixDim*matrixDim)/
                                (totalTime/num_iter)*1E-9);
  }
/*
  MPI_Finalize();
  return 0;
*/
} /* end function main */
