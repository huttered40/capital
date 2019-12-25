HOST=$(shell hostname)
PORTER=$(shell echo $(HOST) |grep 'porter')
BGQ=$(shell echo $(HOST) |grep 'mira\|cetus')
THETA=$(shell echo $(HOST) |grep 'theta')
STAMPEDE2=$(shell echo $(HOST) |grep 'stampede2')
BLUEWATERS=$(shell echo $(HOST) |grep 'h2o')
BIN=./../../bin/
ALG=./../../alg/inverse/newton/

# If make is called outside of the batch script generator, a few environment variables need to be set by default (double,int)
ifeq ($(MPITYPE),)
  MPITYPE=MPI_TYPE
endif

ifneq ($(PORTER),)
  MACHINE=PORTER
  CCMPI = ~/hutter2/external/MPICH/installDir/bin/mpicxx
  ifeq ($(MPITYPE),"MPI_TYPE")
    CCMPI = ~/hutter2/external/MPICH/installDir/bin/mpicxx
  endif
  ifeq ($(MPITYPE),"AMPI_TYPE")
    CCMPI = ~/hutter2/external/CHARM/charm/bin/ampicxx
  endif
  CFLAGS = -g -Wall -O3 -std=c++14
  LIB_PATH=-L/home/hutter2/hutter2/critter/lib
  LIBS=~/hutter2/external/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a ~/hutter2/external/BLAS/OpenBLAS/lapack-netlib/liblapacke.a ~/hutter2/external/BLAS/OpenBLAS/lapack-netlib/liblapack.a -lgfortran -lpthread -lcritter

  OBJS = newton

  $(OBJS): $(OBJS).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS) $(OBJS).o $(LIB_PATH) $(LIBS)
	rm *.o; \
	#mv charmrun $(BIN)charmrun

  $(OBJS).o: newton.cpp $(ALG)newton.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(MPITYPE) -o $(OBJS).o -c newton.cpp

  clean:
	-rm -f *.o *.err *.out *.gch $(BIN)charmrun $(BIN)$(OBJS)
endif

ifneq ($(BGQ),)
  MACHINE=BGQ
  CCMPI = mpic++11
  CFLAGS = -std=c++11 -g
  INCLUDES=-I/soft/libraries/essl/current/essl/5.1/include/
  LIB_PATH=-L../../../../CritterBlueGene/critter/lib -L/soft/libraries/alcf/current/xl/LAPACK/lib -L/soft/libraries/alcf/current/xl/BLAS/lib -L/soft/libraries/alcf/current/xl/CBLAS/lib -L/soft/libraries/essl/current/essl/5.1/lib64/ -L/soft/compilers/ibmcmp-may2016/xlf/bg/14.1/lib64
  LIBS=-llapack -lcblas -lblas -lesslbg -lgfortran -lxlopt -lxlf90_r -lxlfmath -lxl -lcritter

  OBJS1 = newton

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o $(INCLUDES) $(LIB_PATH) $(LIBS)
	rm *.o

  $(OBJS1).o: newton.cpp $(ALG)newton.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(MPITYPE) -o $(OBJS1).o -c newton.cpp

  clean:
	-rm -f *.o *.err *.out *.gch $(BIN)$(OBJS1)
endif

ifneq ($(THETA),)
  MACHINE=THETA
  CCMPI = CC
  CFLAGS = -g -Wall -fast -std=c++11 -mkl=parallel -xMIC-AVX512
  LIB_PATH=-L../../../../CritterXC40/critter/lib
  LIBS=-lcritter

  OBJS1 = newton

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)
	rm *.o

  $(OBJS1).o: newton.cpp $(ALG)newton.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(MPITYPE) -o $(OBJS1).o -c newton.cpp

  clean:
	-rm -f *.o *.err *.out *.gch $(BIN)$(OBJS1)
endif

ifneq ($(STAMPEDE2),)
  MACHINE=STAMPEDE2
  CCMPI = mpicxx
  CFLAGS = -g -Wall -O3 -std=c++14 -mkl=parallel -xMIC-AVX512
  LIB_PATH=-L../../../../critter/lib
  LIBS=-lcritter

  OBJS1 = newton

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)
	rm *.o

  $(OBJS1).o: newton.cpp $(ALG)newton.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(MPITYPE) -o $(OBJS1).o -c newton.cpp

  clean:
	-rm -f *.o *.err *.out *.gch $(BIN)$(OBJS1)
endif

ifneq ($(BLUEWATERS),)
  MACHINE=BLUEWATERS
  CCMPI = CC
  PRG_ENV_FLAG_INTEL=$(shell echo $(PE_ENV) |grep 'INTEL')
  PRG_ENV_FLAG_GNU=$(shell echo $(PE_ENV) |grep 'GNU')
  GPU_ACCEL=$(shell echo $(GPU) |grep 'GPUACCEL')
  GPU_NO_ACCEL=$(shell echo $(GPU) |grep 'NoGPUACCEL')
  LIB_PATH=-L../../../../critter/lib
  LIBS=-lcritter
  ENVS=-D$(MACHINE) -D$(MPITYPE)

  ifneq ($(GPU_ACCEL),)
    CCMPI=nvcc
    LIBS=-lcublas -lcusolver
  endif
  ifneq ($(GPU_NO_ACCEL),)
    CCMPI=CC
  endif

  ifneq ($(PRG_ENV_FLAG_INTEL),)
    CFLAGS = -fast -std=c++1y
  endif
  ifneq ($(PRG_ENV_FLAG_GNU),)
    CFLAGS = -O3 -ffast-math -funroll-loops -std=gnu++1y
  endif

  OBJS1 = newton

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)
	rm *.o

  $(OBJS1).o: newton.cpp $(ALG)newton.h
	$(CCMPI) $(CFLAGS) $(ENVS) -o $(OBJS1).o -c $(OBJS1).cpp

  clean:
	-rm -f *.o *.err *.out *.gch $(BIN)$(OBJS1)
endif
