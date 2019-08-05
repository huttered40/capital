HOST=$(shell hostname)
PORTER=$(shell echo $(HOST) |grep 'porter')
BGQ=$(shell echo $(HOST) |grep 'mira\|cetus')
THETA=$(shell echo $(HOST) |grep 'theta')
STAMPEDE2=$(shell echo $(HOST) |grep 'stampede2')
BLUEWATERS=$(shell echo $(HOST) |grep 'h2o')
BIN=./../../../bin/

# If make is called outside of the batch script generator, a few environment variables need to be set by default (double,int)
ifeq ($(MPITYPE),)
  MPITYPE=MPI_TYPE
endif

ifneq ($(PORTER),)
  MACHINE=PORTER
  CCMPI = /home/hutter2/hutter2/ExternalLibraries/MPICH/installDir/bin/mpicxx
  ifeq ($(MPITYPE),"MPI_TYPE")
    CCMPI = /home/hutter2/hutter2/ExternalLibraries/MPICH/installDir/bin/mpicxx
    echo "HERE??"
  endif
  ifeq ($(MPITYPE),"AMPI_TYPE")
    CCMPI = /home/hutter2/hutter2/ExternalLibraries/CHARM/charm/bin/ampicxx
    echo "HERE??"
  endif
  CFLAGS = -g -Wall -O0 -std=c++14
  LIB_PATH=-L/home/hutter2/hutter2/ExternalLibraries/critter/lib
  LIBS=/home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran -lpthread -lcritter

  OBJS = cacqr2

  $(OBJS): $(OBJS).o
        $(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS) $(OBJS).o $(LIB_PATH) $(LIBS)
	rm *.o
	mv charmrun $(BIN)charmrun	# Note: this will just not work for MPI, but that shouldn't cause a problem

  $(OBJS).o: test.cpp cacqr.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -D$(MPITYPE) -D$(PROFTYPE) -o $(OBJS).o -c test.cpp

  clean:
	-rm -f *.o *.gch $(BIN)charmrun $(BIN)$(OBJS)
endif

ifneq ($(BGQ),)
  MACHINE=BGQ
  CCMPI = mpic++11
  CFLAGS = -std=c++11 -g
  INCLUDES=-I/soft/libraries/essl/current/essl/5.1/include/
  LIB_PATH=-L../../../../../CritterBlueGene/critter/lib -L/soft/libraries/alcf/current/xl/LAPACK/lib -L/soft/libraries/alcf/current/xl/BLAS/lib -L/soft/libraries/alcf/current/xl/CBLAS/lib -L/soft/libraries/essl/current/essl/5.1/lib64/ -L/soft/compilers/ibmcmp-may2016/xlf/bg/14.1/lib64
  LIBS=-llapack -lcblas -lblas -lesslbg -lgfortran -lxlopt -lxlf90_r -lxlfmath -lxl -lcritter

  OBJS1 = cacqr2

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o $(INCLUDES) $(LIB_PATH) $(LIBS)
	rm *.o

  $(OBJS1).o: test.cpp cacqr.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -D$(MPITYPE) -D$(PROFTYPE) -o $(OBJS1).o -c test.cpp

  clean:
	-rm -f *.o *.gch $(BIN)$(OBJS1)

endif

ifneq ($(THETA),)
  MACHINE=THETA
  CCMPI = CC
  CFLAGS = -g -Wall -fast -std=c++11 -mkl=parallel -xMIC-AVX512
  LIB_PATH=-L../../../../../CritterXC40/critter/lib
  LIBS=-lcritter

  OBJS1 = cacqr2

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)
	rm *.o

  $(OBJS1).o: test.cpp cacqr.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -D$(MPITYPE) -D$(PROFTYPE) -o $(OBJS1).o -c test.cpp

  clean:
	-rm -f *.o *.gch $(BIN)$(OBJS1)
endif

ifneq ($(STAMPEDE2),)
  MACHINE=STAMPEDE2
  CCMPI = mpicxx
  CFLAGS = -g -Wall -O3 -std=c++14 -mkl=parallel -xMIC-AVX512
  LIB_PATH=-L../../../../../critter/lib
  LIBS=-lcritter

  OBJS1 = cacqr2

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)
	rm *.o

  $(OBJS1).o: test.cpp cacqr.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -D$(MPITYPE) -D$(PROFTYPE) -o $(OBJS1).o -c test.cpp

  clean:
	-rm -f *.o *.gch $(BIN)$(OBJS1)
endif

ifneq ($(BLUEWATERS),)
  MACHINE=BLUEWATERS
  PRG_ENV_FLAG_INTEL=$(shell echo $(PE_ENV) |grep 'INTEL')
  PRG_ENV_FLAG_GNU=$(shell echo $(PE_ENV) |grep 'GNU')
  GPU_ACCEL=$(shell echo $(GPU) |grep 'GPUACCEL')
  GPU_NO_ACCEL=$(shell echo $(GPU) |grep 'NOGPU')
  LIB_PATH=-L../../../../../critter/lib
  LIBS=-lcritter
  CCMPI=CC
  CCGPU=CC	# default unless overwritten
  ENVS=-D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -D$(MPITYPE) -D$(PROFTYPE) -D$(GPU)

  ifneq ($(GPU_ACCEL),)
    LIBS=$(LIBS) -lcublas -lcusolver
    CCGPU=nvcc
  endif

  ifneq ($(PRG_ENV_FLAG_INTEL),)
    CFLAGS = -fast -std=c++1y
  endif
  ifneq ($(PRG_ENV_FLAG_GNU),)
    ifneq ($(GPU_ACCEL),)
      CFLAGS = -gencode=arch=compute_35,code=compute_35 -O3 -use_fast_math -std=c++11
    endif
    ifneq ($(GPU_NO_ACCEL),)
      CFLAGS = -O3 -ffast-math -funroll-loops -ftree-vectorize -std=gnu++1y
    endif
  endif

  OBJS1 = cacqr2_$(GPU)

  $(OBJS1): $(OBJS1).o #offload.o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)
	rm *.o

  $(OBJS1).o: test.cpp cacqr.h
	$(CCMPI) $(CFLAGS) $(ENVS) -o $(OBJS1).o -c test.cpp

  offload.o: ../../../Offload/offload.h
	$(CCGPU) $(CFLAGS) $(ENVS) -o offload.o

  clean:
	-rm -f *.o *.gch $(BIN)$(OBJS1)
endif
