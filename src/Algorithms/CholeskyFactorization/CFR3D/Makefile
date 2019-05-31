HOST=$(shell hostname)
PORTER_machine=$(shell echo $(HOST) |grep 'porter')
BGQ_machine=$(shell echo $(HOST) |grep 'mira\|cetus')
THETA_machine=$(shell echo $(HOST) |grep 'theta')
STAMPEDE2_machine=$(shell echo $(HOST) |grep 'stampede2')
BlueWaters_machine=$(shell echo $(HOST) |grep 'h2o')
BIN=./../../../bin/

# If make is called outside of the batch script generator, a few environment variables need to be set by default (double,int)
ifeq ($(DATATYPE),)
  DATATYPE=DOUBLE_TYPE
endif
ifeq ($(INTTYPE),)
  INTTYPE=INT_TYPE
endif
ifeq ($(MPITYPE),)
  MPITYPE=MPI_TYPE
endif
ifeq ($(PROFTYPE),)
  PROFTYPE=TIMER_TYPE
endif

ifneq ($(PORTER_machine),)
  MACHINE=PORTER
  CCMPI = /home/hutter2/hutter2/ExternalLibraries/MPICH/installDir/bin/mpicxx
  CCAMPI = /home/hutter2/hutter2/ExternalLibraries/CHARM/charm/multicore-linux-x86_64/bin/ampicxx
  CFLAGS = -g -Wall -O3 -std=c++14
  LIB_PATH=-L/home/hutter2/hutter2/ExternalLibraries/CRITTER/critter/lib
  LIBS=/home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/lapack-netlib/liblapacke.a /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/lapack-netlib/liblapack.a -lgfortran -lpthread -lcritter

  OBJS1 = cfr3d_$(MACHINE)_mpi_$(PROFTYPE)
  OBJS2 = cfr3d_$(MACHINE)_ampi_$(PROFTYPE)

  MPI: $(OBJS1)
	rm *.o
  AMPI: $(OBJS2)
	rm *.o; \
	mv charmrun $(BIN)charmrun

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)

  $(OBJS1).o: test.cpp CFR3D.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -D$(MPITYPE) -D$(PROFTYPE) -o $(OBJS1).o -c test.cpp

  $(OBJS2): $(OBJS2).o
	$(CCAMPI) $(CFLAGS) -o $(BIN)$(OBJS2) $(OBJS2).o $(LIBS)

  $(OBJS2).o: test.cpp CFR3D.h
	$(CCAMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -D$(MPITYPE) -D$(PROFTYPE) -o $(OBJS2).o -c test.cpp

  clean:
	-rm -f *.o *.err *.out *.gch $(BIN)charmrun $(BIN)$(OBJS1) $(BIN)$(OBJS2)
endif

ifneq ($(BGQ_machine),)
  MACHINE=BGQ
  CCMPI = mpic++11
  CFLAGS = -std=c++11 -g
  INCLUDES=-I/soft/libraries/essl/current/essl/5.1/include/
  LIB_PATH=-L../../../../../CritterBlueGene/critter/lib -L/soft/libraries/alcf/current/xl/LAPACK/lib -L/soft/libraries/alcf/current/xl/BLAS/lib -L/soft/libraries/alcf/current/xl/CBLAS/lib -L/soft/libraries/essl/current/essl/5.1/lib64/ -L/soft/compilers/ibmcmp-may2016/xlf/bg/14.1/lib64
  LIBS=-llapack -lcblas -lblas -lesslbg -lgfortran -lxlopt -lxlf90_r -lxlfmath -lxl -lcritter

  OBJS1 = cfr3d_$(MACHINE)_$(PROFTYPE)

  MPI: $(OBJS1)
	rm *.o

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o $(INCLUDES) $(LIB_PATH) $(LIBS)

  $(OBJS1).o: test.cpp CFR3D.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -D$(MPITYPE) -D$(PROFTYPE) -o $(OBJS1).o -c test.cpp

  clean:
	-rm -f *.o *.err *.out *.gch $(BIN)$(OBJS1)
endif

ifneq ($(THETA_machine),)
  MACHINE=THETA
  CCMPI = CC
  CFLAGS = -g -Wall -fast -std=c++11 -mkl=parallel -xMIC-AVX512
  LIB_PATH=-L../../../../../CritterXC40/critter/lib
  LIBS=-lcritter

  OBJS1 = cfr3d_$(MACHINE)_$(PROFTYPE)

  MPI: $(OBJS1)
	rm *.o

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)

  $(OBJS1).o: test.cpp CFR3D.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -D$(MPITYPE) -D$(PROFTYPE) -o $(OBJS1).o -c test.cpp

  clean:
	-rm -f *.o *.err *.out *.gch $(BIN)$(OBJS1)
endif

ifneq ($(STAMPEDE2_machine),)
  MACHINE=STAMPEDE2
  CCMPI = mpicxx
  CFLAGS = -g -Wall -O3 -std=c++14 -mkl=parallel -xMIC-AVX512
  LIB_PATH=-L../../../../../critter/lib
  LIBS=-lcritter

  OBJS1 = cfr3d_$(MACHINE)_$(PROFTYPE)

  MPI: $(OBJS1)
	rm *.o

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)

  $(OBJS1).o: test.cpp CFR3D.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -D$(MPITYPE) -D$(PROFTYPE) -o $(OBJS1).o -c test.cpp

  clean:
	-rm -f *.o *.err *.out *.gch $(BIN)$(OBJS1)
endif

ifneq ($(BlueWaters_machine),)
  MACHINE=BLUEWATERS
  CCMPI = CC
  PRG_ENV_FLAG_INTEL=$(shell echo $(PE_ENV) |grep 'INTEL')
  PRG_ENV_FLAG_GNU=$(shell echo $(PE_ENV) |grep 'GNU')
  GPU_ACCEL=$(shell echo $(GPU) |grep 'GPUACCEL')
  GPU_NO_ACCEL=$(shell echo $(GPU) |grep 'NoGPUACCEL')
  LIB_PATH=-L../../../../../critter/lib
  LIBS=-lcritter

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

  OBJS1 = cfr3d_$(MACHINE)_$(GPU)_$(PROFTYPE)

  MPI: $(OBJS1)
	rm *.o

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)

  $(OBJS1).o: test.cpp CFR3D.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -D$(MPITYPE) -D$(PROFTYPE) -o $(OBJS1).o -c test.cpp

  clean:
	-rm -f *.o *.err *.out *.gch $(BIN)$(OBJS1)
endif
