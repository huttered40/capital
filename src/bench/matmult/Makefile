HOST=$(shell hostname)
PORTER=$(shell echo $(HOST) |grep 'porter')
BGQ=$(shell echo $(HOST) |grep 'mira\|cetus')
THETA=$(shell echo $(HOST) |grep 'theta')
STAMPEDE2=$(shell echo $(HOST) |grep 'stampede2')
BLUEWATERS=$(shell echo $(HOST) |grep 'h2o')
BIN = ./../../../bin/

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
  PROFTYPE=PERFORMANCE
endif

ifneq ($(PORTER),)
  MACHINE=PORTER
  CCMPI = /home/hutter2/hutter2/ExternalLibraries/MPICH/installDir/bin/mpicxx
  ifeq ($(MPITYPE),"MPI_TYPE")
    CCMPI = /home/hutter2/hutter2/ExternalLibraries/MPICH/installDir/bin/mpicxx
  endif
  ifeq ($(MPITYPE),"AMPI_TYPE")
    CCMPI = /home/hutter2/hutter2/ExternalLibraries/CHARM/charm/bin/ampicxx
  endif
  CFLAGS = -g -Wall -O3 -std=c++14
  LIB_PATH=-L/home/hutter2/hutter2/ExternalLibraries/critter/lib
  LIBS=/home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran -lcritter -lpthread

  OBJS = summa3d_$(PROFTYPE)

  MPI: $(OBJS)
	rm *.o
  AMPI: $(OBJS2)
	rm *.o; \
	mv charmrun $(BIN)charmrun

  $(OBJS): $(OBJS).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS) $(OBJS).o $(LIB_PATH) $(LIBS)

  $(OBJS).o: MM3D.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -D$(MPITYPE) -D$(PROFTYPE) -o $(OBJS).o -c test.cpp

  clean:
	-rm -f *.o *.gch $(BIN)charmrun $(BIN)$(OBJS)
endif

ifneq ($(BGQ),)
  MACHINE=BGQ
  CCMPI = mpic++11
  CFLAGS = -std=c++11 -g -fopenmp
  INCLUDES=-I/soft/libraries/essl/current/essl/5.1/include/
  LIB_PATH=-L/home/huttered/scratch/CritterBlueGene/critter/lib -L/soft/libraries/alcf/current/xl/BLAS/lib -L/soft/libraries/alcf/current/xl/CBLAS/lib -L/soft/libraries/essl/current/essl/5.1/lib64/ -L$(IBM_MAIN_DIR)/xlf/bg/14.1/bglib64 -L$(IBM_MAIN_DIR)/xlsmp/bg/3.1/bglib64/ -L$(IBM_MAIN_DIR)/xlmass/bg/7.3/bglib64/
  LIBS=-lcblas -lblas -lgfortran -lesslbg -lxlf90_r -lxl -lxlopt -lxlfmath -lm -lcritter -Wl,--allow-multiple-definition

  OBJS1 = summa3d_$(PROFTYPE)

  MPI: $(OBJS1)
	rm *.o

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o $(INCLUDES) $(LIB_PATH) $(LIBS)

  $(OBJS1).o: MM3D.h
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

  OBJS1 = summa3d_$(PROFTYPE)

  MPI: $(OBJS1)
	rm *.o

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)

  $(OBJS1).o: MM3D.h
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

  OBJS1 = summa3d_$(PROFTYPE)

  MPI: $(OBJS1)
	rm *.o

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)

  $(OBJS1).o: MM3D.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -D$(MPITYPE) -D$(PROFTYPE) -o $(OBJS1).o -c test.cpp

  clean:
	-rm -f *.o *.gch $(BIN)$(OBJS1)

endif

ifneq ($(BLUEWATERS),)
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
    LIBS=$(LIBS) -lcublas -lcusolver
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

  OBJS1 = summa3d_$(GPU)_$(PROFTYPE)

  MPI: $(OBJS1)
	rm *.o

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)

  $(OBJS1).o: MM3D.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -D$(MPITYPE) -D$(PROFTYPE) -o $(OBJS1).o -c test.cpp

  clean:
	-rm -f *.o *.gch $(BIN)$(OBJS1)

endif
