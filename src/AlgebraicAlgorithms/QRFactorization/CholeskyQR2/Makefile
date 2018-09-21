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
  MPITYPE=MPI
endif
ifeq ($(PROFTYPE),)
  PROFTYPE=TIMER_TYPE
endif

ifneq ($(PORTER_machine),)
  MACHINE=PORTER
  CCMPI = /home/hutter2/hutter2/ExternalLibraries/MPICH/installDir/bin/mpicxx
  CCAMPI = /home/hutter2/hutter2/ExternalLibraries/CHARM/charm/bin/ampicxx
  CFLAGS = -g -Wall -O0 -std=c++11

  OBJS1 = cqr2_$(MACHINE)_mpi_$(PROFTYPE)
  OBJS2 = cqr2_$(MACHINE)_ampi_$(PROFTYPE)

  MPI: $(OBJS1)
	rm *.o
  AMPI: $(OBJS2)
	rm *o; \
	mv charmrun $(BIN)charmrun

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o -L/home/hutter2/hutter2/ExternalLibraries/CRITTER/critter/lib /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran -lpthread -lcritter

  $(OBJS1).o: test.cpp CholeskyQR2.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -D$(MPITYPE) -D$(PROFTYPE) -o $(OBJS1).o -c test.cpp

  $(OBJS2): $(OBJS2).o
	$(CCAMPI) $(CFLAGS) -o $(BIN)$(OBJS2) $(OBJS2).o /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran -lpthread

  $(OBJS2).o: test.cpp CholeskyQR2.h
	$(CCAMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -D$(MPITYPE) -D$(PROFTYPE) -o $(OBJS2).o -c test.cpp

  clean:
	-rm -f *.o *.gch $(BIN)charmrun $(BIN)$(OBJS1) $(BIN)$(OBJS2)
endif

ifneq ($(BGQ_machine),)
  MACHINE=BGQ
  CCMPI = mpic++11
  CFLAGS = -std=c++11 -g

  OBJS1 = cqr2_$(MACHINE)_$(PROFTYPE)

  MPI: $(OBJS1)
	rm *.o

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o -I/soft/libraries/essl/current/essl/5.1/include/ -L../../../../../CritterBlueGene/critter/lib -L/soft/libraries/alcf/current/xl/LAPACK/lib -L/soft/libraries/alcf/current/xl/BLAS/lib -L/soft/libraries/alcf/current/xl/CBLAS/lib -L/soft/libraries/essl/current/essl/5.1/lib64/ -L/soft/compilers/ibmcmp-may2016/xlf/bg/14.1/lib64 -llapack -lcblas -lblas -lesslbg -lgfortran -lxlopt -lxlf90_r -lxlfmath -lxl -lcritter

  $(OBJS1).o: test.cpp CholeskyQR2.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -D$(MPITYPE) -D$(PROFTYPE) -o $(OBJS1).o -c test.cpp

  clean:
	-rm -f *.o *.gch $(BIN)$(OBJS1)

endif

ifneq ($(THETA_machine),)
  MACHINE=THETA
  CCMPI = CC
  CFLAGS = -g -Wall -fast -std=c++11 -mkl=parallel -xMIC-AVX512

  OBJS1 = cqr2_$(MACHINE)_$(PROFTYPE)

  MPI: $(OBJS1)
	rm *.o

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o -L../../../../../CritterXC40/critter/lib -lcritter

  $(OBJS1).o: test.cpp CholeskyQR2.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -D$(MPITYPE) -D$(PROFTYPE) -o $(OBJS1).o -c test.cpp

  clean:
	-rm -f *.o *.gch $(BIN)$(OBJS1)
endif

ifneq ($(STAMPEDE2_machine),)
  MACHINE=STAMPEDE2
  CCMPI = mpicxx
  CFLAGS = -g -Wall -O3 -std=c++11 -mkl=parallel -xMIC-AVX512

  OBJS1 = cqr2_$(MACHINE)_$(PROFTYPE)

  MPI: $(OBJS1)
	rm *.o

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o -L../../../../../critter/lib -lcritter

  $(OBJS1).o: test.cpp CholeskyQR2.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -D$(MPITYPE) -D$(PROFTYPE) -o $(OBJS1).o -c test.cpp

  clean:
	-rm -f *.o *.gch $(BIN)$(OBJS1)
endif

ifneq ($(BlueWaters_machine),)
  MACHINE=BLUEWATERS
  CCMPI = CC
  PRG_ENV_FLAG_INTEL=$(shell echo $(PE_ENV) |grep 'INTEL')
  PRG_ENV_FLAG_GNU=$(shell echo $(PE_ENV) |grep 'GNU')

  ifneq ($(PRG_ENV_FLAG_INTEL),)
    CFLAGS = -fast -std=c++11
  endif
  ifneq ($(PRG_ENV_FLAG_GNU),)
    CFLAGS = -O3 -ffast-math -funroll-loops -std=gnu++11
  endif

  OBJS1 = cqr2_$(MACHINE)_$(PROFTYPE)

  MPI: $(OBJS1)
	rm *.o

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o -L../../../../../critter/lib -lcritter

  $(OBJS1).o: test.cpp CholeskyQR2.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -D$(MPITYPE) -D$(PROFTYPE) -o $(OBJS1).o -c test.cpp

  clean:
	-rm -f *.o *.gch $(BIN)$(OBJS1)
endif
