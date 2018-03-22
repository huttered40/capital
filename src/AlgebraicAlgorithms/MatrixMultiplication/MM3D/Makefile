HOST=$(shell hostname)
PORTER_machine=$(shell echo $(HOST) |grep 'porter')
BGQ_machine=$(shell echo $(HOST) |grep 'mira\|cetus')
THETA_machine=$(shell echo $(HOST) |grep 'theta')
BIN = ./../../../bin/

ifneq ($(PORTER_machine),)
  MACHINE=PORTER
  CCMPI = /home/hutter2/hutter2/ExternalLibraries/MPICH/installDir/bin/mpicxx
  CCAMPI = /home/hutter2/hutter2/ExternalLibraries/CHARM/charm/bin/ampicxx
  CFLAGS = -g -Wall -O3 -std=c++11
  OBJS1 = mm3d_$(MACHINE)_mpi
  OBJS2 = mm3d_$(MACHINE)_ampi

  MPI: $(OBJS1)
	rm *.o
  AMPI: $(OBJS2)
	rm *.o

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o -L/home/hutter2/hutter2/ExternalLibraries/CRITTER/critter/lib /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lcritter -lpthread

  $(OBJS1).o: MM3D.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -o $(OBJS1).o -c test.cpp

  $(OBJS2): $(OBJS2).o
	$(CCAMPI) $(CFLAGS) -o $(BIN)$(OBJS2) $(OBJS2).o /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lpthread

  $(OBJS2).o: MM3D.h
	$(CCAMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -o $(OBJS2).o -c test.cpp

  clean:
	-rm -f *.o *.gch charmrun $(BIN)$(OBJS1) $(BIN)$(OBJS2)
endif

ifneq ($(BGQ_machine),)
  MACHINE=BGQ
  CCMPI = mpic++11
  CFLAGS = -std=c++11 -g -fopenmp
  #-qlanglvl=rvaluereferences

  OBJS1 = mm3d_$(MACHINE)

  MPI: $(OBJS1)
	rm *.o

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o -I/soft/libraries/essl/current/essl/5.1/include/ -L/home/huttered/scratch/CritterBlueGene/critter/lib -L/soft/libraries/alcf/current/xl/BLAS/lib -L/soft/libraries/alcf/current/xl/CBLAS/lib -L/soft/libraries/essl/current/essl/5.1/lib64/ -L$(IBM_MAIN_DIR)/xlf/bg/14.1/bglib64 -L$(IBM_MAIN_DIR)/xlsmp/bg/3.1/bglib64/ -L$(IBM_MAIN_DIR)/xlmass/bg/7.3/bglib64/ -lcblas -lblas -lgfortran -lesslbg -lxlf90_r -lxl -lxlopt -lxlfmath -lm -lcritter -Wl,--allow-multiple-definition

  $(OBJS1).o: MM3D.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -o $(OBJS1).o -c test.cpp

  clean:
	-rm -f *.o *.gch $(BIN)$(OBJS1)
endif

ifneq ($(THETA_machine),)
  MACHINE=THETA
  CCMPI = CC
  CFLAGS = -g -Wall -O3 -std=c++11 -mkl=parallel

  OBJS1 = mm3d_$(MACHINE)

  MPI: $(OBJS1)
	rm *.o

  $(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o -L../../../../../CritterXC40/critter/lib -lcritter

  $(OBJS1).o: MM3D.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -D$(DATATYPE) -D$(INTTYPE) -o $(OBJS1).o -c test.cpp

  clean:
	-rm -f *.o *.gch $(BIN)$(OBJS1)

endif
