HOST=$(shell hostname)
PORTER_machine=$(shell echo $(HOST) |grep 'porter')
BGQ_machine=$(shell echo $(HOST) |grep 'mira'\|'cetus')
THETA_machine=$(shell echo $(HOST) |grep 'theta')

ifneq ($(PORTER_machine),)
  MACHINE=PORTER
  CCMPI = /home/hutter2/hutter2/ExternalLibraries/MPICH/installDir/bin/mpicxx
  CCAMPI = /home/hutter2/hutter2/ExternalLibraries/CHARM/charm/bin/ampicxx
  CFLAGS = -g -Wall -O3 -std=c++11
  EXT = ~/hutter2/Executables/
  OBJS1 = MM3D_$(MACHINE)
  OBJS2 = mm3d_$(MACHINE)
  MPI: $(OBJS2)MPI
	rm *.o
  AMPI: $(OBJS2)AMPI
	rm *.o

  $(OBJS2)MPI: $(OBJS2)MPI.o
	$(CCMPI) $(CFLAGS) -o $(EXT)$(OBJS2)MPI $(OBJS2)MPI.o -L/home/hutter2/hutter2/ExternalLibraries/CRITTER/critter/lib /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lcritter -lpthread

  $(OBJS2)MPI.o: MM3D.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS2)MPI.o -c test.cpp

  $(OBJS1)MPI.o: MM3D.h MM3D.hpp
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS1)MPI.o -c MM3D.h

  $(OBJS2)AMPI: $(OBJS2)AMPI.o
	$(CCAMPI) $(CFLAGS) -o $(EXT)$(OBJS2)AMPI $(OBJS2)AMPI.o /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lpthread

  $(OBJS2)AMPI.o: MM3D.h
	$(CCAMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS2)AMPI.o -c test.cpp

  $(OBJS1)AMPI.o: MM3D.h MM3D.hpp
	$(CCAMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS1)AMPI.o -c MM3D.h

  clean:
	-rm -f *.o *.gch charmrun $(OBJS2)*MPI
endif

ifneq ($(BGQ_machine),)
  MACHINE=BGQ
  CCMPI = mpic++11
  CFLAGS = -std=c++11 -g -fopenmp
  #-qlanglvl=rvaluereferences

  EXT = ~/scratch/Executables/
  OBJS1 = MM3D_$(MACHINE)
  OBJS2 = mm3d_$(MACHINE)

  MPI: $(OBJS2)MPI
	rm *.o
  AMPI: $(OBJS2)AMPI
	rm *.o

  $(OBJS2)MPI: $(OBJS2)MPI.o
	$(CCMPI) $(CFLAGS) -o $(EXT)$(OBJS2)MPI $(OBJS2)MPI.o -I/soft/libraries/essl/current/essl/5.1/include/ -L/home/huttered/scratch/CritterBlueGene/critter/lib -L/soft/libraries/alcf/current/xl/BLAS/lib -L/soft/libraries/alcf/current/xl/CBLAS/lib -L/soft/libraries/essl/current/essl/5.1/lib64/ -L$(IBM_MAIN_DIR)/xlf/bg/14.1/bglib64 -L$(IBM_MAIN_DIR)/xlsmp/bg/3.1/bglib64/ -L$(IBM_MAIN_DIR)/xlmass/bg/7.3/bglib64/ -lcblas -lblas -lgfortran -lesslbg -lxlf90_r -lxl -lxlopt -lxlfmath -lm -lcritter -Wl,--allow-multiple-definition

  $(OBJS2)MPI.o: MM3D.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS2)MPI.o -c test.cpp

  $(OBJS1)MPI.o: MM3D.h MM3D.hpp
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS1)MPI.o -c MM3D.h

  clean:
	-rm -f *.o *.gch charmrun $(OBJS2)*MPI
endif

ifneq ($(THETA_machine),)
  MACHINE=THETA
  CCMPI = CC
  CFLAGS = -g -Wall -O3 -std=c++11 -mkl=parallel

  EXT = ~/scratch/Executables/
  OBJS1 = MM3D_$(MACHINE)
  OBJS2 = mm3d_$(MACHINE)

  MPI: $(OBJS2)MPI
	rm *.o

  $(OBJS2)MPI: $(OBJS2)MPI.o
	$(CCMPI) $(CFLAGS) -o $(EXT)$(OBJS2)MPI $(OBJS2)MPI.o -L../../../../../CritterXC40/critter/lib -lcritter

  $(OBJS2)MPI.o: MM3D.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS2)MPI.o -c test.cpp

  $(OBJS1)MPI.o: MM3D.h MM3D.hpp
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS1)MPI.o -c MM3D.h
  clean:
	-rm -f *.o *.gch charmrun $(OBJS2)*MPI

endif
