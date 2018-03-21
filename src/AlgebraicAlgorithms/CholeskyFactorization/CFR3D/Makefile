HOST=$(shell hostname)
PORTER_machine=$(shell echo $(HOST) |grep 'porter')
BGQ_machine=$(shell echo $(HOST) |grep 'mira'\|'cetus')
THETA_machine=$(shell echo $(HOST) |grep 'theta')

ifneq ($(PORTER_machine),)
  MACHINE=PORTER
  CCMPI = /home/hutter2/hutter2/ExternalLibraries/MPICH/installDir/bin/mpicxx
  CCAMPI = /home/hutter2/hutter2/ExternalLibraries/CHARM/charm/multicore-linux-x86_64/bin/ampicxx
  CFLAGS = -g -Wall -O3 -std=c++11

  EXT = ~/hutter2/Executables/
  OBJS1 = CFR3D_$(MACHINE)
  OBJS2 = cfr3d_$(MACHINE)

  MPI: $(OBJS2)MPI
	rm *.o

  AMPI: $(OBJS2)AMPI
	rm *.o

  $(OBJS2)MPI: $(OBJS2)MPI.o
	$(CCMPI) $(CFLAGS) -o $(EXT)$(OBJS2)MPI $(OBJS2)MPI.o -L/home/hutter2/hutter2/ExternalLibraries/CRITTER/critter/lib /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/lapack-netlib/liblapacke.a /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/lapack-netlib/liblapack.a -lgfortran -lpthread -lcritter

  $(OBJS2)MPI.o: test.cpp CFR3D.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS2)MPI.o -c test.cpp

  $(OBJS1)MPI.o: CFR3D.h CFR3D.hpp
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS1)MPI.o -c CFR3D.h

  $(OBJS2)AMPI: $(OBJS2)AMPI.o
	$(CCAMPI) $(CFLAGS) -o $(EXT)$(OBJS2)AMPI $(OBJS2)AMPI.o /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapacke.a /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapack.a -lgfortran -lpthread

  $(OBJS2)AMPI.o: test.cpp CFR3D.h
	$(CCAMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS2)AMPI.o -c test.cpp

  $(OBJS1)AMPI.o: CFR3D.h CFR3D.hpp
	$(CCAMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS1)AMPI.o -c CFR3D.h

  clean:
	-rm -f *.o *.err *.out *.gch $(OBJS2)*MPI charmrun
endif

ifneq ($(BGQ_machine),)
  MACHINE=BGQ
  CCMPI = mpic++11
  CFLAGS = -std=c++11 -g

  EXT = ~/scratch/Executables/
  OBJS1 = CFR3D_$(MACHINE)
  OBJS2 = cfr3d_$(MACHINE)

  MPI: $(OBJS2)MPI
	rm *.o

  $(OBJS2)MPI: $(OBJS2)MPI.o
	$(CCMPI) $(CFLAGS) -o $(EXT)$(OBJS2)MPI $(OBJS2)MPI.o -I/soft/libraries/essl/current/essl/5.1/include/ -L../../../../../CritterBlueGene/critter/lib -L/soft/libraries/alcf/current/xl/LAPACK/lib -L/soft/libraries/alcf/current/xl/BLAS/lib -L/soft/libraries/alcf/current/xl/CBLAS/lib -L/soft/libraries/essl/current/essl/5.1/lib64/ -L/soft/compilers/ibmcmp-may2016/xlf/bg/14.1/lib64 -llapack -lcblas -lblas -lesslbg -lgfortran -lxlopt -lxlf90_r -lxlfmath -lxl -lcritter

  $(OBJS2)MPI.o: test.cpp CFR3D.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS2)MPI.o -c test.cpp

  $(OBJS1)MPI.o: CFR3D.h CFR3D.hpp
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS1)MPI.o -c CFR3D.h

  clean:
	-rm -f *.o *.err *.out *.gch $(OBJS2)*MPI charmrun
endif

ifneq ($(THETA_machine),)
  MACHINE=THETA
  CCMPI = CC
  CFLAGS = -g -Wall -fast -std=c++11 -mkl=parallel

  OBJS1 = CFR3D_$(MACHINE)
  OBJS2 = cfr3d_$(MACHINE)

  MPI: $(OBJS2)MPI

  $(OBJS2)MPI: $(OBJS2)MPI.o
	$(CCMPI) $(CFLAGS) -o $(EXT)$(OBJS2)MPI $(OBJS2)MPI.o -L../../../../../CritterXC40/critter/lib -lcritter

  $(OBJS2)MPI.o: test.cpp CFR3D.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS2)MPI.o -c test.cpp

  $(OBJS1)MPI.o: CFR3D.h CFR3D.hpp
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS1)MPI.o -c CFR3D.h

  clean:
	-rm -f *.o *.err *.out *.gch $(OBJS2)*MPI charmrun
endif
