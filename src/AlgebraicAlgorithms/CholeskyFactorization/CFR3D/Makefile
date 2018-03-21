CCMPI = /home/hutter2/hutter2/ExternalLibraries/MPICH/installDir/bin/mpicxx
CCAMPI = /home/hutter2/hutter2/ExternalLibraries/CHARM/charm/multicore-linux-x86_64/bin/ampicxx
CFLAGS = -g -Wall -O3 -std=c++11

EXT = ~/hutter2/Executables/
OBJS1 = CFR3D
OBJS2 = test

MPI: $(OBJS2)MPI
	rm *.o

AMPI: $(OBJS2)AMPI
	rm *.o

$(OBJS2)MPI: $(OBJS2)MPI.o
	$(CCMPI) $(CFLAGS) -o $(EXT)$(OBJS2)MPI $(OBJS2)MPI.o -L/home/hutter2/hutter2/ExternalLibraries/CRITTER/critter/lib /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/lapack-netlib/liblapacke.a /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/lapack-netlib/liblapack.a -lgfortran -lpthread -lcritter

$(OBJS2)MPI.o: test.cpp CFR3D.h
	$(CCMPI) $(CFLAGS) -o $(OBJS2)MPI.o -c test.cpp

$(OBJS1)MPI.o: CFR3D.h CFR3D.hpp
	$(CCMPI) $(CFLAGS) -o $(OBJS1)MPI.o -c CFR3D.h

$(OBJS2)AMPI: $(OBJS2)AMPI.o
	$(CCAMPI) $(CFLAGS) -o $(EXT)$(OBJS2)AMPI $(OBJS2)AMPI.o /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapacke.a /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapack.a -lgfortran -lpthread

$(OBJS2)AMPI.o: test.cpp CFR3D.h
	$(CCAMPI) $(CFLAGS) -o $(OBJS2)AMPI.o -c test.cpp

$(OBJS1)AMPI.o: CFR3D.h CFR3D.hpp
	$(CCAMPI) $(CFLAGS) -o $(OBJS1)AMPI.o -c CFR3D.h

clean:
	-rm -f *.o *.err *.out *.gch test*MPI charmrun
