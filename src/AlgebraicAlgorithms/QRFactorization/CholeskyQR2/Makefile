CCMPI = /home/hutter2/hutter2/ExternalLibraries/MPICH/installDir/bin/mpicxx
CCAMPI = /home/hutter2/hutter2/ExternalLibraries/CHARM/charm/bin/ampicxx
CFLAGS = -g -Wall -O0 -std=c++11

EXT = ~/hutter2/Executables/
OBJS1 = CholeskyQR2
OBJS2 = test

MPI: $(OBJS2)MPI
	rm *.o

AMPI: $(OBJS2)AMPI
	rm *o

$(OBJS2)MPI: $(OBJS2)MPI.o
	$(CCMPI) $(CFLAGS) -o $(EXT)$(OBJS2)MPI $(OBJS2)MPI.o -L/home/hutter2/hutter2/ExternalLibraries/CRITTER/critter/lib /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran -lpthread -lcritter

$(OBJS2)MPI.o: test.cpp CholeskyQR2.h
	$(CCMPI) $(CFLAGS) -o $(OBJS2)MPI.o -c test.cpp

$(OBJS1)MPI.o: CholeskyQR2.h CholeskyQR2.hpp
	$(CCMPI) $(CFLAGS) -o $(OBJS1)MPI.o -c CholeskyQR2.h

$(OBJS2)AMPI: $(OBJS2)AMPI.o
	$(CCAMPI) $(CFLAGS) -o $(EXT)$(OBJS2)AMPI $(OBJS2)AMPI.o /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran -lpthread

$(OBJS2)AMPI.o: test.cpp CholeskyQR2.h
	$(CCAMPI) $(CFLAGS) -o $(OBJS2)AMPI.o -c test.cpp

$(OBJS1)AMPI.o: CholeskyQR2.h CholeskyQR2.hpp
	$(CCAMPI) $(CFLAGS) -o $(OBJS1)AMPI.o -c CholeskyQR2.h

clean:
	-rm -f *.o *.gch test*MPI charmrun
