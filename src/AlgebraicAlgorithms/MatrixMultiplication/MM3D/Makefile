CCMPI = /home/hutter2/hutter2/ExternalLibraries/MPICH/installDir/bin/mpicxx
CCAMPI = /home/hutter2/hutter2/ExternalLibraries/CHARM/charm/bin/ampicxx
CFLAGS = -g -Wall -O3 -std=c++11

OBJS1 = MM3D
OBJS2 = test

MPI: $(OBJS2)MPI

AMPI: $(OBJS2)AMPI

$(OBJS2)MPI: $(OBJS2)MPI.o
	$(CCMPI) $(CFLAGS) -o $(OBJS2)MPI $(OBJS2)MPI.o -L/home/hutter2/hutter2/ExternalLibraries/CRITTER/critter/lib /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lcritter -lpthread

$(OBJS2)MPI.o: MM3D.h
	$(CCMPI) $(CFLAGS) -o $(OBJS2)MPI.o -c test.cpp

$(OBJS1)MPI.o: MM3D.h MM3D.hpp
	$(CCMPI) $(CFLAGS) -o $(OBJS1)MPI.o -c MM3D.h


$(OBJS2)AMPI: $(OBJS2)AMPI.o
	$(CCAMPI) $(CFLAGS) -o $(OBJS2)AMPI $(OBJS2)AMPI.o /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lpthread

$(OBJS2)AMPI.o: MM3D.h
	$(CCAMPI) $(CFLAGS) -o $(OBJS2)AMPI.o -c test.cpp

$(OBJS1)AMPI.o: MM3D.h MM3D.hpp
	$(CCAMPI) $(CFLAGS) -o $(OBJS1)AMPI.o -c MM3D.h

clean:
	-rm -f *.o *.gch charmrun test*MPI
