CCMPI = /home/hutter2/hutter2/ExternalLibraries/MPICH/installDir/bin/mpicxx
CFLAGS = -g -Wall -O0 -std=c++11

OBJS1 = CFR3D.o
OBJS2 = test.o

all: CFR3D

CFR3D: $(OBJS2)
	$(CCMPI) $(CFLAGS) -o CFR3D $(OBJS2) /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapacke.a /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapack.a /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran

test.o: test.cpp CFR3D.h
	$(CCMPI) $(CFLAGS) -o test.o -c test.cpp

CFR3D.o: CFR3D.h CFR3D.hpp
	$(CCMPI) $(CFLAGS) -o CFR3D.o -c CFR3D.h

clean:
	-rm -f *.o *.gch CFR3D
