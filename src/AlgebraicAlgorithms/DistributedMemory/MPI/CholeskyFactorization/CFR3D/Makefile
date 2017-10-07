CCMPI = /home/hutter2/hutter2/ExternalLibraries/MPICH/installDir/bin/mpicxx
CFLAGS = -g -Wall -O3 -std=c++11

OBJS1 = CFR3D.o
OBJS2 = test.o
OBJS3 = testSequentialValidationLower.o
OBJS4 = testSequentialValidationUpper.o

all: test testSequentialValidationLower testSequentialValidationUpper

test: $(OBJS2)
	$(CCMPI) $(CFLAGS) -o test $(OBJS2) /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapacke.a /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapack.a /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran -lpthread

testSequentialValidationLower: $(OBJS3)
	$(CCMPI) $(CFLAGS) -o testSequentialValidationLower $(OBJS3) /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapacke.a /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapack.a /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran -lpthread

testSequentialValidationUpper: $(OBJS4)
	$(CCMPI) $(CFLAGS) -o testSequentialValidationUpper $(OBJS4) /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapacke.a /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapack.a /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran -lpthread

test.o: test.cpp CFR3D.h
	$(CCMPI) $(CFLAGS) -o test.o -c test.cpp

testSequentialValidationLower.o: testSequentialValidationLower.cpp CFR3D.h
	$(CCMPI) $(CFLAGS) -o testSequentialValidationLower.o -c testSequentialValidationLower.cpp

testSequentialValidationUpper.o: testSequentialValidationUpper.cpp CFR3D.h
	$(CCMPI) $(CFLAGS) -o testSequentialValidationUpper.o -c testSequentialValidationUpper.cpp

CFR3D.o: CFR3D.h CFR3D.hpp
	$(CCMPI) $(CFLAGS) -o CFR3D.o -c CFR3D.h

clean:
	-rm -f *.o *.gch test testSequentialValidationLower testSequentialValidationUpper
