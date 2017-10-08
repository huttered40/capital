CCMPI = /home/hutter2/hutter2/ExternalLibraries/MPICH/installDir/bin/mpicxx
CFLAGS = -g -Wall -O3 -std=c++11

OBJS1 = CFR3D.o
OBJS2 = testPerformanceUpper.o
OBJS3 = testPerformanceLower.o
OBJS4 = testSequentialValidationLower.o
OBJS5 = testSequentialValidationUpper.o

all: testPerformanceUpper testPerformanceLower testSequentialValidationLower testSequentialValidationUpper

testPerformanceUpper: $(OBJS2)
	$(CCMPI) $(CFLAGS) -o testPerformanceUpper $(OBJS2) /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapacke.a /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapack.a /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran -lpthread

testPerformanceLower: $(OBJS3)
	$(CCMPI) $(CFLAGS) -o testPerformanceLower $(OBJS3) /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapacke.a /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapack.a /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran -lpthread

testSequentialValidationLower: $(OBJS4)
	$(CCMPI) $(CFLAGS) -o testSequentialValidationLower $(OBJS4) /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapacke.a /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapack.a /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran -lpthread

testSequentialValidationUpper: $(OBJS5)
	$(CCMPI) $(CFLAGS) -o testSequentialValidationUpper $(OBJS5) /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapacke.a /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapack.a /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran -lpthread

testPerformanceUpper.o: testPerformanceUpper.cpp CFR3D.h
	$(CCMPI) $(CFLAGS) -o testPerformanceUpper.o -c testPerformanceUpper.cpp

testPerformanceLower.o: testPerformanceLower.cpp CFR3D.h
	$(CCMPI) $(CFLAGS) -o testPerformanceLower.o -c testPerformanceLower.cpp

testSequentialValidationLower.o: testSequentialValidationLower.cpp CFR3D.h
	$(CCMPI) $(CFLAGS) -o testSequentialValidationLower.o -c testSequentialValidationLower.cpp

testSequentialValidationUpper.o: testSequentialValidationUpper.cpp CFR3D.h
	$(CCMPI) $(CFLAGS) -o testSequentialValidationUpper.o -c testSequentialValidationUpper.cpp

CFR3D.o: CFR3D.h CFR3D.hpp
	$(CCMPI) $(CFLAGS) -o CFR3D.o -c CFR3D.h

clean:
	-rm -f *.o *.gch test testSequentialValidationLower testSequentialValidationUpper
