CCMPI = /home/hutter2/hutter2/ExternalLibraries/MPICH/installDir/bin/mpicxx
CFLAGS = -g -Wall -O0 -std=c++11

OBJS1 = SquareMM3D.o
OBJS2 = test.o
OBJS3 = testSequentialValidation.o
OBJS4 = testSequentialValidationCut.o

all: test testSequentialValidation testSequentialValidationCut

test: $(OBJS2)
	$(CCMPI) $(CFLAGS) -o test $(OBJS2) /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lpthread

testSequentialValidation: $(OBJS3)
	$(CCMPI) $(CFLAGS) -o testSequentialValidation $(OBJS3) /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lpthread

testSequentialValidationCut: $(OBJS4)
	$(CCMPI) $(CFLAGS) -o testSequentialValidationCut $(OBJS4) /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lpthread

test.o: test.cpp SquareMM3D.h
	$(CCMPI) $(CFLAGS) -o test.o -c test.cpp

testSequentialValidation.o: testSequentialValidation.cpp SquareMM3D.h
	$(CCMPI) $(CFLAGS) -o testSequentialValidation.o -c testSequentialValidation.cpp

testSequentialValidationCut.o: testSequentialValidationCut.cpp SquareMM3D.h
	$(CCMPI) $(CFLAGS) -o testSequentialValidationCut.o -c testSequentialValidationCut.cpp

Summa3D.o: SquareMM3D.h SquareMM3D.hpp
	$(CCMPI) $(CFLAGS) -o SquareMM3D.o -c SquareMM3D.h

clean:
	-rm -f *.o *.gch test testSequentialValidation testSequentialValidationCut
