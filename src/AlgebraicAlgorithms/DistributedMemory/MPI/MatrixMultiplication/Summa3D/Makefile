CCMPI = /home/hutter2/hutter2/ExternalLibraries/MPICH/installDir/bin/mpicxx
CFLAGS = -g -Wall -O0 -std=c++11

OBJS1 = Summa3D.o
OBJS2 = test.o
OBJS3 = testSequentialValidation.o
OBJS4 = testSequentialValidationCutBasedSumma.o

all: test testSequentialValidation testSequentialValidationCutBasedSumma

test: $(OBJS2)
	$(CCMPI) $(CFLAGS) -o test $(OBJS2) /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a

testSequentialValidation: $(OBJS3)
	$(CCMPI) $(CFLAGS) -o testSequentialValidation $(OBJS3) /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a

testSequentialValidationCutBasedSumma: $(OBJS4)
	$(CCMPI) $(CFLAGS) -o testSequentialValidationCutBasedSumma $(OBJS4) /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a

test.o: test.cpp Summa3D.h
	$(CCMPI) $(CFLAGS) -o test.o -c test.cpp

testSequentialValidation.o: testSequentialValidation.cpp Summa3D.h
	$(CCMPI) $(CFLAGS) -o testSequentialValidation.o -c testSequentialValidation.cpp

testSequentialValidationCutBasedSumma.o: testSequentialValidationCutBasedSumma.cpp Summa3D.h
	$(CCMPI) $(CFLAGS) -o testSequentialValidationCutBasedSumma.o -c testSequentialValidationCutBasedSumma.cpp

Summa3D.o: Summa3D.h Summa3D.hpp
	$(CCMPI) $(CFLAGS) -o Summa3D.o -c Summa3D.h

clean:
	-rm -f *.o *.gch test testSequentialValidation
