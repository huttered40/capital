CCMPI = /home/hutter2/hutter2/ExternalLibraries/MPICH/installDir/bin/mpicxx
CFLAGS = -g -Wall -O0 -std=c++11

OBJS1 = CholeskyQR2.o
OBJS2 = testSequentialValidation1D.o
OBJS3 = testSequentialValidation3D.o
OBJS4 = testSequentialValidationTunable.o

all: testSequentialValidation1D testSequentialValidation3D testSequentialValidationTunable

testSequentialValidation1D: $(OBJS2)
	$(CCMPI) $(CFLAGS) -o testSequentialValidation1D $(OBJS2) /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lpthread

testSequentialValidation3D: $(OBJS3)
	$(CCMPI) $(CFLAGS) -o testSequentialValidation3D $(OBJS3) /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lpthread

testSequentialValidationTunable: $(OBJS4)
	$(CCMPI) $(CFLAGS) -o testSequentialValidationTunable $(OBJS3) /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lpthread

testSequentialValidation1D.o: testSequentialValidation1D.cpp CholeskyQR2.h
	$(CCMPI) $(CFLAGS) -o testSequentialValidation1D.o -c testSequentialValidation1D.cpp

testSequentialValidation3D.o: testSequentialValidation3D.cpp CholeskyQR2.h
	$(CCMPI) $(CFLAGS) -o testSequentialValidation3D.o -c testSequentialValidation3D.cpp

testSequentialValidationTunable.o: testSequentialValidationTunable.cpp CholeskyQR2.h
	$(CCMPI) $(CFLAGS) -o testSequentialValidationTunable.o -c testSequentialValidationTunable.cpp

Summa3D.o: CholeskyQR2.h CholeskyQR2.hpp
	$(CCMPI) $(CFLAGS) -o CholeskyQR2.o -c CholeskyQR2.h

clean:
	-rm -f *.o *.gch
