<<<<<<< HEAD
CCMPI = /home/hutter2/hutter2/ExternalLibraries/MPICH/installDir/bin/mpicxx
CFLAGS = -g -Wall -O0 -std=c++11
=======
#CCMPI = /home/hutter2/hutter2/ExternalLibraries/MPICH/installDir/bin/mpicxx
#CFLAGS = -g -Wall -O0 -std=c++11
BWFLAGS = CC -std=gnu11 -mkl=sequential -fast
>>>>>>> 38e9d69... Added all the changes and files that I must have forgot to back when debugging and running this last week on Blue Waters.

OBJS1 = CholeskyQR2.o
OBJS2 = testSequentialValidation1D.o
OBJS3 = testSequentialValidation3D.o
OBJS4 = testSequentialValidationTunable.o
OBJS5 = testPerformance1D.o
OBJS6 = testPerformance3D.o
OBJS7 = testPerformanceTunable.o

all: testSequentialValidation1D testSequentialValidation3D testSequentialValidationTunable testPerformance1D testPerformance3D testPerformanceTunable

testSequentialValidation1D: $(OBJS2)
	$(CCMPI) $(CFLAGS) -o testSequentialValidation1D $(OBJS2) /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran -lpthread

testSequentialValidation3D: $(OBJS3)
	$(CCMPI) $(CFLAGS) -o testSequentialValidation3D $(OBJS3) /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran -lpthread

testSequentialValidationTunable: $(OBJS4)
	$(CCMPI) $(CFLAGS) -o testSequentialValidationTunable $(OBJS4) /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran -lpthread

testPerformance1D: $(OBJS5)
	$(CCMPI) $(CFLAGS) -o testPerformance1D $(OBJS5) /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran -lpthread

testPerformance3D: $(OBJS6)
	$(CCMPI) $(CFLAGS) -o testPerformance3D $(OBJS6) /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran -lpthread

testPerformanceTunable: $(OBJS7)
	$(CCMPI) $(CFLAGS) -o testPerformanceTunable $(OBJS7) /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran -lpthread

testSequentialValidation1D.o: testSequentialValidation1D.cpp CholeskyQR2.h
	$(CCMPI) $(CFLAGS) -o testSequentialValidation1D.o -c testSequentialValidation1D.cpp

testSequentialValidation3D.o: testSequentialValidation3D.cpp CholeskyQR2.h
	$(CCMPI) $(CFLAGS) -o testSequentialValidation3D.o -c testSequentialValidation3D.cpp

testSequentialValidationTunable.o: testSequentialValidationTunable.cpp CholeskyQR2.h
	$(CCMPI) $(CFLAGS) -o testSequentialValidationTunable.o -c testSequentialValidationTunable.cpp

testPerformance1D.o: testPerformance1D.cpp CholeskyQR2.h
	$(CCMPI) $(CFLAGS) -o testPerformance1D.o -c testPerformance1D.cpp

testPerformance3D.o: testPerformance3D.cpp CholeskyQR2.h
	$(CCMPI) $(CFLAGS) -o testPerformance3D.o -c testPerformance3D.cpp

testPerformanceTunable.o: testPerformanceTunable.cpp CholeskyQR2.h
	$(CCMPI) $(CFLAGS) -o testPerformanceTunable.o -c testPerformanceTunable.cpp

Summa3D.o: CholeskyQR2.h CholeskyQR2.hpp
	$(CCMPI) $(CFLAGS) -o CholeskyQR2.o -c CholeskyQR2.h

clean:
<<<<<<< HEAD
	-rm -f *.o *.gch testSequentialValidation1D testSequentialValidation3D testSequentialValidationTunable testPerformance1D testPerformance3D testPerformanceTunable
=======
	-rm -f *.o *.err *.out *.gch testSequentialValidation1D testSequentialValidation3D testSequentialValidationTunable testPerformance1D testPerformance3D testPerformanceTunable
>>>>>>> 38e9d69... Added all the changes and files that I must have forgot to back when debugging and running this last week on Blue Waters.
