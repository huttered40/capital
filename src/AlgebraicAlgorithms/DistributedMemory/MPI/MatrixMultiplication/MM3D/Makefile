CCMPI = /home/hutter2/hutter2/ExternalLibraries/MPICH/installDir/bin/mpicxx
CFLAGS = -g -Wall -O3 -std=c++11

OBJS1 = MM3D.o
OBJS2 = testPerformance.o
OBJS3 = testSequentialValidationGemm.o
OBJS4 = testSequentialValidationTrmm.o
OBJS5 = testSequentialValidationSyrk.o

all: testPerformance testSequentialValidationGemm testSequentialValidationTrmm testSequentialValidationSyrk

testPerformance: $(OBJS2)
	$(CCMPI) $(CFLAGS) -o testPerformance $(OBJS2) /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lpthread

testSequentialValidationGemm: $(OBJS3)
	$(CCMPI) $(CFLAGS) -o testSequentialValidationGemm $(OBJS3) /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lpthread

testSequentialValidationTrmm: $(OBJS4)
	$(CCMPI) $(CFLAGS) -o testSequentialValidationTrmm $(OBJS3) /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lpthread

testSequentialValidationSyrk: $(OBJS5)
	$(CCMPI) $(CFLAGS) -o testSequentialValidationSyrk $(OBJS3) /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lpthread

testPerformance.o: testPerformance.cpp MM3D.h
	$(CCMPI) $(CFLAGS) -o testPerformance.o -c testPerformance.cpp

testSequentialValidationGemm.o: testSequentialValidationGemm.cpp MM3D.h
	$(CCMPI) $(CFLAGS) -o testSequentialValidationGemm.o -c testSequentialValidationGemm.cpp

testSequentialValidationTrmm.o: testSequentialValidationTrmm.cpp MM3D.h
	$(CCMPI) $(CFLAGS) -o testSequentialValidationTrmm.o -c testSequentialValidationTrmm.cpp

testSequentialValidationSyrk.o: testSequentialValidationSyrk.cpp MM3D.h
	$(CCMPI) $(CFLAGS) -o testSequentialValidationSyrk.o -c testSequentialValidationSyrk.cpp

MM3D.o: MM3D.h MM3D.hpp
	$(CCMPI) $(CFLAGS) -o MM3D.o -c MM3D.h

clean:
	-rm -f *.o *.gch testPerformance testSequentialValidationGemm testSequentialValidationTrmm testSequentialValidationSyrk
