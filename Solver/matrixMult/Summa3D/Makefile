CCMPI = mpicxx
CFLAGS = -g -Wall -O0 -std=c++11

OBJS1 = matrixMult.o
OBJS2 = main.o

all: matrixMult

matrixMult: $(OBJS1) $(OBJS2)
#	$(CCMPI) $(CFLAGS) -o matrixMult $(OBJS2) /home/hutter2/hutter2/ExternalLibraries/SCALAPACK/scalapack-2.0.2/libscalapack.a /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapacke.a /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran
	$(CCMPI) $(CFLAGS) -o matrixMult $(OBJS2) /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran

main.o: main.cpp matrixMult.h
	$(CCMPI) $(CFLAGS) -c main.cpp

matrixMult.o: matrixMult.h matrixMultImp.h ./../../scalapack/scalapackMatrixMult.h
	$(CCMPI) $(CFLAGS) -c matrixMult.h

clean:
	-rm -f *.o *.gch matrixMult $(OBJS2)
