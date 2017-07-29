CCMPI = mpicxx
CFLAGS = -g -Wall -O0 -std=c++11

OBJS1 = cholesky.o
OBJS2 = main.o
all: cholesky

cholesky: $(OBJS1) $(OBJS2)
	$(CCMPI) $(CFLAGS) -o cholesky $(OBJS2) /home/hutter2/hutter2/ExternalLibraries/SCALAPACK/scalapack-2.0.2/libscalapack.a /home/hutter2/hutter2/ExternalLibraries/LAPACK/lapack-3.7.1/liblapacke.a /home/hutter2/hutter2/ExternalLibraries/BLAS/OpenBLAS/libopenblas_haswellp-r0.3.0.dev.a -lgfortran

main.o: main.cpp cholesky.h
	$(CCMPI) $(CFLAGS) -c main.cpp

cholesky.o: cholesky.h choleskyImp.h ./../../scalapack/scalapackCholesky.h
	$(CCMPI) $(CFLAGS) -c cholesky.h

clean:
	-rm -f *.o *.gch cholesky $(OBJS2)
