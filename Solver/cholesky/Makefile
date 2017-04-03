CCMPI = mpicxx
CFLAGS = -g -Wall -O0 -std=c++11

OBJS1 = cholesky.o
OBJS2 = main.o
all: cholesky

cholesky: $(OBJS1) $(OBJS2)
	$(CCMPI) $(CFLAGS) -o cholesky $(OBJS2) ./../../scalapack-2.0.2/libscalapack.a ./../../lapack/lapack-3.4.2/liblapack.a ./../../OpenBLAS/libopenblas.a -lgfortran

main.o: main.cpp cholesky.h
	$(CCMPI) $(CFLAGS) -c main.cpp

cholesky.o: cholesky.h choleskyImp.h scalapackCholesky.h
	$(CCMPI) $(CFLAGS) -c cholesky.h

clean:
	-rm -f *.o *.gch cholesky $(OBJS2)
