CCMPI = mpicxx
CFLAGS = -g -Wall -O0 -std=c++11

OBJS1 = CholeskyRecursive3DSolver.o
OBJS2 = CholeskyRecursive3DMain.o
all: CholeskyRecursive3D

CholeskyRecursive3D: $(OBJS2)
	$(CCMPI) $(CFLAGS) -o CholeskyRecursive3D $(OBJS2) ./../../scalapack-2.0.2/libscalapack.a ./../../lapack/lapack-3.4.2/liblapack.a ./../../OpenBLAS/libopenblas.a -lgfortran

CholeskyRecursive3DMain.o: CholeskyRecursive3DMain.cpp CholeskyRecursive3DSolver.h
	$(CCMPI) $(CFLAGS) -c CholeskyRecursive3DMain.cpp

CholeskyRecursive3DSolver.o: CholeskyRecursive3DSolver.h CholeskyRecursive3DSolverImp.h scalapackCholesky.h
	$(CCMPI) $(CFLAGS) -c CholeskyRecursive3DSolver.h ./../../OpenBLAS/libopenblas.a ./../../scalapack-2.0.2/libscalapack.a

clean:
	-rm -f *.o CholeskyRecursive3D $(OBJS2)
