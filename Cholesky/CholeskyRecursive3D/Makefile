CCMPI = mpicxx
CFLAGS = -g -Wall -O0 -std=c++11

OBJS1 = CholeskyRecursive3DSolver.o
OBJS2 = CholeskyRecursive3DMain.o
all: CholeskyRecursive3D

CholeskyRecursive3D: $(OBJS2)
	$(CCMPI) $(CFLAGS) -o CholeskyRecursive3D $(OBJS2) -lgfortran ./../../OpenBLAS/libopenblas.a ./../../scalapack-2.0.2/libscalapack.a

CholeskyRecursive3DMain.o: CholeskyRecursive3DMain.cpp CholeskyRecursive3DSolver.h
	$(CCMPI) $(CFLAGS) -c CholeskyRecursive3DMain.cpp

CholeskyRecursive3DSolver.o: CholeskyRecursive3DSolver.h CholeskyRecursive3DSolverImp.h
	$(CCMPI) $(CFLAGS) -c CholeskyRecursive3DSolver.h ./../../OpenBLAS/libopenblas.a ./../../scalapack-2.0.2/libscalapack.a

clean:
	-rm -f *.o CholeskyRecursive3D $(OBJS2)
