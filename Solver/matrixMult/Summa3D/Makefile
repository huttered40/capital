CCMPI = mpicxx
CFLAGS = -g -Wall -O0 -std=c++11

OBJS1 = matrixMult.o
OBJS2 = main.o
all: matrixMult

matrixMult: $(OBJS1) $(OBJS2)
	$(CCMPI) $(CFLAGS) -o matrixMult $(OBJS2) ./../../../scalapack-2.0.2/libscalapack.a ./../../../lapack/lapack-3.4.2/liblapack.a ./../../../OpenBLAS/libopenblas.a -lgfortran

main.o: main.cpp matrixMult.h
	$(CCMPI) $(CFLAGS) -c main.cpp

matrixMult.o: matrixMult.h matrixMultImp.h ./../../scalapack/scalapackMatrixMult.h
	$(CCMPI) $(CFLAGS) -c matrixMult.h

clean:
	-rm -f *.o *.gch matrixMult $(OBJS2)
