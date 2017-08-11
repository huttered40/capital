CCMPI = g++  #mpicxx
CFLAGS = -g -Wall -O0 -std=c++11

OBJS1 = Matrix.o
OBJS2 = test.o
OBSJ3 = MatrixDistributer.o

all: Matrix

Matrix: $(OBJS2)
	$(CCMPI) $(CFLAGS) -o Matrix $(OBJS2)

test.o: test.cpp Matrix.h
	$(CCMPI) $(CFLAGS) -o test.o -c test.cpp

Matrix.o: Matrix.h Matrix.hpp MatrixDistributer.h MatrixAllocator.h MatrixSerializer.h
	$(CCMPI) $(CFLAGS) -o Matrix.o -c Matrix.h

MatrixDistributer.o: MatrixDiatributer.h MatrixDistributer.hpp
	$(CCMPI) $(CFLAGS) -o MatrixDistributer.o -c MatrixDistributer.h

MatrixAllocator.o: MatrixAllocator.h MatrixAllocator.hpp
	$(CCMPI) $(CFLAGS) -o MatrixAllocator.o -c MatrixAllocator.h

MatrixSerializer.o: MatrixSerializer.h MatrixSerializer.hpp
	$(CCMPI) $(CFLAGS) -o MatrixSerializer.o -c MatrixSerializer.h

clean:
	-rm -f *.o *.gch Matrix
