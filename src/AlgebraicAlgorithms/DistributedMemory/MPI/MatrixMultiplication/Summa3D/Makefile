CCMPI = g++  #mpicxx
CFLAGS = -g -Wall -O0 -std=c++11

OBJS1 = MatrixMultiplication3D.o
OBJS2 = test3D.o

all: MatrixMultiplication3D

MatrixMultiplication3D: $(OBJS2)
	$(CCMPI) $(CFLAGS) -o MatrixMultiplication3D $(OBJS2)

test3D.o: test3D.cpp MatrixMultiplication3D.h
	$(CCMPI) $(CFLAGS) -o test3D.o -c test3D.cpp

MatrixMultiplication3D.o: MatrixMultiplication3D.h MatrixMultiplication3D.hpp
	$(CCMPI) $(CFLAGS) -o MatrixMultiplication3D.o -c MatrixMultiplication3D.h

clean:
	-rm -f *.o *.gch MatrixMultiplication3D
