CCMPI = g++  #mpicxx
CFLAGS = -g -Wall -O0 -std=c++11

OBJS1 = Matrix.o
OBJS2 = test.o
OBSJ3 = MatrixDistribution.o

all: Matrix

Matrix: $(OBJS2)
	$(CCMPI) $(CFLAGS) -o Matrix $(OBJS2)

test.o: test.cpp Matrix.h
	$(CCMPI) $(CFLAGS) -o test.o -c test.cpp

Matrix.o: Matrix.h Matrix.hpp MatrixDistribution.h
	$(CCMPI) $(CFLAGS) -o Matrix.o -c Matrix.h

MatrixDistribution.o: MatrixDiatribution.h MatrixDistribution.hpp
	$(CCMPI) $(CFLAGS) -o MatrixDistribution.o -c MatrixDistribution.h

clean:
	-rm -f *.o *.gch Matrix
