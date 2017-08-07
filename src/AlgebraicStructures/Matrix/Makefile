CCMPI = g++  #mpicxx
CFLAGS = -g -Wall -O0 -std=c++11

OBJS1 = Matrix.o
OBJS2 = test.o

all: Matrix

Matrix: $(OBJS2)
	$(CCMPI) $(CFLAGS) -o Matrix $(OBJS2)

test.o: test.cpp Matrix.h
	$(CCMPI) $(CFLAGS) -o test.o -c test.cpp

Matrix.o: Matrix.h Matrix.hpp
	$(CCMPI) $(CFLAGS) -o Matrix.o -c Matrix.h

clean:
	-rm -f *.o *.gch Matrix
