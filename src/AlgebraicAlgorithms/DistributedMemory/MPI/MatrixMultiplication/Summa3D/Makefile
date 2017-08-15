CCMPI = g++  #mpicxx
CFLAGS = -g -Wall -O0 -std=c++11

OBJS1 = Summa3D.o
OBJS2 = test.o

all: Summa3D

Summa3D: $(OBJS2)
	$(CCMPI) $(CFLAGS) -o Summa3D $(OBJS2)

test.o: test.cpp Summa3D.h
	$(CCMPI) $(CFLAGS) -o test.o -c test.cpp

Summa3D.o: Summa3D.h Summa3D.hpp
	$(CCMPI) $(CFLAGS) -o Summa3D.o -c Summa3D.h

clean:
	-rm -f *.o *.gch Summa3D
