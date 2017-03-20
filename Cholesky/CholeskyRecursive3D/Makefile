CCMPI = mpicxx
CFLAGS = -g -Wall -O0

OBJS1 = solver.o
OBJS2 = main.o
all: solver

solver: $(OBJS2)
	$(CCMPI) $(CFLAGS) -o solver $(OBJS2) -lgfortran ./../OpenBLAS/libopenblas.a ./../scalapack-2.0.2/libscalapack.a

main.o: main.cpp solver.h
	$(CCMPI) $(CFLAGS) -c main.cpp

solver.o: solver.h solverImp.h
	$(CCMPI) $(CFLAGS) -c solver.h ./../OpenBLAS/libopenblas.a ./../scalapack-2.0.2/libscalapack.a

clean:
	-rm -f *.o main $(OBJS2)
