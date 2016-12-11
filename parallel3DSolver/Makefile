CCMPI = mpicxx
CFLAGS = -g -Wall

OBJS1 = solver.o
OBJS2 = main.o
all: solver

solver: $(OBJS2)
	$(CCMPI) $(CFLAGS) -o solver $(OBJS2) ./../OpenBLAS/libopenblas.a

main.o: main.cpp solver.h
	$(CCMPI) $(CFLAGS) -c main.cpp

solver.o: solver.h solverImp.h
	$(CCMPI) $(CFLAGS) -c solver.cpp ./../OpenBLAS/libopenblas.a

clean:
	-rm -f *.o main $(OBJS2)
