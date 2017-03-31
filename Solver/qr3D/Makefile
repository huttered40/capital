CCMPI = mpicxx
CFLAGS = -g -Wall -O0 -std=c++11

OBJS1 = qr3D.o
OBJS2 = main.o
all: qr3D

qr3D: $(OBJS1) $(OBJS2)
	$(CCMPI) $(CFLAGS) -o qr3D $(OBJS2) ./../../scalapack-2.0.2/libscalapack.a ./../../lapack/lapack-3.4.2/liblapack.a ./../../OpenBLAS/libopenblas.a -lgfortran

main.o: main.cpp qr3D.h
	$(CCMPI) $(CFLAGS) -c main.cpp

qr3D.o: qr3D.h qr3DImp.h scalapackSolver.h
	$(CCMPI) $(CFLAGS) -c qr3D.h ./../../scalapack-2.0.2/libscalapack.a ./../../lapack/lapack-3.4.2/liblapack.a ./../../OpenBLAS/libopenblas.a -lgfortran

clean:
	-rm -f *.o qr3D $(OBJS2)
