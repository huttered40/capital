CCMPI = mpicxx
CFLAGS = -g -Wall -O0 -std=c++11

OBJS1 = qr.o
OBJS2 = main.o
all: qr

qr: $(OBJS1) $(OBJS2)
	$(CCMPI) $(CFLAGS) -o qr $(OBJS2) ./../../../scalapack-2.0.2/libscalapack.a ./../../../lapack/lapack-3.4.2/liblapack.a ./../../../OpenBLAS/libopenblas.a -lgfortran

main.o: main.cpp qr.h
	$(CCMPI) $(CFLAGS) -c main.cpp

qr.o: qr.h qrImp.h ./../../scalapack/scalapackQR.h
	$(CCMPI) $(CFLAGS) -c qr.h

clean:
	-rm -f *.o *.gch qr $(OBJS2)
