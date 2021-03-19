include ../../config.mk

ALG=$(HOME)/capital/src/alg/qr/cacqr/
OBJS1 = tune

all: $(OBJS1)

$(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)autotune/cacqr_$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)
	rm *.o

$(OBJS1).o: cacqr/$(OBJS1).cpp $(ALG)cacqr.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS1).o -c cacqr/$(OBJS1).cpp

clean:
	-rm -f *.o *.err *.out *.gch $(BIN)autotune/cacqr_$(OBJS1)
