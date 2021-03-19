include ../../config.mk

ALG=$(HOME)/capital/src/alg/cholesky/cholinv/
OBJS1 = tune

all: $(OBJS1)

$(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)autotune/cholinv_$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)
	rm *.o

$(OBJS1).o: cholinv/$(OBJS1).cpp $(ALG)cholinv.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS1).o -c cholinv/$(OBJS1).cpp

clean:
	-rm -f *.o *.err *.out *.gch $(BIN)autotune/cholinv_$(OBJS1)
