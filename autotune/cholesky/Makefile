include ../../config.mk

ALG=$(HOME)/camfs/src/alg/cholesky/cholinv/
OBJS1 = baseline
OBJS2 = space1
OBJS3 = space2

all: $(OBJS1) $(OBJS2) $(OBJS3)

$(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)autotune/cholinv_$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)
	rm *.o

$(OBJS1).o: cholinv/$(OBJS1).cpp $(ALG)cholinv.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS1).o -c cholinv/$(OBJS1).cpp

$(OBJS2): $(OBJS2).o
	$(CCMPI) $(CFLAGS) -o $(BIN)autotune/cholinv_$(OBJS2) $(OBJS2).o $(LIB_PATH) $(LIBS)
	rm *.o

$(OBJS2).o: cholinv/$(OBJS2).cpp $(ALG)cholinv.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS2).o -c cholinv/$(OBJS2).cpp

$(OBJS3): $(OBJS3).o
	$(CCMPI) $(CFLAGS) -o $(BIN)autotune/cholinv_$(OBJS3) $(OBJS3).o $(LIB_PATH) $(LIBS)
	rm *.o

$(OBJS3).o: cholinv/$(OBJS3).cpp $(ALG)cholinv.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS3).o -c cholinv/$(OBJS3).cpp

clean:
	-rm -f *.o *.err *.out *.gch $(BIN)autotune/cholinv_$(OBJS1) $(BIN)autotune/cholinv_$(OBJS2) $(BIN)autotune/cholinv_$(OBJS3)
