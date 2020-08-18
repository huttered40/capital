include ../../config.mk

ALG=$(HOME)/camfs/src/alg/cholesky/cholinv/
OBJS1 = cholinv_space1
OBJS2 = cholinv_space2
OBJS3 = cholinv_baseline

all: $(OBJS1) $(OBJS2) $(OBJS3)

$(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)
	rm *.o
$(OBJS1).o: $(OBJS1).cpp $(ALG)cholinv.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS1).o -c $(OBJS1).cpp

$(OBJS2): $(OBJS2).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS2) $(OBJS2).o $(LIB_PATH) $(LIBS)
	rm *.o
$(OBJS2).o: $(OBJS2).cpp $(ALG)cholinv.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS2).o -c $(OBJS2).cpp

$(OBJS3): $(OBJS3).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS3) $(OBJS3).o $(LIB_PATH) $(LIBS)
	rm *.o
$(OBJS3).o: $(OBJS3).cpp $(ALG)cholinv.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS3).o -c $(OBJS3).cpp

clean:
	-rm -f *.o *.err *.out *.gch $(BIN)$(OBJS1) $(BIN)$(OBJS2) $(BIN)$(OBJS3)
