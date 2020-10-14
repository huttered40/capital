include ../../config.mk

ALG=$(HOME)/camfs/src/alg/cholesky/cholinv/
OBJS1 = baseline_variants
OBJS2 = baseline_search
OBJS4 = tuning_test1

all: $(OBJS1) $(OBJS2) $(OBJS4)

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

$(OBJS4): $(OBJS4).o
	$(CCMPI) $(CFLAGS) -o $(BIN)autotune/cholinv_$(OBJS4) $(OBJS4).o $(LIB_PATH) $(LIBS)
	rm *.o

$(OBJS4).o: cholinv/$(OBJS4).cpp $(ALG)cholinv.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS4).o -c cholinv/$(OBJS4).cpp

clean:
	-rm -f *.o *.err *.out *.gch $(BIN)autotune/cholinv_$(OBJS1) $(BIN)autotune/cholinv_$(OBJS2) $(BIN)autotune/cholinv_$(OBJS4)
