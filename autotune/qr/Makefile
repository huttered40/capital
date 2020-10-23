include ../../config.mk

ALG=$(HOME)/camfs/src/alg/qr/cacqr/
OBJS1 = baseline_variants
OBJS2 = baseline_search
OBJS4 = tuning_test1

all: $(OBJS1) $(OBJS2) $(OBJS4)

$(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)autotune/cacqr_$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)
	rm *.o

$(OBJS1).o: cacqr/$(OBJS1).cpp $(ALG)cacqr.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS1).o -c cacqr/$(OBJS1).cpp

$(OBJS2): $(OBJS2).o
	$(CCMPI) $(CFLAGS) -o $(BIN)autotune/cacqr_$(OBJS2) $(OBJS2).o $(LIB_PATH) $(LIBS)
	rm *.o

$(OBJS2).o: cacqr/$(OBJS2).cpp $(ALG)cacqr.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS2).o -c cacqr/$(OBJS2).cpp

$(OBJS4): $(OBJS4).o
	$(CCMPI) $(CFLAGS) -o $(BIN)autotune/cacqr_$(OBJS4) $(OBJS4).o $(LIB_PATH) $(LIBS)
	rm *.o

$(OBJS4).o: cacqr/$(OBJS4).cpp $(ALG)cacqr.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS4).o -c cacqr/$(OBJS4).cpp

clean:
	-rm -f *.o *.err *.out *.gch $(BIN)autotune/cacqr_$(OBJS1) $(BIN)autotune/cacqr_$(OBJS2) $(BIN)autotune/cacqr_$(OBJS4)
