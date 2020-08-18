include ../../config.mk

ALG=$(HOME)/camfs/src/alg/cholesky/cholinv/
OBJS1 = cholinv

$(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)
	rm *.o
$(OBJS1).o: $(OBJS1).cpp $(ALG)cholinv.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS1).o -c $(OBJS1).cpp

clean:
	-rm -f *.o *.err *.out *.gch $(BIN)$(OBJS1) $(BIN)$(OBJS2) $(BIN)$(OBJS3)
