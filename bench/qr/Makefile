include ../../config.mk

ALG=$(HOME)/capital/src/alg/qr/cacqr/
OBJS1 = cacqr
$(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)bench/$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)
	rm *.o
$(OBJS1).o:# cacqr.cpp $(ALG)cacqr.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS1).o -c $(OBJS1).cpp
clean:
	-rm -f *.o *.gch $(BIN)bench/$(OBJS1)
