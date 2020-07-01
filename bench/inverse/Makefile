include ../../config.mk

ALG=$(HOME)/camfs/src/alg/inverse/rectri/
OBJS1 = rectri
$(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)
	rm *.o
$(OBJS1).o: rectri.cpp
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS1).o -c rectri.cpp
clean:
	-rm -f *.o *.err *.out *.gch $(BIN)$(OBJS1)
