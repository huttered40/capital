include ../../config.mk

ALG=$(HOME)/camfs/src/alg/matmult/summa/
OBJS1 = summa_gemm
$(OBJS1): $(OBJS1).o
	$(CCMPI) $(CFLAGS) -o $(BIN)$(OBJS1) $(OBJS1).o $(LIB_PATH) $(LIBS)
	rm *.o
$(OBJS1).o: summa_gemm.cpp $(ALG)summa.h
	$(CCMPI) $(CFLAGS) -D$(MACHINE) -o $(OBJS1).o -c summa_gemm.cpp
clean:
	-rm -f *.o *.gch $(BIN)$(OBJS1)
