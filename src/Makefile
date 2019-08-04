all:
	make -C./bench/qr/ cacqr2
	make -C./bench/cholesky/ cholinv
	make -C./bench/matmult/ summa_gemm
	make -C./bench/matmult/ summa_trmm

cacqr2:
	make -C./bench/qr/ cacqr2

cholinv:
	make -C./bench/cholesky/ cholinv

summa_gemm:
	make -C./bench/matmult/ summa_gemm

summa_trmm:
	make -C./bench/matmult/ summa_trmm

clean:
	make -C./bench/qr/ clean
	make -C./bench/cholesky/ clean
	make -C./bench/matmult/ clean
