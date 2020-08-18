all:
	make -C./autotune/cholesky/ all
	make -C./bench/qr/ cacqr
	make -C./bench/cholesky/ cholinv
	make -C./bench/inverse/ rectri
	make -C./bench/matmult/ summa_gemm

bench:
	make -C./bench/qr/ cacqr
	make -C./bench/cholesky/ cholinv
	make -C./bench/inverse/ rectri
	make -C./bench/matmult/ summa_gemm
	
autotune:
	make -C./autotune/cholesky/ all

cacqr:
	make -C./bench/qr/ cacqr

cholinv:
	make -C./autotune/cholesky/ all
	make -C./bench/cholesky/ cholinv

rectri:
	make -C./bench/inverse/ rectri

summa_gemm:
	make -C./bench/matmult/ summa_gemm


clean:
	make -C./bench/qr/ clean
	make -C./bench/cholesky/ clean
	make -C./bench/inverse/ clean
	make -C./bench/matmult/ clean
