all:
	make -C./alg/qr/cacqr/ MPI
	make -C./alg/cholesky/CFR3D/ MPI
	make -C./alg/matmult/MM3D/ MPI

all_ampi:
	make -C./alg/qr/cacqr/ AMPI
	make -C./alg/cholesky/CFR3D/ AMPI
	make -C./alg/matmult/MM3D/ AMPI

caqr2:
	make -C./alg/qr/cacqr/ MPI

cfr3d:
	make -C./alg/cholesky/CFR3D/ MPI

mm3d:
	make -C./alg/matmult/MM3D/ MPI

caqr2_ampi:
	make -C./alg/qr/cacqr/ AMPI

cfr3d_ampi:
	make -C./alg/cholesky/CFR3D/ AMPI

mm3d_ampi:
	make -C./alg/matmult/MM3D/ AMPI

clean:
	make -C./alg/qr/cacqr/ clean
	make -C./alg/cholesky/CFR3D/ clean
	make -C./alg/matmult/MM3D/ clean
