all: cqr2_WS_5_1_256_64_0_1_1_1 cqr2_WS_5_1_256_64_0_1_1_1_numerics cqr2_WS_5_1_128_32_0_1_1_1 cqr2_WS_5_1_128_32_0_1_1_1_numerics


cqr2_WS_5_1_256_64_0_1_1_1: cqr2_WS_5_1_256_64_0_1_1_1.pdf
cqr2_WS_5_1_256_64_0_1_1_1_numerics: cqr2_WS_5_1_256_64_0_1_1_1_numerics.pdf
cqr2_WS_5_1_128_32_0_1_1_1: cqr2_WS_5_1_128_32_0_1_1_1.pdf
cqr2_WS_5_1_128_32_0_1_1_1_numerics: cqr2_WS_5_1_128_32_0_1_1_1_numerics.pdf

cqr2_WS_5_1_256_64_0_1_1_1.pdf:
	-gnuplot ../../myPictures/Plots/benchQR1_2018-04-21-16_29_43_PORTER/cqr2_WS_5_1_256_64_0_1_1_1.plot
	-epstopdf cqr2_WS_5_1_256_64_0_1_1_1.eps
	-rm cqr2_WS_5_1_256_64_0_1_1_1.eps
	-mv cqr2_WS_5_1_256_64_0_1_1_1.pdf ../../myPictures/Plots/benchQR1_2018-04-21-16_29_43_PORTER

cqr2_WS_5_1_256_64_0_1_1_1_numerics.pdf:
	-gnuplot ../../myPictures/Plots/benchQR1_2018-04-21-16_29_43_PORTER/cqr2_WS_5_1_256_64_0_1_1_1_numerics.plot
	-epstopdf cqr2_WS_5_1_256_64_0_1_1_1_numerics.eps
	-rm cqr2_WS_5_1_256_64_0_1_1_1_numerics.eps
	-mv cqr2_WS_5_1_256_64_0_1_1_1_numerics.pdf ../../myPictures/Plots/benchQR1_2018-04-21-16_29_43_PORTER

cqr2_WS_5_1_128_32_0_1_1_1.pdf:
	-gnuplot ../../myPictures/Plots/benchQR1_2018-04-21-16_29_43_PORTER/cqr2_WS_5_1_128_32_0_1_1_1.plot
	-epstopdf cqr2_WS_5_1_128_32_0_1_1_1.eps
	-rm cqr2_WS_5_1_128_32_0_1_1_1.eps
	-mv cqr2_WS_5_1_128_32_0_1_1_1.pdf ../../myPictures/Plots/benchQR1_2018-04-21-16_29_43_PORTER

cqr2_WS_5_1_128_32_0_1_1_1_numerics.pdf:
	-gnuplot ../../myPictures/Plots/benchQR1_2018-04-21-16_29_43_PORTER/cqr2_WS_5_1_128_32_0_1_1_1_numerics.plot
	-epstopdf cqr2_WS_5_1_128_32_0_1_1_1_numerics.eps
	-rm cqr2_WS_5_1_128_32_0_1_1_1_numerics.eps
	-mv cqr2_WS_5_1_128_32_0_1_1_1_numerics.pdf ../../myPictures/Plots/benchQR1_2018-04-21-16_29_43_PORTER


clean:
	-rm *.pdf
