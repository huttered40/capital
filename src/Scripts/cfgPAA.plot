#set all these to 3 to make solid histograms
ptn1=1
ptn2=2
ptn3=4
ptn4=5
ptn5=6

set size 0.60, 0.6
set pointsize 0.8
set xtics 8
set mxtics 8
set boxwidth 1.0 relative
set style data histograms
set style histogram cluster gap 1
set style fill solid 1.0 noborder

set format x "%g"

set style line 1 lt 1 lc rgb "#008B8B" lw 5 pt 2 ps 1.2
set style line 2 lt 3 lc rgb "#66CCFF" lw 5 pt 2 ps 1.2
set style line 3 lt 5 lc rgb "#CC3300" lw 5 pt 2 ps 1.2
set style line 4 lt 11 lc rgb "#CC66CC" lw 5 pt 2 ps 1.2
set style line 5 lt 2 lc rgb "#00CC33" lw 5 pt 2 ps 1.2
set style line 6 lt 11 lc rgb "#500000" lw 5 pt 6 ps 1.2
set style line 7 lt 7 lc rgb "#005000" lw 5 pt 7 ps 1.2
set style line 8 lt 8 lc rgb "#000050" lw 5 pt 8 ps 1.2
set style line 9 lt 9 lc rgb "#050505" lw 5 pt 9 ps 1.2
set style line 11 lt 2 lc rgb "#CC0022" lw 5 pt 7 ps 1.2
set style line 12 lt 3 lc rgb "#008B8B" lw 5 pt 7 ps 1.2
#set style line 11 lt 3 lc rgb "#66CCFF" lw 5 pt 7 ps 1.2
set style line 13 lt 5 lc rgb "#CC3300" lw 5 pt 7 ps 1.2
set style line 14 lt 11 lc rgb "#CC66CC" lw 5 pt 7 ps 1.2
set style line 99 lt 2 lc rgb "gray" lw .7 pt .8
set style line 15 lt 2 lc rgb "#00CC33" lw 5 pt 7 ps 1.2
set style line 21 lt 1 lc rgb "#008B8B" lw 5 pt 4 ps 1.2
set style line 24 lt 6 lc rgb "#CC66CC" lw 5 pt 4 ps 1.2
set style line 31 lt 1 lc rgb "#008B8B" lw 5 pt 9 ps 1.2
set style line 34 lt 6 lc rgb "#CC66CC" lw 5 pt 9 ps 1.2

E8_mn = 32.876
E128_mn = 270.325
E8_m = 66781711
E128_m = 923180018

#set grid xtics mxtics ytics mytics ls 99
set terminal postscript eps enhanced color "NimbusSanL-Regu" 14
set key font ",12" spacing 1.1

flops(m,n) = 1.E-9*(2.*m*n*n-n*n*n*2./3.)
CQR2flops(m,n) = 1.E-9*(4.*m*n*n+(5./3.)*n*n*n)
