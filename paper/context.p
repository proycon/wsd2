# Gnuplot script file for plotting data in file "force.dat"
# This file is called   force.p
set terminal pngcairo size 640,480 enhanced font 'Droid Sans,10'
set   autoscale                        # scale axes automatically
unset log                              # remove any log-scaling
unset label                            # remove any previous labels
set xrange [1:3]
set xtics (1,2,3)
set yrange [14:26]
set ytic auto                          # set ytics automatically
#set title "Average Accuracy for different local context sizes"
set xlabel "Local context size"
set ylabel "Average accuracy"
plot    "context.dat" using 1:2 title 'Dutch' with linespoints , \
    	"context.dat" using 1:3 title 'Spanish' with linespoints, \
    	"context.dat" using 1:4 title 'Italian' with linespoints, \
    	"context.dat" using 1:5 title 'German' with linespoints, \
    	"context.dat" using 1:6 title 'French' with linespoints
