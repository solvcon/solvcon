"""
Compare the results between CESE solver and Analytic solver
                         or two different CESE solvers
Editor: You-Hao Chang
2016/01/09

Input: two text files which contain information of Sod shock tube:
       FORMAT => position density velocity pressure
       Two directories -> result1 and result2
       You can simply put your text files under one of those directories.
       And the one you want to compare with will be put in the other directory

       IMPORTANT: The name of the file will be referred to the iteration number
                  or the time stamp. For example: 001.dat, 002.dat and etc.

Command: python results_comparison.py [DIR_1] [DIR_2]

"""

import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Global variables
ITERATION = 100 # time stamp
DIRECTORY_1 = sys.argv[1]
DIRECTORY_2 = sys.argv[2]

# starting to extract the inforamtion of input files 
file_idx = 1
while file_idx <= ITERATION:
    file1 = open(DIRECTORY_1 + "/" + "%03d" % file_idx + ".dat", 'r')
    file2 = open(DIRECTORY_2 + "/" + "%03d" % file_idx + ".dat", 'r')

    position = []
    diff_in_density = []
    diff_in_velocity = []
    diff_in_pressure = []
    
    while True:
        line1 = file1.readline().split()
        line2 = file2.readline().split()
        if not line1 or not line2: break

        position.append(float(line1[0]))
        #diff_in_density.append(float(line1[1]) - float(line2[1]))
        #diff_in_velocity.append(float(line1[2]) - float(line2[2]))
        #diff_in_pressure.append(float(line1[3]) - float(line2[3]))
        diff_in_density.append((float(line1[1]) - float(line2[1])) / float(line1[1]) * 100 if float(line1[1]) > 0. else 0.)
        diff_in_velocity.append((float(line1[2]) - float(line2[2])) / float(line1[2]) * 100 if float(line1[2]) > 0. else 0.)
        diff_in_pressure.append((float(line1[3]) - float(line2[3])) / float(line1[3]) * 100 if float(line1[3]) > 0. else 0.)

    fig = plt.figure(figsize=(9, 7))
    plt.subplot(311)
    plot_rho = plt.plot(position, diff_in_density, color="r")
    plt.xlabel("x")
    plt.ylabel("Diff. in density [%]")
    plt.xlim(-0.505, 0.505)
    plt.ylim(min(diff_in_density) - 3, max(diff_in_density) + 3)
    plt.subplot(312)
    plot_vel = plt.plot(position, diff_in_velocity, color="g")
    plt.xlabel("x")
    plt.ylabel("Diff. in velocity [%]")
    plt.xlim(-0.505, 0.505)
    plt.ylim(min(diff_in_velocity) - 3, max(diff_in_velocity) + 3)
    plt.subplot(313)
    plot_p = plt.plot(position, diff_in_pressure, color="b")
    plt.xlabel("x")
    plt.ylabel("Diff. in pressure [%]")
    plt.xlim(-0.505, 0.505)
    plt.ylim(min(diff_in_pressure) - 3, max(diff_in_pressure) + 3)
    plt.show()

    pdf_output = PdfPages("Diff_" + "%03d" % file_idx + ".pdf")
    pdf_output.savefig(fig)
    pdf_output.close()

    plt.close()

    file1.close()
    file2.close()
    file_idx += 1
    


