#
# helper_plot.py
#
# Description:
#   provide many helper functions to plot and show the input solutions.
#

import sys
import scipy.optimize as so
import matplotlib.pyplot as plt

# a number to claim two floating number value are equal.
delta_precision = 0.0000000000001

def show_gas_status(solution):
    """
    @ solution: a solution list, format [(x, rho, v, p)...]
    """
    # prepare list for x, rho, v and p
    list_x = []
    list_rho = []
    list_v = []
    list_p = []
    for i in solution:
        list_x.append(i[0])
        list_rho.append(i[1])
        list_v.append(i[2])
        list_p.append(i[3])
    # now plot
    plt.scatter(list_x, list_rho, color='y')
    plt.scatter(list_x, list_v, color='g')
    plt.scatter(list_x, list_p, color='b')

