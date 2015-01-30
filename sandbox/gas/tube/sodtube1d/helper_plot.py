#
# helper_plot.py
#
# Description:
#   provide many helper functions to plot and show the input solutions.
#

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


def get_gas_status_plot(solution):
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
    plt.subplot(311)
    artist_rho = plt.scatter(list_x, list_rho, color='y')
    plt.subplot(312)
    artist_v = plt.scatter(list_x, list_v, color='g')
    plt.subplot(313)
    artist_p = plt.scatter(list_x, list_p, color='b')
    return (artist_rho, artist_v, artist_p)
