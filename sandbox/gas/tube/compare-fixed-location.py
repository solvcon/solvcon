#!/usr/bin/env python
"""

Compare the npy data dumped by the probe hook of 3D tube example and the data
of 1D Sod tube analytic solution.

"""

import matplotlib.pyplot as plt
import numpy as np
import sodtube1d.solver_analytic as analytic


def load_from_np(filename):
    # load npy data of 3D tube
    arr = np.load(filename)

    arr_t = arr[:, 0]
    arr_rho = arr[:, 1]
    arr_p = arr[:, 2]

    return arr_t, arr_rho, arr_p

def load_from_analytic(location, arr_time):
    # load analytic solution data of 1D tube
    # note the coordination of analytic solution is different from the one that SOLVCON used.
    mesh_1d = (location,)
    analytic_solver = analytic.Solver()
    vals = list()
    for t_step in arr_time:
        solution_analytic = analytic_solver.get_analytic_solution(mesh_1d, t=t_step)
        # refer to gas probe.py:Probe::__call__
        vlist = [t_step, solution_analytic[0][1], solution_analytic[0][3]]
        vals.append(vlist)
    arr_aly = np.array(vals, dtype='float64')
    arr_aly_t = arr_time
    arr_aly_rho = arr_aly[:, 1]
    arr_aly_p = arr_aly[:, 2]

    return arr_aly_t, arr_aly_rho, arr_aly_p

def get_deviation(numeric, analytic):
    deviation = list()
    #import ipdb; ipdb.set_trace()
    for index in xrange(len(numeric)):
        if index == 0:
            deviation.append(numeric[index])
        else:
            deviation.append(numeric[index] - analytic[index])
    return tuple(deviation)

def der_plot(der, deviation=False):
    if deviation:
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(3, 3, sharex='col', sharey='row')
    else:
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, sharex='col', sharey='row')

    if der == 1:
        f.suptitle("DENSITY")
    elif der == 2:
        f.suptitle("PRESSURE")

    data_np_1 = load_from_np("./result/tube_pt_ppank_Left.npy")
    ax1.set_title('LEFT - SOLVCON')
    data_np_2 = load_from_np("./result/tube_pt_ppank_Diaphragm.npy")
    ax2.set_title('DIAPHRAGM - SOLVCON')
    data_np_3 = load_from_np("./result/tube_pt_ppank_Right.npy")
    ax3.set_title('RIGHT - SOLVCON')

    data_aly_1 = load_from_analytic(-0.25, data_np_1[0])
    ax4.set_title('LEFT - ANALYTIC')
    data_aly_2 = load_from_analytic(0.0, data_np_2[0])
    ax5.set_title('DIAPHRAGM - ANALYTIC')
    data_aly_3 = load_from_analytic(0.25, data_np_3[0])
    ax6.set_title('RIGHT - ANALYTIC')

    if deviation:
        data_dev_1 = get_deviation(data_np_1, data_aly_1)
        ax7.set_title('LEFT - DEVIATION')
        data_dev_2 = get_deviation(data_np_2, data_aly_2)
        ax8.set_title('DIAPHRAGM - DEVIATION')
        data_dev_3 = get_deviation(data_np_3, data_aly_3)
        ax9.set_title('RIGHT - DEVIATION')


    ax1.scatter(data_np_1[0], data_np_1[der], s=10, c='b', marker='s')
    ax2.scatter(data_np_2[0], data_np_2[der], s=10, c='b', marker='s')
    ax3.scatter(data_np_3[0], data_np_3[der], s=10, c='b', marker='s')
    ax4.scatter(data_aly_1[0], data_aly_1[der], s=10, c='r', marker="o")
    ax5.scatter(data_aly_2[0], data_aly_2[der], s=10, c='r', marker="o")
    ax6.scatter(data_aly_3[0], data_aly_3[der], s=10, c='r', marker="o")

    if deviation:
        ax7.scatter(data_dev_1[0], data_dev_1[der], s=10, c='g', marker="o")
        ax8.scatter(data_dev_2[0], data_dev_2[der], s=10, c='g', marker="o")
        ax9.scatter(data_dev_3[0], data_dev_3[der], s=10, c='g', marker="o")

#der_plot(1)
der_plot(1, deviation=True)
#der_plot(2)
der_plot(2, deviation=True)

plt.show()
