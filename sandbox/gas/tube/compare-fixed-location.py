#!/usr/bin/env python
# -*- coding: UTF-8 -*-
#
# Copyright (c) 2017, Taihsiang Ho <tai271828@gmail.com>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
"""
Compare the npy data dumped by the probe hook of 3D tube example and the data
of 1D Sod tube analytic solution.

This script creates a plot to compare the associated derived at certain
location.

Usage:
    $ ./go run
    $ ./compare-fixed-location.py

Use the driving script go to generate data and then create the plot with this
script.
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
    # note the coordination of analytic solution is different from the one that
    # SOLVCON used.
    mesh_1d = (location,)
    analytic_solver = analytic.Solver()
    vals = list()
    for t_step in arr_time:
        solution_analytic = analytic_solver.get_analytic_solution(mesh_1d,
                                                                  t=t_step)
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
    for index in xrange(len(numeric)):
        if index == 0:
            deviation.append(numeric[index])
        else:
            deviation.append(numeric[index] - analytic[index])
    return tuple(deviation)


def der_plot(der, deviation=False):
    """

    :param der: derived data type index, an integer
    :param deviation: if want to show deviation value? True for yes.
    :return: None
    """
    if deviation:
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = \
            plt.subplots(3, 3, sharex='col', sharey='row')
    else:
        f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = \
            plt.subplots(2, 3, sharex='col', sharey='row')

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


# der_plot(1)
der_plot(1, deviation=True)
# der_plot(2)
der_plot(2, deviation=True)

plt.show()
