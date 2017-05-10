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

This script creates a plot to compare the associated derived at certain time.

Usage:
    $ ./go run
    $ ./compare-fixed-time.py

Use the driving script go to generate data and then create the plot with this
script.
"""

import matplotlib.pyplot as plt
import numpy as np
import sodtube1d.solver_analytic as analytic


def load_from_np(filename, arr_idx_der):
    """
    arr_idx_der 1 for rho and 2 for p
    """
    # load npy data of 3D tube
    arr = np.load(filename)

    arr_t = arr[:, 0]
    arr_der = arr[:, arr_idx_der]

    return arr_t, arr_der


def load_from_analytic(location, arr_time, arr_idx_der):
    # load analytic solution data of 1D tube
    # note the coordination of analytic solution is different from
    # the one that SOLVCON used.
    mesh_1d = (location,)
    analytic_solver = analytic.Solver()
    # parse the analytic data object to be arr type data object
    vals = list()
    for t_step in arr_time:
        solution_analytic = analytic_solver.get_analytic_solution(mesh_1d,
                                                                  t=t_step)
        # refer to gas probe.py:Probe::__call__
        vlist = [t_step, solution_analytic[0][1], solution_analytic[0][3]]
        vals.append(vlist)
    arr_aly = np.array(vals, dtype='float64')
    # parsed.

    arr_aly_t = arr_time
    arr_aly_der = arr_aly[:, arr_idx_der]

    return arr_aly_t, arr_aly_der


def get_deviation(numeric, analytic):
    deviation = list()
    for index in xrange(len(numeric)):
        if index == 0:
            deviation.append(numeric[index])
        else:
            deviation.append(numeric[index] - analytic[index])
    return tuple(deviation)


def der_plots(step):
    """
    :param step: integeter, nth step
    """
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = \
        plt.subplots(3, 3, sharex='col', sharey='row')
    t = der_plot(step, 1, f, ax1, ax4, ax7)
    der_plot(step, 2, f, ax2, ax5, ax8)
    der_plot(step, 3, f, ax3, ax6, ax9)
    f.suptitle("SOLVCON 3D CESE vs. 1D ANALYTICAL - " + t + ' sec.')


def der_plot(step, der_index, f, ax1, ax2, ax3):

    np_filenames = []
    locations = []
    for idx in range(99):
        np_filenames.append("tube_pt_ppank_" + str(idx) + ".npy")
        locations.append(idx/100.0)

    data_der = []
    for np_filename in np_filenames:
        arr_t, arr_der = load_from_np("./result/" + np_filename, der_index)
        data_der.append(arr_der[step])

    # shift locations so we could input it in analytic coordination.
    locations_aly = []
    for location in locations:
        location_aly = location - 0.5
        locations_aly.append(location_aly)

    analytic_solver = analytic.Solver()
    analytic_solution = analytic_solver.get_analytic_solution(locations_aly,
                                                              t=arr_t[step])
    data_der_analytic = []
    for location_idx in range(len(locations)):
        if der_index == 1:
            title = 'RHO'
        elif der_index == 2:
            title = 'M'
        elif der_index == 3:
            title = 'PRESSURE'
        else:
            title = 'UNKNOWN'
        data_der_analytic.append(analytic_solution[location_idx][der_index])

    data_der_deviation = []
    for idx in range(len(data_der)):
        data_der_deviation.append(data_der[idx] - data_der_analytic[idx])

    ax1.set_title('3D - ' + title)
    ax1.scatter(locations, data_der, s=10, c='b', marker='s')
    ax2.set_title('1D - ' + title)
    ax2.scatter(locations, data_der_analytic, s=10, c='b', marker='8')
    ax3.set_title('DEVIATION')
    ax3.scatter(locations, data_der_deviation, s=10, c='b', marker='o')

    return str(arr_t[step])


# CHANGE ME: Give the specific time step index number
der_plots(286)

plt.show()
