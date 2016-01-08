#!/usr/bin/python
#
# run-dump-status.py
#
# Usage:
#     ./run-dump-status.py
#
# Description:
#     will generate many file with the format <iteration number>.dat

import sodtube1d.solver_cese as cese
import sodtube1d.helper_plot as helper_plot
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# initialize the solver
# initialize the gas status
solver_cese = cese.Solver()
solver_cese.init_gas_status()
solver_cese._data.it_pt_nb = 2

fig = plt.figure()
frame_seq = []

def dump_to_file(f, solution):
    for item in solution:
        output = ""
        for ele in item:
            output = output + str(ele) + " "
        output = output.strip()
        output  = output + "\n"
        f.write(output)

# iterate
for it_nb in range(0, 100):
    solver_cese._data.it_nb = it_nb
    solver_cese.cal_cese_status_before_half_dt()
    solver_cese.cal_cese_status_after_half_dt()
    solver_cese.push_status_along_t()
    # plot the status
    solver_cese.data.refresh_solution()
    solution = list(solver_cese.data.solution)
    filename = "%03d" % it_nb
    f = open(str(filename) + ".dat", "w")
    dump_to_file(f, solution)
    f.close()

