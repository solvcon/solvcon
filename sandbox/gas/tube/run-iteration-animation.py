#!/usr/bin/python
#
# run-animation.py
#
# Usage:
#     ./run-animation.py
#
# Description:
#     An example to show the iteration animation

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
ims = []
# iterate
for it_nb in range(0, 100):
    solver_cese._data.it_nb = it_nb
    solver_cese.cal_cese_status_before_half_dt()
    solver_cese.cal_cese_status_after_half_dt()
    solver_cese.push_status_along_t()
    # plot the status
    solver_cese.data.refresh_solution()
    solution = list(solver_cese.data.solution)
    ims.append((helper_plot.get_gas_status_plot(solution), ))

im_ani = animation.ArtistAnimation(fig,
                                   ims,
                                   interval=25,
                                   repeat_delay=3000,
                                   blit=True)

plt.show()
