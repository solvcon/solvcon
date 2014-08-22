#!/usr/bin/python
#
# run3.py
#
# Usage:
#     ./run3.py
#
# Description:
#     An example to show how to use sodtube1d class


import sodtube1d

sodtube = sodtube1d.SodTube()
sodtube.gen_mesh()
mesh = sodtube.get_mesh()

solution_analytic = sodtube.cal_analytic_solution(mesh)
solution_cese = sodtube.get_cese_solution()

dm = sodtube1d.DataManager()

#solution_deviation = dm.get_deviation(solution_analytic, solution_cese)
#solution_deviation_percent = dm.get_deviation_percent(solution_analytic, solution_cese)

#dm.dump_solution(solution_analytic)
#dm.dump_solution(solution_cese)
#dm.dump_solution(solution_deviation)
#dm.dump_solution(solution_deviation_percent)
dm.get_plot_solutions_fig_rho(solution_analytic, solution_cese, "analytic rho", "cese rho")                                                                                                                  
dm.get_plot_solutions_fig_v(solution_analytic, solution_cese, "analytic v", "cese v")
dm.get_plot_solutions_fig_p(solution_analytic, solution_cese, "analytic p", "cese p")
dm.show_solution_comparison()
