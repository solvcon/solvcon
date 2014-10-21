#!/usr/bin/python
#
# run.py
#
# Usage:
#     ./run.py
#
# Description:
#     An example to show how to use sodtube1d class


import sodtube1d

sodtube = sodtube1d.SodTube()

sodtube.gen_mesh()
mesh = sodtube.get_mesh()
solution_analytic = sodtube.get_analytic_solution(mesh)
solution_cese_fp = sodtube.get_cese_solution_fortran_porting()

dm = sodtube1d.DataManager()

#solution_deviation = dm.get_deviation(solution_analytic, solution_cese_fp)
#solution_deviation_percent = dm.get_deviation_percent(solution_analytic, solution_cese_fp)

#dm.dump_solution(solution_analytic)
#dm.dump_solution(solution_cese_fp)
#dm.dump_solution(solution_deviation)
#dm.dump_solution(solution_deviation_percent)

#dm.get_plot_solutions_fig_rho(solution_analytic, solution_cese_fp, "analytic rho", "cese rho")
#dm.get_plot_solutions_fig_v(solution_analytic, solution_cese_fp, "analytic v", "cese v")
#dm.get_plot_solutions_fig_p(solution_analytic, solution_cese_fp, "analytic p", "cese p")
#print("l2 norm is: (rho, v, p)")
#print(dm.get_l2_norm(solution_analytic, solution_cese_fp,1),
#      dm.get_l2_norm(solution_analytic, solution_cese_fp,2),
#      dm.get_l2_norm(solution_analytic, solution_cese_fp,3))
print(dm.get_l2_norm(solution_analytic, solution_cese_fp,1,[(-0.25,-0.15),(-0.05,0.05),(0.15,0.25),(0.30,0.40)]),
      dm.get_l2_norm(solution_analytic, solution_cese_fp,2,[(-0.25,-0.15),(0.30,0.40)]),
      dm.get_l2_norm(solution_analytic, solution_cese_fp,3,[(-0.25,-0.15),(-0.05,0.05),(0.30,0.40)]))

sodtube.gen_mesh(50, -5025, 5025)
mesh = sodtube.get_mesh()
solution_analytic = sodtube.get_analytic_solution(mesh)
solution_cese_fp = sodtube.get_cese_solution_fortran_porting(200, 0.002, 0.005)

dm.get_plot_solutions_fig_rho(solution_analytic, solution_cese_fp, "analytic rho (diff mesh)", "cese rho")
#dm.get_plot_solutions_fig_v(solution_analytic, solution_cese_fp, "analytic v (diff mesh)", "cese v")
#dm.get_plot_solutions_fig_p(solution_analytic, solution_cese_fp, "analytic p (diff mesh)", "cese p")
#dm.show_solution_comparison()
print(dm.get_l2_norm(solution_analytic, solution_cese_fp,1,[(-0.25,-0.15),(-0.05,0.05),(0.15,0.25),(0.30,0.40)]),
      dm.get_l2_norm(solution_analytic, solution_cese_fp,2,[(-0.25,-0.15),(0.30,0.40)]),
      dm.get_l2_norm(solution_analytic, solution_cese_fp,3,[(-0.25,-0.15),(-0.05,0.05),(0.30,0.40)]))

#solution_cese_fp = sodtube.get_cese_solution_fortran_porting(400, 0.001)
#print(dm.get_l2_norm(solution_analytic, solution_cese_fp,1,[(-0.25,-0.15),(-0.05,0.05),(0.15,0.25),(0.30,0.40)]),
#      dm.get_l2_norm(solution_analytic, solution_cese_fp,2,[(-0.25,-0.15),(0.30,0.40)]),
#      dm.get_l2_norm(solution_analytic, solution_cese_fp,3,[(-0.25,-0.15),(-0.05,0.05),(0.30,0.40)]))
