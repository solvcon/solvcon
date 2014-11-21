#!/usr/bin/python
#
# run.py
#
# Usage:
#     ./run.py
#
# Description:
#     An example to show how to use sodtube1d class

# solvers
import sodtube1d.generator_mesh as gmesh
import sodtube1d.solver_analytic as analytic
import sodtube1d.solver_cese as cese
import sodtube1d.solver_porting as porting
# helper handlers
import sodtube1d.handler_data as hd
import sodtube1d.handler_plot as hp


generator_mesh = gmesh.Mesher()
generator_mesh.gen_mesh()
mesh = generator_mesh.get_mesh()
analytic_solver = analytic.Solver()
solution_analytic = analytic_solver.get_analytic_solution(mesh)

solver_cese = cese.Solver()
solution_cese = solver_cese.get_cese_solution()

solver_porting = porting.Solver()
solution_porting = solver_porting.get_cese_solution_fortran_porting()

dm = hd.DataManager()
pm = hp.PlotManager()

#solution_deviation = dm.get_deviation(solution_analytic, solution_cese_porting)
solution_deviation = dm.get_deviation(solution_porting, solution_cese)
#solution_deviation_percent = dm.get_deviation_percent(solution_analytic, solution_cese_porting)

#dm.dump_solution(solution_analytic)
#dm.dump_solution(solution_cese_porting)
#dm.dump_solution(solution_cese)
dm.dump_solution(solution_deviation)
#dm.dump_solution(solution_deviation_percent)

#dm.get_plot_solutions_fig_rho(solution_analytic, solution_cese_porting, "analytic rho", "cese rho")
#dm.get_plot_solutions_fig_v(solution_analytic, solution_cese_porting, "analytic v", "cese v")
#dm.get_plot_solutions_fig_p(solution_analytic, solution_cese_porting, "analytic p", "cese p")
pm.get_plot_solutions_fig_rho(solution_analytic, solution_cese, "analytic rho", "cese rho")
pm.get_plot_solutions_fig_v(solution_analytic, solution_cese, "analytic v", "cese v")
pm.get_plot_solutions_fig_p(solution_analytic, solution_cese, "analytic p", "cese p")
pm.show_solution_comparison()

#print("l2 norm is: (rho, v, p)")
##print(dm.get_l2_norm(solution_analytic, solution_cese_porting,1),
##      dm.get_l2_norm(solution_analytic, solution_cese_porting,2),
##      dm.get_l2_norm(solution_analytic, solution_cese_porting,3))
#print(dm.get_l2_norm(solution_analytic, solution_cese_porting,1,[(-0.25,-0.15),(-0.05,0.05),(0.15,0.25),(0.30,0.40)]),
#      dm.get_l2_norm(solution_analytic, solution_cese_porting,2,[(-0.25,-0.15),(0.30,0.40)]),
#      dm.get_l2_norm(solution_analytic, solution_cese_porting,3,[(-0.25,-0.15),(-0.05,0.05),(0.30,0.40)]))
#
#sodtube.gen_mesh(50, -5050, 5050)
#mesh = sodtube.get_mesh()
#solution_analytic = sodtube.get_analytic_solution(mesh)
#solution_cese_porting = sodtube.get_cese_solution_fortran_porting(201, 0.002, 0.005)
##dm.dump_solution(solution_analytic)
##dm.dump_solution(solution_cese_porting)
#
##dm.get_plot_solutions_fig_rho(solution_analytic, solution_cese_porting, "analytic rho (diff mesh)", "cese rho")
##dm.get_plot_solutions_fig_v(solution_analytic, solution_cese_porting, "analytic v (diff mesh)", "cese v")
##dm.get_plot_solutions_fig_p(solution_analytic, solution_cese_porting, "analytic p (diff mesh)", "cese p")
#print(dm.get_l2_norm(solution_analytic, solution_cese_porting,1,[(-0.25,-0.15),(-0.05,0.05),(0.15,0.25),(0.30,0.40)]),
#      dm.get_l2_norm(solution_analytic, solution_cese_porting,2,[(-0.25,-0.15),(0.30,0.40)]),
#      dm.get_l2_norm(solution_analytic, solution_cese_porting,3,[(-0.25,-0.15),(-0.05,0.05),(0.30,0.40)]))
#
#sodtube.gen_mesh(25, -5050, 5050)
#mesh = sodtube.get_mesh()
#solution_analytic = sodtube.get_analytic_solution(mesh)
#solution_porting = sodtube.get_cese_solution_fortran_porting(403, 0.001, 0.0025)
##dm.dump_solution(solution_cese_porting)
#print(dm.get_l2_norm(solution_analytic, solution_cese_porting,1,[(-0.25,-0.15),(-0.05,0.05),(0.15,0.25),(0.30,0.40)]),
#      dm.get_l2_norm(solution_analytic, solution_cese_porting,2,[(-0.25,-0.15),(0.30,0.40)]),
#      dm.get_l2_norm(solution_analytic, solution_cese_porting,3,[(-0.25,-0.15),(-0.05,0.05),(0.30,0.40)]))
