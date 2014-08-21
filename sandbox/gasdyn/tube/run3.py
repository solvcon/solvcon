#!/usr/bin/python

import sodtube1d

sodtube = sodtube1d.SodTube()
sodtube.gen_mesh()
mesh = sodtube.get_mesh()
solution_analytic = sodtube.cal_analytic_solution(mesh)

solution_cese = sodtube.get_cese_solution()

dm = sodtube1d.DataManager()

dm.get_deviation(solution_analytic, solution_cese)
#solution_deviation = dm.get_deviation(solution_analytic, solution_cese)
solution_deviation_percent = dm.get_deviation_percent(solution_analytic, solution_cese)

#dm.dump_solution(solution_analytic)
#dm.dump_solution(solution_cese)
#dm.dump_solution(solution_deviation)
dm.dump_solution(solution_deviation_percent)
