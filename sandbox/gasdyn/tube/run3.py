#!/usr/bin/python

import sodtube1d

sodtube = sodtube1d.SodTube()
sodtube.gen_mesh()
mesh = sodtube.get_mesh()
solution_analytic = sodtube.cal_analytic_solution(mesh)

solution_cese = sodtube.get_cese_solution()

dm = sodtube1d.DataManager()
dm.dump_solution(solution_analytic)
dm.dump_solution(solution_cese)
