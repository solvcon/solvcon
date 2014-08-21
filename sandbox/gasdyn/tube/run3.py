#!/usr/bin/python
import sodtube1d
sodtube = sodtube1d.SodTube()
sodtube.gen_mesh()
mesh = sodtube.get_mesh()
solution = sodtube.cal_analytic_Solution(mesh)


