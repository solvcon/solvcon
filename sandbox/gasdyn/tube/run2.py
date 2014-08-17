#!/usr/bin/python
#
# script to run sodtube1d.py to get the analytic solution
#
# usage:
#   ./run2.py
#

import ipdb
import sodtube1d
import sodtubecmdp
sodtube = sodtube1d.SodTube()
solution_client = sodtubecmdp.SolutionClient()

###############
x_steps = 10
t = 0.004
###############

### get grid ####
solution_client.invoke("grid")
mesh = solution_client._solver._grid
#################

rho4 = sodtube.get_analytic_density_region4()
u4 = sodtube.get_analytic_velocity_region4()
p4 = sodtube.get_analytic_pressure_region4()

rho3 = sodtube.get_analytic_density_region3()
u3 = sodtube.get_analytic_velocity_region3()
p3 = sodtube.get_analytic_pressure_region3()
#print rho4
#print rho3
#print p4
#print p3

x_shock = sodtube.get_velocity_shock()*t
x_disconti = u3*t
x_fan_right = sodtube.get_velocity_fan_right()*t
x_fan_left = sodtube.get_velocity_fan_left()*t

x_fan_delta = x_fan_right - x_fan_left
x_fan_delta_step = x_fan_delta/float(x_steps)


for x in mesh:
    #if x > 0.02:
        #ipdb.set_trace()
    if x < x_fan_left or x == x_fan_left:
        print'%f, %f, %f, %f' % (x,
                                 sodtube.get_density_region1(),
                                 sodtube.get_velocity_region1(),
                                 sodtube.get_pressure_region1())
    elif x > x_fan_left and (x < x_fan_right or x == x_fan_right):
        print'%f, %f, %f, %f' % (x,
                                 sodtube.get_analytic_density_region2(float(x),t),
                                 sodtube.get_analytic_velocity_region2(float(x),t),
                                 sodtube.get_analytic_pressure_region2(float(x),t))
    elif x > x_fan_right and (x < x_disconti or x == x_disconti):
        print'%f, %f, %f, %f' % (x,
                                 rho3,
                                 u3,
                                 p3)
    elif x > x_disconti and (x < x_shock or x == x_shock):
        print'%f, %f, %f, %f' % (x,
                                 rho4,
                                 u4,
                                 p4)
    elif x > x_shock:
        print'%f, %f, %f, %f' % (x,
                                 sodtube.get_density_region5(),
                                 sodtube.get_velocity_region5(),
                                 sodtube.get_pressure_region5())
    else:
        print("Something wrong!!!")
