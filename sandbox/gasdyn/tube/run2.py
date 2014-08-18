#!/usr/bin/python
#
# script to run sodtube1d.py to get the analytic solution
# this is under development for the refractor-ing sodtube1d.py testing
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
#t = 0.004
t = 0.19690027600282164 #0.345/1.752156
###############

### gen grid, get mesh ####
solution_client.invoke("grid")
mesh = solution_client._solver._grid
#################

rho4 = sodtube.get_analytic_density_region4()
u4 = sodtube.get_analytic_velocity_region4()
p4 = sodtube.get_analytic_pressure_region4()

rho3 = sodtube.get_analytic_density_region3()
u3 = sodtube.get_analytic_velocity_region3()
p3 = sodtube.get_analytic_pressure_region3()

x_shock = sodtube.get_velocity_shock()*t
x_disconti = u3*t
x_fan_right = sodtube.get_velocity_fan_right()*t
x_fan_left = sodtube.get_velocity_fan_left()*t

for x in mesh:
    if x < x_fan_left or x == x_fan_left:
        print'x %f, rho %f, u %f, p %f' % (x,
                                 sodtube.get_density_region1(),
                                 sodtube.get_velocity_region1(),
                                 sodtube.get_pressure_region1())
    elif x > x_fan_left and (x < x_fan_right or x == x_fan_right):
        print'x %f, rho %f, u %f, p %f' % (x,
                                 sodtube.get_analytic_density_region2(float(x),t),
                                 sodtube.get_analytic_velocity_region2(float(x),t),
                                 sodtube.get_analytic_pressure_region2(float(x),t))
    elif x > x_fan_right and (x < x_disconti or x == x_disconti):
        print'x %f, rho %f, u %f, p %f' % (x,
                                 rho3,
                                 u3,
                                 p3)
    elif x > x_disconti and (x < x_shock or x == x_shock):
        print'x %f, rho %f, u %f, p %f' % (x,
                                 rho4,
                                 u4,
                                 p4)
    elif x > x_shock:
        print'x %f, rho %f, u %f, p %f' % (x,
                                 sodtube.get_density_region5(),
                                 sodtube.get_velocity_region5(),
                                 sodtube.get_pressure_region5())
    else:
        print("Something wrong!!!")

# pre check
# rho4 0.265574 u4 0.927453 p4 0.303130
# compare to the fortran
# 0.3350  0.2663  0.9296  0.3041  0.7352
# and my porting
# 0.335000 0.254647 0.917552 0.306570
#print "x_fan_right %f rho2 %f u2 %f p2 %f" % (x_fan_right, rho2, u2, p2)
print "x_disconti %f rho3 %f u3 %f p3 %f" % (x_disconti, rho3, u3, p3)
print "x_shock    %f rho4 %f u4 %f p4 %f" % (x_shock, rho4, u4, p4)
print "v_shock %f " % (sodtube.get_velocity_shock())
print "x_fan_left %f, x_fan_right %f, x_disconti %f, x_shock %f" % (x_fan_left, x_fan_right, x_disconti, x_shock)
