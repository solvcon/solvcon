#!/usr/bin/python
#
# script to run sodtube1d.py to get the analytic solution
#
# usage:
#   ./run.py
#

import sodtube1d
sodtube = sodtube1d.SodTube()

###############
x_steps = 10
t = 0.004
###############

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

# region 1
for x_step in xrange(x_steps,0,-1): # -1, -2, ... -10
    x = -x_fan_delta_step*x_step + x_fan_left
    print'%f, %f, %f, %f' % (x,
                             sodtube.get_density_region1(),
                             sodtube.get_velocity_region1(),
                             sodtube.get_pressure_region1())

# region 2
for x_step in xrange(x_steps): # 0, 1, ... 9
    x_fan = x_fan_delta_step*x_step + x_fan_left
    print'%f, %f, %f, %f' % (x_fan,
                             sodtube.get_analytic_density_region2(x_fan,t),
                             sodtube.get_analytic_velocity_region2(x_fan,t),
                             sodtube.get_analytic_pressure_region2(x_fan,t))

# region 3
x_disconti_delta = x_disconti - x_fan_right
x_disconti_delta_step = x_disconti_delta/float(x_steps)
for x_step in xrange(x_steps): # 0, 1, ... 9
    x_3 = x_disconti_delta_step*x_step + x_fan_right
    print'%f, %f, %f, %f' % (x_3,
                             rho3,
                             u3,
                             p3)

# region 4
x_shock_delta = x_shock - x_disconti
x_shock_delta_step = x_shock_delta/float(x_steps)
for x_step in xrange(x_steps): # 0, 1, ... 9
    x_4 = x_shock_delta_step*x_step + x_disconti
    print'%f, %f, %f, %f' % (x_4,
                             rho4,
                             u4,
                             p4)

# region 5
for x_step in xrange(x_steps):
    x = x_fan_delta_step*x_step + x_shock
    print'%f, %f, %f, %f' % (x,
                             sodtube.get_density_region5(),
                             sodtube.get_velocity_region5(),
                             sodtube.get_pressure_region5())
