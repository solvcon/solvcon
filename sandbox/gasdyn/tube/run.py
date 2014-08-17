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

rho4 = sodtube.get_analytic_DensityRegion4()
u4 = sodtube.get_analytic_VelocityRegion4()
p4 = sodtube.get_analytic_PressureRegion4()

rho3 = sodtube.get_analytic_DensityRegion3()
u3 = sodtube.get_analytic_VelocityRegion3()
p3 = sodtube.get_analytic_PressureRegion3()
#print rho4
#print rho3
#print p4
#print p3

x_shock = sodtube.get_VelocityShock()*t
x_disconti = u3*t
x_fan_right = sodtube.get_VelocityFanRight()*t
x_fan_left = sodtube.get_VelocityFanLeft()*t

x_fan_delta = x_fan_right - x_fan_left
x_fan_delta_step = x_fan_delta/float(x_steps)

# Region 1
for x_step in xrange(x_steps,0,-1): # -1, -2, ... -10
    x = -x_fan_delta_step*x_step + x_fan_left
    print'%f, %f, %f, %f' % (x,
                             sodtube.get_DensityRegion1(),
                             sodtube.get_VelocityRegion1(),
                             sodtube.get_PressureRegion1())

# Region 2
for x_step in xrange(x_steps): # 0, 1, ... 9
    x_fan = x_fan_delta_step*x_step + x_fan_left
    print'%f, %f, %f, %f' % (x_fan,
                             sodtube.get_analytic_DensityRegion2(x_fan,t),
                             sodtube.get_analytic_VelocityRegion2(x_fan,t),
                             sodtube.get_analytic_PressureRegion2(x_fan,t))

# Region 3
x_disconti_delta = x_disconti - x_fan_right
x_disconti_delta_step = x_disconti_delta/float(x_steps)
for x_step in xrange(x_steps): # 0, 1, ... 9
    x_3 = x_disconti_delta_step*x_step + x_fan_right
    print'%f, %f, %f, %f' % (x_3,
                             rho3,
                             u3,
                             p3)

# Region 4
x_shock_delta = x_shock - x_disconti
x_shock_delta_step = x_shock_delta/float(x_steps)
for x_step in xrange(x_steps): # 0, 1, ... 9
    x_4 = x_shock_delta_step*x_step + x_disconti
    print'%f, %f, %f, %f' % (x_4,
                             rho4,
                             u4,
                             p4)

# Region 5
for x_step in xrange(x_steps):
    x = x_fan_delta_step*x_step + x_shock
    print'%f, %f, %f, %f' % (x,
                             sodtube.get_DensityRegion5(),
                             sodtube.get_VelocityRegion5(),
                             sodtube.get_PressureRegion5())
