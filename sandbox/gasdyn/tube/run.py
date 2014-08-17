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

rho4 = sodtube.getAnalyticDensityRegion4()
u4 = sodtube.getAnalyticVelocityRegion4()
p4 = sodtube.getAnalyticPressureRegion4()

rho3 = sodtube.getAnalyticDensityRegion3()
u3 = sodtube.getAnalyticVelocityRegion3()
p3 = sodtube.getAnalyticPressureRegion3()
#print rho4
#print rho3
#print p4
#print p3

x_shock = sodtube.getVelocityShock()*t
x_disconti = u3*t
x_fan_right = sodtube.getVelocityFanRight()*t
x_fan_left = sodtube.getVelocityFanLeft()*t

x_fan_delta = x_fan_right - x_fan_left
x_fan_delta_step = x_fan_delta/float(x_steps)

# Region 1
for x_step in xrange(x_steps,0,-1): # -1, -2, ... -10
    x = -x_fan_delta_step*x_step + x_fan_left
    print'%f, %f, %f, %f' % (x,
                             sodtube.getDensityRegion1(),
                             sodtube.getVelocityRegion1(),
                             sodtube.getPressureRegion1())

# Region 2
for x_step in xrange(x_steps): # 0, 1, ... 9
    x_fan = x_fan_delta_step*x_step + x_fan_left
    print'%f, %f, %f, %f' % (x_fan,
                             sodtube.getAnalyticDensityRegion2(x_fan,t),
                             sodtube.getAnalyticVelocityRegion2(x_fan,t),
                             sodtube.getAnalyticPressureRegion2(x_fan,t))

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
                             sodtube.getDensityRegion5(),
                             sodtube.getVelocityRegion5(),
                             sodtube.getPressureRegion5())
