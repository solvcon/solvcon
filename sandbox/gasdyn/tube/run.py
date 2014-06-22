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

rhoIV = sodtube.getAnalyticDensityRegionIV()
uIV = sodtube.getAnalyticVelocityRegionIV()
pIV = sodtube.getAnalyticPressureRegionIV()

rhoIII = sodtube.getAnalyticDensityRegionIII()
uIII = sodtube.getAnalyticVelocityRegionIII()
pIII = sodtube.getAnalyticPressureRegionIII()
#print rhoIV
#print rhoIII
#print pIV
#print pIII

x_shock = sodtube.getVelocityShock()*t
x_disconti = uIII*t
x_fan_right = sodtube.getVelocityFanRight()*t
x_fan_left = sodtube.getVelocityFanLeft()*t

x_fan_delta = x_fan_right - x_fan_left
x_fan_delta_step = x_fan_delta/float(x_steps)

# Region I
for x_step in xrange(x_steps,0,-1): # -1, -2, ... -10
    x = -x_fan_delta_step*x_step + x_fan_left
    print'%f, %f, %f, %f' % (x,
                             sodtube.getDensityRegionI(),
                             sodtube.getVelocityRegionI(),
                             sodtube.getPressureRegionI())

# Region II
for x_step in xrange(x_steps): # 0, 1, ... 9
    x_fan = x_fan_delta_step*x_step + x_fan_left
    print'%f, %f, %f, %f' % (x_fan,
                             sodtube.getAnalyticDensityRegionII(x_fan,t),
                             sodtube.getAnalyticVelocityRegionII(x_fan,t),
                             sodtube.getAnalyticPressureRegionII(x_fan,t))

# Region III
x_disconti_delta = x_disconti - x_fan_right
x_disconti_delta_step = x_disconti_delta/float(x_steps)
for x_step in xrange(x_steps): # 0, 1, ... 9
    x_III = x_disconti_delta_step*x_step + x_fan_right
    print'%f, %f, %f, %f' % (x_III,
                             rhoIII,
                             uIII,
                             pIII)

# Region IV
x_shock_delta = x_shock - x_disconti
x_shock_delta_step = x_shock_delta/float(x_steps)
for x_step in xrange(x_steps): # 0, 1, ... 9
    x_IV = x_shock_delta_step*x_step + x_disconti
    print'%f, %f, %f, %f' % (x_IV,
                             rhoIV,
                             uIV,
                             pIV)

# Region V
for x_step in xrange(x_steps):
    x = x_fan_delta_step*x_step + x_shock
    print'%f, %f, %f, %f' % (x,
                             sodtube.getDensityRegionV(),
                             sodtube.getVelocityRegionV(),
                             sodtube.getPressureRegionV())
