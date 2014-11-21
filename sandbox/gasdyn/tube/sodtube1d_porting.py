#!/usr/bin/python
#
# sodtube1d.py
#
# Description:
#     1D Sod Tube Test
#
#     This program is implemented by OO style to be
#     a part of ipython notebook demo materials.
#
#     The derivation of the equations for the analytic solution
#     is based on the book,
#     Principles of Computational Fluid Dynamics,
#     written by Pieter Wesseling.
#     Or, someone could refer to the solvcon website
#     http://www.solvcon.net/en/latest/cese.html#sod-s-shock-tube-problem
#
#
# DEBUG: search string 'DEBUG'
# why: somewhere don't understand very much ...

import sys
import scipy.optimize as so
import matplotlib.pyplot as plt

# a number to claim two floating number value are equal.
delta_precision = 0.0000000000001

class Solver():
    """
    The core to generate the 1D Sod tube test
    """
    def __init__(self):
        # initial condition
        # [(rhol, ul, pl), (rhor, ur, pr)]
        #
        # Sod's initial condition
        self.RHOL = 1.0
        self.UL = 0.0
        self.PL = 1.0
        self.RHOR = 0.125
        self.UR = 0.0
        self.PR = 0.1
        self.initcondition_sod = [(self.RHOL, self.UL, self.PL),
                                  (self.RHOR, self.UR, self.PR)]
        # initial condition for a shock tube problem
        # default is Sod's initial condition
        # users could change this initial conditions
        self.initcondition = self.initcondition_sod
        # constants and conventions
        self.GAMMA = 1.4 # ideal gas constant
        self.GAMMA2 = (self.GAMMA - 1.0) / (self.GAMMA + 1.0)
        self.ALPHA = (self.GAMMA + 1.0) / (self.GAMMA - 1.0)
        self.BETA = (self.GAMMA - 1.0) / (2.0*self.GAMMA)
        # a mesh, which has this format:
        # [point0, point1, point2, point3, ......, pointn]
        self.mesh = []
        # solution has this format:
        # [(x0, rho0, u0, p0),
        #  (x1, rho1, u1, p1),
        #  ......,
        #  (xn, rhon, un, pn)]
        self.solution = []
        self.ceseparameters = []

    def get_initcondition(self):
        return self.initcondition

    def set_initcondition(self, initcondition):
        self.initcondition = initcondition

    def gen_mesh(self,
                 xstep = 100,
                 xstart = -5050,
                 xstop = 5050):
        mesh = []
        for x in range(xstart, xstop + xstep, xstep):
            mesh.append(float(x)/10000.0)
        self.mesh = tuple(mesh)

    def get_mesh(self):
        return self.mesh

    def get_cese_solution_fortran_porting(self, iteration=100, grid_t=0.004, grid_x = 0.01):
        """
        porting chang95 demo example written in Fortran on
        Sin-Chung Chang,
        "The Method of Space-Time Conservation Element and Solution Element -
        A New Approach for Solving the Navier-Stokes and Euler Equations",
        Journal of Computational Physics, Volume 119,
        Issue 2, July 1995, Pages 295-324
        """
        import numpy as np
        
        it = iteration # iteration, which is integer
        dt = grid_t
        dx = grid_x
        ga = 1.4

        # Sod tube initial conditions of the left and right
        # of the diaphragm
        rhol = 1.0
        ul = 0.0
        pl = 1.0
        rhor = 0.125
        ur = 0.0
        pr = 0.1
        
        ia = 1

        # u_m of 1D Eular equation.
        # this also means the status of the gas
        # on grids. 
        mtxq = np.asmatrix(np.zeros(shape=(3,1000)))
        mtxqn = np.asmatrix(np.zeros(shape=(3,1000)))
        mtxqx = np.asmatrix(np.zeros(shape=(3,1000)))
        mtxqt = np.asmatrix(np.zeros(shape=(3,1000)))
        mtxs = np.asmatrix(np.zeros(shape=(3,1000)))
        vxl = np.zeros(shape=(3,1))
        vxr = np.zeros(shape=(3,1))
        xx = np.zeros(shape=(1000))
        
        mtxf = np.asmatrix(np.zeros(shape=(3,3)))
        
        hdt = dt/2.0
        qdt = dt/4.0 #q:quad
        hdx = dx/2.0
        qdx = dx/4.0
        
        tt = hdt*it
        dtx = dt/dx
        
        a1 = ga - 1.0
        a2 = 3.0 - ga
        a3 = a2/2.0
        a4 = 1.5*a1
        mtxq[0][0] = rhol
        mtxq[1][0] = rhol*ul
        mtxq[2][0] = pl/a1 + 0.5*rhol*ul**2.0
        itp = it + 1
        # initialize the gas status before the diaphragm
        # was removed.
        for i in xrange(itp):
            mtxq[0,i+1] = rhor
            mtxq[1,i+1] = rhor*ur
            mtxq[2,i+1] = pr/a1 + 0.5*rhor*ur**2.0
            # this was done by qx = np.zeros(shape=(3,1000))
            # for j in xrange(3):
            #     qx[j][i] = 0.0
        
        m = 2 # move out from the diaphragm which the 0th grid.
        for i in xrange(it):
            for j in xrange(m):
                w2 = mtxq[1,j]/mtxq[0,j]
                w3 = mtxq[2,j]/mtxq[0,j]
                # f[0][0] = 0.0
                mtxf[0,1] = 1.0
                # f[0][2] = 0.0
                mtxf[1,0] = -a3*w2**2
                mtxf[1,1] = a2*w2
                mtxf[1,2] = ga - 1.0
                mtxf[2,0] = a1*w2**3 - ga*w2*w3
                mtxf[2,1] = ga*w3 - a1*w2**2
                mtxf[2,2] = ga*w2

                # (4.17) in chang95
                mtxqt[:,j] = -1.0*mtxf*mtxqx[:,j]
                # (4.25) in chang95
                # the n_(fmt)_j of the last term should be substitubed
                # by the other terms.
                mtxs[:,j] = qdx*mtxqx[:,j] + dtx*mtxf*mtxq[:,j] \
                            - dtx*qdt*mtxf*mtxf*mtxqx[:,j]
        
            mm = m - 1
            for j in xrange(mm):
                # (4.24) in chang95
                # Please note the j+1 on the left hand side addresses
                # different t (n) from the j/j+1 on the right hand side,
                # which address the other t (n-1/2) on the space-time
                # surface.
                mtxqn[:,j+1] = 0.5*(mtxq[:,j] \
                                    + mtxq[:,j+1] \
                                    + mtxs[:,j] - mtxs[:,j+1])
                # (4.27) and (4.36) in chang95
                vxl = np.asarray((mtxqn[:,j+1] \
                                  - mtxq[:,j] - hdt*mtxqt[:,j]) \
                                  /hdx)
                vxr = np.asarray((mtxq[:,j+1] + hdt*mtxqt[:,j+1] \
                                  - mtxqn[:,j+1]) \
                                  /hdx)
                # (4.39) in chang95
                mtxqx[:,j+1] = np.asmatrix((vxl*((abs(vxr))**ia) \
                                            + vxr*((abs(vxl))**ia)) \
                                            /(((abs(vxl))**ia) \
                                                + ((abs(vxr))**ia) + 1.0E-60))
        
            for j in xrange(1,m):
                mtxq[:,j] = mtxqn[:,j]
        
            m = m + 1
        
        # draw the grid mesh
        # total distance the wave goes through
        # 100 iterations
        # 102 mesh points
        # so 101 delta segaments
        t2 = dx*float(itp) # total distance the wave goes through
        xx[0] = -0.5*t2 # ask the diaphragm location x to be zero.
        for i in xrange(itp):
            xx[i+1] = xx[i] + dx
       
        solution = []
        for i in xrange(m):
            x = mtxq[1,i]/mtxq[0,i]
            z = a1*(mtxq[2,i] - 0.5*(x**2)*mtxq[0,i])
            solution.append((xx[i],mtxq[0,i],x,z))

        #return self.solution
        return solution

