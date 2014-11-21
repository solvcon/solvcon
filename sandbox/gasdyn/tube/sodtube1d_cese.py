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

    def get_cese_solution(self,
                          iteration = 100,
                          grid_size_t = 0.004,
                          mesh_t_stop = 0.2,
                          grid_size_x = 100,
                          mesh_x_start = -10050,
                          mesh_x_stop = 10050):
        """
        given the mesh size
        output the solution based on CESE method

        iteration: int, please note n iteration will has n+2 mesh points.
        
        """

        import numpy as np

        #self.check_input(iteration,
        #                 grid_size_t,
        #                 mesh_t_stop,
        #                 grid_size_x,
        #                 mesh_x_start,
        #                 mesh_x_stop)

        self.gen_mesh(grid_size_x, mesh_x_start, mesh_x_stop)
        mesh_x = self.get_mesh()

        it = iteration # iteration, which is integer
        dt = grid_size_t
        dx = grid_size_x / 10000.0
        # mesh point number along x
        mesh_pt_number_x_at_half_t = iteration + 1
        mesh_pt_number_x = iteration + 2
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
        # prefix mtx_ means 'matrix of sth.'
        mtxq = np.asmatrix(np.zeros(shape=(3, mesh_pt_number_x)))
        # u_m, but means 'next' u_m
        # u_m at the next time step 
        mtxqn = np.asmatrix(np.zeros(shape=(3, mesh_pt_number_x)))
        # partial matrix of q partial x
        mtxqx = np.asmatrix(np.zeros(shape=(3, mesh_pt_number_x)))
        # partial mastrix of q partial t
        mtxqt = np.asmatrix(np.zeros(shape=(3, mesh_pt_number_x)))
        # matrix s, it is a part of the marching mtx_qn
        mtxs = np.asmatrix(np.zeros(shape=(3, mesh_pt_number_x)))
        vxl = np.zeros(shape=(3,1))
        vxr = np.zeros(shape=(3,1))

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
        mesh_pt_number_x_at_half_t = it + 1
        # initialize the gas status before the diaphragm
        # was removed.
        for i in xrange(mesh_pt_number_x_at_half_t):
            mtxq[0,i+1] = rhor
            mtxq[1,i+1] = rhor*ur
            mtxq[2,i+1] = pr/a1 + 0.5*rhor*ur**2.0

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
            
        #solution__x_start_index = mesh_x.get_start_grid(mesh_pt_number_x)
        solution_x_start_index = 50
        solution = []
        for i in range(mesh_pt_number_x):
            solution_x = mesh_x[i + solution_x_start_index]
            x = solution_x
            solution_rho = mtxq[0,i]
            solution_v = mtxq[1,i]/mtxq[0,i]
            solution_p = a1*(mtxq[2,i] - 0.5*(solution_v**2)*mtxq[0,i])
            solution.append((solution_x, solution_rho, solution_v, solution_p))
        
        return solution

    def get_cese_solution_mesh_size(self,
                                    iteration = 100,
                                    grid_t = 0.004,
                                    mesh_x_start = -0.5050,
                                    mesh_x_stop = 0.5050, 
                                    grid_x = 0.01):
        # iteration should be an even number to
        # make sure the grids return to the same
        # as the initial one.
        if iteration % 2 != 0:
            raise Exception(iteration, 'should be even!')
        self.gen_mesh(grid_x * 10000, mesh_x_start * 10000, mesh_x_stop * 10000)
        mesh_pt_number_x_at_half_t = iteration + 1
        # grids generated by the iteration +
        # the most left grid +
        # the most right grid
        mesh_pt_number_x = iteration + 2




    def cal_cese_solution(self, initcondition, mesh, ceseparameters):
        return self.solution

    def get_cese_status_before_half_dt(m, mtxq, mtxf):
        """
        the status is 
        """
        pass

    def get_cese_status_after_half_dt():
        pass

