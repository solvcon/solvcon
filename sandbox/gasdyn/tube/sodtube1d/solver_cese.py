#!/usr/bin/python
#
# solver_cese.py
#
# Description:
#     1D Sod Tube sover based on CESE method.
#
#     This program is implemented by OO style to be
#     a part of ipython notebook demo materials.
#
#     The derivation of the equations for the analytic solution
#     is based on the book,
#     Principles of Computational Fluid Dynamics,
#     written by Pieter Wesseling.
#     Or, people could refer to the solvcon website
#     http://www.solvcon.net/en/latest/cese.html#sod-s-shock-tube-problem
#

import numpy as np
import generator_mesh

gmesh = generator_mesh.Mesher()

# a number to claim two floating number value are equal.
delta_precision = 0.0000000000001

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

a1 = ga - 1.0
a2 = 3.0 - ga
a3 = a2/2.0
a4 = 1.5*a1


tt = 0.0
dtx = 0.0

class Solver():
    """
    CESE method to generate the 1D Sod tube solution

    TODO:
        1. grid_size_x, mesh_x_start, and mesh_x_stop should be integer
        for gmesh.gen_mesh method, which requires integer inputs to be easier
        to generate mesh points along x. This should be smarter and allow user
        to input their own numbers. Users' inputs are 10000 times as
        the real inputs for CESE interation.

    """
    def __init__(self,
                 iteration = 100,
                 grid_size_t = 0.004,
                 mesh_t_stop = 0.2,
                 grid_size_x = 100,
                 mesh_x_start = -10050,
                 mesh_x_stop = 10050):
        self.iteration = iteration
        self.grid_size_t = grid_size_t
        self.mesh_t_stop = mesh_t_stop
        self.mesh_x_start = mesh_x_start
        self.mesh_x_stop = mesh_x_stop

        gmesh.gen_mesh(grid_size_x, mesh_x_start, mesh_x_stop)
        self.mesh_x = gmesh.get_mesh()
        self.grid_size_x = grid_size_x / 10000.0

    def get_cese_solution(self):
        """
        given the mesh size
        output the solution based on CESE method

        iteration: int, please note n iteration will has n+2 mesh points.
        
        """
        global tt, dtx 
        iteration = self.iteration
        grid_size_t = self.grid_size_t
        mesh_t_stop = self.mesh_t_stop
        grid_size_x = self.grid_size_x
        mesh_x_start = self.mesh_x_start
        mesh_x_stop = self.mesh_x_stop
        mesh_x = self.mesh_x

        #self.check_input(iteration,
        #                 grid_size_t,
        #                 mesh_t_stop,
        #                 grid_size_x,
        #                 mesh_x_start,
        #                 mesh_x_stop)

        it = iteration # iteration, which is integer
        dt = grid_size_t
        # mesh point number along x
        mesh_pt_number_x_at_half_t = iteration + 1
        mesh_pt_number_x = iteration + 2

        # u_m of 1D Eular equation.
        # this also means the status of the gas
        # on grids.
        # prefix mtx_ means 'matrix of sth.'
        mtx_q = np.asmatrix(np.zeros(shape=(3, mesh_pt_number_x)))
        # u_m, but means 'next' u_m
        # u_m at the next time step 
        mtx_qn = np.asmatrix(np.zeros(shape=(3, mesh_pt_number_x)))
        # partial matrix of q partial x
        mtx_qx = np.asmatrix(np.zeros(shape=(3, mesh_pt_number_x)))
        # partial mastrix of q partial t
        mtx_qt = np.asmatrix(np.zeros(shape=(3, mesh_pt_number_x)))
        # matrix s, it is a part of the marching mtx_qn
        mtx_s = np.asmatrix(np.zeros(shape=(3, mesh_pt_number_x)))
        vxl = np.zeros(shape=(3,1))
        vxr = np.zeros(shape=(3,1))

        mtx_f = np.asmatrix(np.zeros(shape=(3,3)))
        
        tt = (grid_size_t/2.0)*it
        dtx = dt/self.grid_size_x
        
        mtx_q[0][0] = rhol
        mtx_q[1][0] = rhol*ul
        mtx_q[2][0] = pl/a1 + 0.5*rhol*ul**2.0
        mesh_pt_number_x_at_half_t = it + 1
        # initialize the gas status before the diaphragm
        # was removed.
        for i in xrange(mesh_pt_number_x_at_half_t):
            mtx_q[0,i+1] = rhor
            mtx_q[1,i+1] = rhor*ur
            mtx_q[2,i+1] = pr/a1 + 0.5*rhor*ur**2.0

        # m is the number used to calculate the status before
        # the half delta t stepping is applied.
        m = 2 # move out from the diaphragm which the 0th grid.
        for i in xrange(it):
            self.get_cese_status_before_half_dt(m, mtx_q, mtx_f, mtx_qt, mtx_qx, mtx_s)
            # stepping into the next halt delta t
            # m mesh points along t could introduce m - 1 mesh points along t + 0.5*dt
            self.get_cese_status_after_half_dt(m, mtx_q, mtx_qn, mtx_qt, mtx_qx, mtx_s)
            #  ask the status at t + 0.5*dt to be the next status before the half delta t is applied
            m = self.push_status_along_t(m, mtx_q, mtx_qn)
            
        #solution__x_start_index = mesh_x.get_start_grid(mesh_pt_number_x)
        solution_x_start_index = 50
        solution = []
        for i in range(mesh_pt_number_x):
            solution_x = mesh_x[i + solution_x_start_index]
            x = solution_x
            solution_rho = mtx_q[0,i]
            solution_v = mtx_q[1,i]/mtx_q[0,i]
            solution_p = a1*(mtx_q[2,i] - 0.5*(solution_v**2)*mtx_q[0,i])
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
        gmesh.gen_mesh(grid_x * 10000, mesh_x_start * 10000, mesh_x_stop * 10000)
        mesh_pt_number_x_at_half_t = iteration + 1
        # grids generated by the iteration +
        # the most left grid +
        # the most right grid
        mesh_pt_number_x = iteration + 2




    def cal_cese_solution(self, initcondition, mesh, ceseparameters):
        return self.solution

    def get_cese_status_before_half_dt(self, m, mtx_q, mtx_f, mtx_qt, mtx_qx, mtx_s):
        """
        the status is 
        """
        for j in xrange(m):
            w2 = mtx_q[1,j]/mtx_q[0,j]
            w3 = mtx_q[2,j]/mtx_q[0,j]
            # f[0][0] = 0.0
            mtx_f[0,1] = 1.0
            # f[0][2] = 0.0
            mtx_f[1,0] = -a3*w2**2
            mtx_f[1,1] = a2*w2
            mtx_f[1,2] = ga - 1.0
            mtx_f[2,0] = a1*w2**3 - ga*w2*w3
            mtx_f[2,1] = ga*w3 - a1*w2**2
            mtx_f[2,2] = ga*w2

            # (4.17) in chang95
            mtx_qt[:,j] = -1.0*mtx_f*mtx_qx[:,j]
            # (4.25) in chang95
            # the n_(fmt)_j of the last term should be substitubed
            # by the other terms.
            mtx_s[:,j] = (self.grid_size_x/4.0)*mtx_qx[:,j] + dtx*mtx_f*mtx_q[:,j] \
                        - dtx*(self.grid_size_t/4.0)*mtx_f*mtx_f*mtx_qx[:,j]
        return  m, mtx_q, mtx_f, mtx_qt, mtx_qx, mtx_s

    def get_cese_status_after_half_dt(self, m, mtx_q, mtx_qn, mtx_qt, mtx_qx, mtx_s):
        mm = m - 1
        for j in xrange(mm):
            # (4.24) in chang95
            # Please note the j+1 on the left hand side addresses
            # different t (n) from the j/j+1 on the right hand side,
            # which address the other t (n-1/2) on the space-time
            # surface.
            mtx_qn[:,j+1] = 0.5*(mtx_q[:,j] \
                                + mtx_q[:,j+1] \
                                + mtx_s[:,j] - mtx_s[:,j+1])
            # (4.27) and (4.36) in chang95
            vxl = np.asarray((mtx_qn[:,j+1] \
                              - mtx_q[:,j] - (self.grid_size_t/2.0)*mtx_qt[:,j]) \
                              /(self.grid_size_x/2.0))
            vxr = np.asarray((mtx_q[:,j+1] + (self.grid_size_t/2.0)*mtx_qt[:,j+1] \
                              - mtx_qn[:,j+1]) \
                              /(self.grid_size_x/2.0))
            # (4.39) in chang95
            mtx_qx[:,j+1] = np.asmatrix((vxl*((abs(vxr))**ia) \
                                        + vxr*((abs(vxl))**ia)) \
                                        /(((abs(vxl))**ia) \
                                            + ((abs(vxr))**ia) + 1.0E-60))
        
    def push_status_along_t(self, number_mesh_points_before_hdt, mtx_q, mtx_qn):
        # hdt means 0.5*grid_size_t
        for j in xrange(1,number_mesh_points_before_hdt):
            mtx_q[:,j] = mtx_qn[:,j]
        number_mesh_points_before_hdt_next_iter = number_mesh_points_before_hdt + 1
        return number_mesh_points_before_hdt_next_iter
