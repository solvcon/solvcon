#
# solver_cese.py
#
# Description:
#   1D Sod Tube sover based on CESE method.
#
#   This program is implemented by OO style to be
#   a part of ipython notebook demo materials.
#
#   The derivation of the equations for the analytic solution
#   is based on the book,
#   Principles of Computational Fluid Dynamics,
#   written by Pieter Wesseling.
#   Or, people could refer to the solvcon website
#   http://www.solvcon.net/en/latest/cese.html#sod-s-shock-tube-problem
#

import numpy as np
import generator_mesh

gmesh = generator_mesh.Mesher()

GAMMA = 1.4

# Sod tube initial conditions of the left and right
# of the diaphragm
RHO_L = 1.0
U_L = 0.0
P_L = 1.0
RHO_R = 0.125
U_R = 0.0
P_R = 0.1

class Data(object):
    """
    a container of the data during the CESE iteration process
    """
    _excludes = ['__class__',
                 '__delattr__',
                 '__dict__',
                 '__doc__',
                 '__format__',
                 '__getattribute__',
                 '__hash__',
                 '__init__',
                 '__module__',
                 '__new__',
                 '__reduce__',
                 '__reduce_ex__',
                 '__repr__',
                 '__setattr__',
                 '__sizeof__',
                 '__str__',
                 '__subclasshook__',
                 '__weakref__']

    _includes = ['iteration',
                 'it_nb', # the nth iteration number, 0 <= it_nb < iteration, -1 means the iteration does not begin
                 'it_pt_nb', # iteration point number at the it_nb
                 'grid_size_t',
                 'grid_size_x',
                 'mesh_pt_number_x',
                 'mesh_pt_number_x_at_half_t',
                 'mesh_t_stop',
                 'mesh_x',
                 'mesh_x_start',
                 'mesh_x_stop',
                 'mtx_f',
                 'mtx_q',
                 'mtx_qn',
                 'mtx_qx',
                 'mtx_qt',
                 'mtx_s',
                 'vxl',
                 'vxr',
                 'solution'
                ]

    def __init__(self, **kwargs):
        for k,v in kwargs.items():
            if k in self._excludes or k not in self._includes:
                raise TypeError("{0} is not a valide keyword argument".format(k))
            self.__dict__[k] = v

    def refresh_solution(self):
        """
        Use the latest mtx_q to caculate the associated rho, v and p.
        Also, apply physic meaning, e.g. location, according to the given parameters.
        """
        # TODO: warning or implementation for refreshing on odd iteration numbers (status at half t)
        solution = []
        # TODO: only for even iteration number status
        #
        # -1 : iteration does not begin (yet), there are two grid points with ready status,
        # There are totally self.it_nb + 3 iterated grid points.
        #
        # Every two iteration steps gains one x step in lefthand and righthand side separately,
        # and we begin the iteration by 2 grid points (it_pt_nb = 2 in the initialized state)
        # For 100 iterations, totally 51 grid points are 'used' in the left hand side.
        # There is len(solver_cese.data.mesh_x) /2 grid points ready in the left hand side,
        # so we associate x mesh grid location index with the mtx_q by solution_x_start_index as
        #solution_x_start_index = 50
        #solution_x_start_index = len(self.mesh_x) /2 - (self.iteration + 2)/2
        total_iterated_pt_nb = self.it_nb + 3
        solution_x_start_index = (len(self.mesh_x)/2) - (total_iterated_pt_nb/2)
        for i in range(total_iterated_pt_nb):
            solution_x = self.mesh_x[i + solution_x_start_index]
            x = solution_x
            solution_rho = self.mtx_q[0,i]
            solution_v = self.mtx_q[1,i]/self.mtx_q[0,i]
            solution_p = (GAMMA-1.0)*(self.mtx_q[2,i] - 0.5*(solution_v**2)*self.mtx_q[0,i])
            solution.append((solution_x, solution_rho, solution_v, solution_p))
        self.solution = self.fill_solution(solution)


    def fill_solution(self, solution_list):
        """
        fill solution array with initial values if there is without solution
        """
        solution_list_rtn = [0] * len(self.mesh_x)
        solution_x_list = []

        for solution in solution_list:
            solution_x_list.append(solution[0])

        for idx in range(len(self.mesh_x)):
            location = self.mesh_x[idx]
            # mapping known solutions
            for solution in solution_list:
                solution_x = solution[0]
                if location == solution_x:
                    solution_list_rtn[idx] = solution
            # fill initial values for the remaining mesh points
            if location not in solution_x_list:
                if location < 0:
                    solution_list_rtn[idx] = (location, RHO_L, U_L, P_L)
                else:
                    solution_list_rtn[idx] = (location, RHO_R, U_R, P_R)

        return solution_list_rtn


class Solver(object):
    """
    CESE method to generate the 1D Sod tube solution
    @iteration, which is integer
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

        gmesh.gen_mesh(grid_size_x, mesh_x_start, mesh_x_stop)
        mesh_x = gmesh.get_mesh()
        grid_size_x = grid_size_x / 10000.0

        #self.check_input(iteration,
        #                 grid_size_t,
        #                 mesh_t_stop,
        #                 grid_size_x,
        #                 mesh_x_start,
        #                 mesh_x_stop)

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
        
        # introduce data object to contain data used during the CESE iteration
        self._data = Data(iteration=iteration,
                         grid_size_t=grid_size_t,
                         grid_size_x=grid_size_x,
                         mesh_pt_number_x=mesh_pt_number_x,
                         mesh_pt_number_x_at_half_t=mesh_pt_number_x_at_half_t,
                         mesh_t_stop=mesh_t_stop,
                         mesh_x=mesh_x,
                         mesh_x_start=mesh_x_start,
                         mesh_x_stop=mesh_x_stop,
                         mtx_f=mtx_f,
                         mtx_q=mtx_q,
                         mtx_qn=mtx_qn,
                         mtx_qx=mtx_qx,
                         mtx_qt=mtx_qt,
                         mtx_s=mtx_s,
                         vxl=vxl,
                         vxr=vxr
                         )

    @property
    def data(self):
        return self._data

    def run_cese_iteration(self):
        """
        the whole CESE iteration process
        """
        # initialize the gas status before the diaphragm was removed.
        self.init_gas_status()
        # The number used to calculate the status before
        # the half delta t stepping is applied is 2.
        # This means we begin the iteration process by two grid points.
        self._data.it_pt_nb = 2
        for i in xrange(self._data.iteration):
            self._data.it_nb = i
            self.cal_cese_status_before_half_dt()
            self.cal_cese_status_after_half_dt()
            self.push_status_along_t()
        # this is not necessary
        # but it is not risk to refresh and make sure
        # our solution is up-to-date.
        self._data.refresh_solution()

    def get_cese_solution(self):
        self._data.refresh_solution()
        return list(self._data.solution)

    def get_cese_solution_mesh_size(self,
                                    iteration = 100,
                                    grid_t = 0.004,
                                    mesh_x_start = -0.5050,
                                    mesh_x_stop = 0.5050, 
                                    grid_x = 0.01):
        # TODO: not complete!

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

    def cal_cese_status_before_half_dt(self):
        """
        the gas current status
        """
        data = self._data
        m = data.it_pt_nb
        mtx_q = data.mtx_q
        mtx_f = data.mtx_f
        mtx_qt = data.mtx_qt
        mtx_qx = data.mtx_qx
        mtx_s = data.mtx_s

        for j in xrange(m):
            w2 = mtx_q[1,j]/mtx_q[0,j]
            w3 = mtx_q[2,j]/mtx_q[0,j]
            # f[0][0] = 0.0
            mtx_f[0,1] = 1.0
            # f[0][2] = 0.0
            mtx_f[1,0] = -((3.0-GAMMA)/2.0)*w2**2
            mtx_f[1,1] = (3.0-GAMMA)*w2
            mtx_f[1,2] = GAMMA - 1.0
            mtx_f[2,0] = (GAMMA-1.0)*w2**3 - GAMMA*w2*w3
            mtx_f[2,1] = GAMMA*w3 - (GAMMA-1.0)*w2**2
            mtx_f[2,2] = GAMMA*w2

            # (4.17) in chang95
            mtx_qt[:,j] = -1.0*mtx_f*mtx_qx[:,j]
            # (4.25) in chang95
            # the n_(fmt)_j of the last term should be substitubed
            # by the other terms.
            mtx_s[:,j] = (self._data.grid_size_x/4.0)*mtx_qx[:,j] + (self._data.grid_size_t/self._data.grid_size_x)*mtx_f*mtx_q[:,j] \
                        - (self._data.grid_size_t/self._data.grid_size_x)*(self._data.grid_size_t/4.0)*mtx_f*mtx_f*mtx_qx[:,j]

    def cal_cese_status_after_half_dt(self):
        """
        the gas status after half of dt
        """
        # stepping into the next halt delta t
        # m mesh points along t could introduce m - 1 mesh points along t + 0.5*dt
        data = self._data
        m = data.it_pt_nb
        mm = m - 1
        mtx_q = data.mtx_q
        mtx_qn = data.mtx_qn
        mtx_qt = data.mtx_qt
        mtx_qx = data.mtx_qx
        mtx_s = data.mtx_s
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
                              - mtx_q[:,j] - (self._data.grid_size_t/2.0)*mtx_qt[:,j]) \
                              /(self._data.grid_size_x/2.0))
            vxr = np.asarray((mtx_q[:,j+1] + (self._data.grid_size_t/2.0)*mtx_qt[:,j+1] \
                              - mtx_qn[:,j+1]) \
                              /(self._data.grid_size_x/2.0))
            # (4.39) in chang95
            mtx_qx[:,j+1] = np.asmatrix((vxl*((abs(vxr))**1.0) \
                                        + vxr*((abs(vxl))**1.0)) \
                                        /(((abs(vxl))**1.0) \
                                            + ((abs(vxr))**1.0) + 1.0E-60))
        
    def push_status_along_t(self):
        """
        step into the next iteration status
        """
        #  ask the status at t + 0.5*dt to be the next status before the half delta t is applied
        # hdt means 0.5*grid_size_t
        data = self._data
        number_mesh_points_before_hdt = data.it_pt_nb
        mtx_q = data.mtx_q
        mtx_qn = data.mtx_qn
        for j in xrange(1,number_mesh_points_before_hdt):
            mtx_q[:,j] = mtx_qn[:,j]
        data.it_pt_nb = number_mesh_points_before_hdt + 1

    def init_gas_status(self,
                        rho_l=RHO_L,
                        u_l=U_L,
                        p_l=P_L,
                        rho_r=RHO_R,
                        u_r=U_R,
                        p_r=P_R
                        ):
        # access necessary data member
        mesh_pt_number_x_at_half_t = self._data.mesh_pt_number_x_at_half_t
        mtx_q = self._data.mtx_q
        # set up the status in the lefthand side
        mtx_q[0][0] = rho_l
        mtx_q[1][0] = rho_l*u_l
        mtx_q[2][0] = p_l/(GAMMA-1.0) + 0.5*rho_l*u_l**2.0
        # set up the status in the righthand side
        for i in xrange(mesh_pt_number_x_at_half_t):
            mtx_q[0,i+1] = rho_r
            mtx_q[1,i+1] = rho_r*u_r
            mtx_q[2,i+1] = p_r/(GAMMA-1.0) + 0.5*rho_r*u_r**2.0
        self._data.it_nb = -1 # no CESE iteration begins

