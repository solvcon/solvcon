#
# generator_mesh.py
#
# Description:
#   a generator to help you to have a mesh.
#

import matplotlib.pyplot as plt

class Mesher():
    def __init__(self):
        pass

    def gen_mesh(self,
                 xstep = 100,
                 xstart = -5050,
                 xstop = 5050):
        factor = 10000.0
        self.xstep = xstep/factor
        # TODO: you may use np.linspace
        mesh = []
        for x in range(xstart, xstop + xstep, xstep):
            mesh.append(float(x)/factor)
        self.mesh = tuple(mesh)

    def get_mesh(self):
        return self.mesh

    def show_mesh_ipython_nb(self):
        """
        Show the mesh by plotting the mesh points.
        matplotlib.pyplot.plot is not used
        because this method is used image inline ipython notebook.
        """
        list_values_y = [0]*len(self.mesh)
        plt.scatter(self.mesh, list_values_y)

    def show_mesh_in_space_time_ipynb(self,
                                highlight=False,
                                highlight_along_time=0,
                                delta_t=0.004,
                                bound=0.02):
        """ 
        Show the mesh represented in time and space.
        @ highlight: a flag to highlight the grids or not
        @ highlight_along_time: which grid group should be
            shown along specific time
        @ delta_t
        @ bound: |delta_t*N/2| <= bound
        """
        # prepare the grid points at delta_t/2*n
        mesh_for_loop = list(self.mesh)
        mesh_x_on_dt_2_n = []
        for i in range((len(mesh_for_loop) - 1)):
            mesh_x_on_dt_2_n.append(self.mesh[i] + self.xstep)
            
        N = int(bound/delta_t)
        number_grid_x = len(self.mesh)
        list_all_func = []
        step = delta_t/2
        for i in xrange(N):
            if i % 2 == 0:
                list_all_func.append([i*step]*number_grid_x)
            else:
                list_all_func.append([i*step]*(number_grid_x - 1))

        # extend it to negavie values
        list_all_func_for_loop = list(list_all_func)
        for i in range(len(list_all_func_for_loop)):
            if not(i == 0):
                func_negative = []
                for ele in list_all_func_for_loop[i]:
                    func_negative.append(-ele)
                list_all_func.append(func_negative)

        # now plot
        for i in range(len(list_all_func)):
            if i % 2 == 0:
                plt.scatter(self.mesh, list_all_func[i])
            else:
                plt.scatter(mesh_x_on_dt_2_n, list_all_func[i])
