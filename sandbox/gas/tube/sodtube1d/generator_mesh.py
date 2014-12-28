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
        self.xstep = float(xstep)/factor
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
                                bound_x=0.5050,
                                bound_t=0.02):
        """ 
        Show the mesh represented in time and space.
        @ highlight: a flag to highlight the grids or not
        @ highlight_along_time: which grid group should be
            shown along specific time
        @ delta_t
        @ bound: |delta_t*N/2| <= bound

        TODO:
        bound_x
        bound_t
        to zoom in and out
        """
        # prepare the grid points at delta_t/2*n
        # decide the region we want to show
        mesh = list(self.mesh)
        mesh_cut = self._get_cut_mesh_by_xbound(mesh, bound_x)
        mesh_x_on_dt_n = mesh_cut
        # generate grids along (dt/2)*n
        mesh_for_loop = list(mesh_x_on_dt_n)
        mesh_x_on_dt_2_n = []
        for i in range((len(mesh_for_loop) - 1)):
            mesh_x_on_dt_2_n.append(mesh_x_on_dt_n[i] + self.xstep/2.0)

        # associate each grids along t 
        N = int(bound_t/delta_t)
        list_all_func = []
        step = delta_t/2
        for i in xrange(N):
            if i % 2 == 0:
                list_all_func.append([i*step]*len(mesh_x_on_dt_n))
            else:
                list_all_func.append([i*step]*len(mesh_x_on_dt_2_n))

        # extend it to negavie values of time
        number_of_one_side = len(list_all_func) - 1
        list_all_func_for_loop = list(list_all_func)
        for i in range(len(list_all_func_for_loop)):
            if not(i == 0):
                func_negative = []
                for ele in list_all_func_for_loop[i]:
                    func_negative.append(-ele)
                list_all_func.append(func_negative)

        # mesh is ready. now plot
        highlight_color = None # then matplotlib uses the default color
        if highlight:
            highlight_color = "green"

        # TODO: These if statement is horrible. It may be fixed to be easier to understand
        for i in range(len(list_all_func)):
            color = highlight_color if list_all_func[i][0] == highlight_along_time else None
            if number_of_one_side % 2 == 0:
                if i % 2 == 0:
                    plt.scatter(mesh_x_on_dt_n, list_all_func[i], color=color)
                else:
                    plt.scatter(mesh_x_on_dt_2_n, list_all_func[i], color=color)
            else:
                if i > number_of_one_side:
                    if i % 2 == 0:
                        plt.scatter(mesh_x_on_dt_2_n, list_all_func[i], color=color)
                    else:
                        plt.scatter(mesh_x_on_dt_n, list_all_func[i], color=color)
                else:
                    if i % 2 == 0:
                        plt.scatter(mesh_x_on_dt_n, list_all_func[i], color=color)
                    else:
                        plt.scatter(mesh_x_on_dt_2_n, list_all_func[i], color=color)


    def _get_cut_mesh_by_xbound(self, mesh, bound_x):
        mesh_cut = []
        for pt in mesh:
            if abs(pt) < bound_x or abs(pt) == bound_x:
                mesh_cut.append(pt)
        return mesh_cut
