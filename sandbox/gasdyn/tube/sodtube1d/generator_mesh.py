#!/usr/bin/python
#
# generator_mesh.py
#
# Description:
#   a generator to help you to have a mesh.
#

from matplotlib.pyplot import scatter

class Mesher():
    def __init__(self):
        pass

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

    def show_mesh_ipython_nb(self):
        """
        Show the mesh by plotting the mesh points.
        matplotlib.pyplot.plot is not used
        because this method is used image inline ipython notebook.
        """
        list_values_y = [0]*len(self.mesh)
        scatter(self.mesh, list_values_y)
