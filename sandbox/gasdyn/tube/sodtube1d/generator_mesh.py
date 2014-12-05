#!/usr/bin/python
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
        plt.scatter(self.mesh, list_values_y)

    def show_mesh_physical_model(self):
        """
        Show how 1D Sod tube may look like.
        TODO:
            1. provide a bar for users to zoom in and out
            2. indicate the location of the diaphragm
        """
        from mpl_toolkits.mplot3d import axes3d
        import numpy as np

        fig = plt.figure()
        ax = axes3d.Axes3D(fig,azim=30,elev=30)

        x = np.linspace(-1,1,10)
        y = np.linspace(-1,1,10)

        X, Y = np.meshgrid(x,y)
        Z = np.sqrt(1-X**2)

        ax.plot_wireframe(X,Y,Z)
        ax.plot_wireframe(X,Y,-Z)

        ax.set_xbound(lower=-10.0, upper=10.0)
        ax.set_zbound(lower=-10.0, upper=10.0)
        #ax.set_axis_off()
        ax.set_xlabel('x-axis')
        ax.set_ylabel('y-axis')
        ax.set_zlabel('z-axis')
