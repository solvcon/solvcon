#!/usr/bin/python

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
