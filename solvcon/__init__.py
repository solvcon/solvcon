# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2009 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
Solvcon is a framework to write explicit and time-accurate simulation codes for
PDEs with the unstructured mesh.  All the needed two- and three-dimensional
elements are supported, including triangles, quadrilaterals, tetrahedra,
pyramids, prisms, and hexahedra.  As a framework, solvcon provides:

1. A data structure for two- and three-dimensional mixed-type unstructured
   meshes.

2. Unstructured mesh importers.

3. Simulation data writers in `VTK <http://www.vtk.org/>`_ format.

4. An organized and flexible system to write pre- and post-processing codes.
   These "Hooks" are fully decoupled from the numerical algorithm of solvers.
   Several generic hooks are pre-built in solvcon.

5. Predefined, automated domain-decomposition logic.  You get it for free in
   solvcon.

6. Out-of-box RPC and IPC.  In solvcon, your code is automatically 
   parallelized (over the network, on a cluster!) in the distributed-memory
   fashion via domain decomposition.  You can write a single copy of solver
   code and run it in serial or parallel.

All you need to do is to code your numerical algorithm in the pre-defined
skeletons for one- and multi-dimensional time-marching solvers (`solver`) and 
simulation cases (`case`).

Copyright (C) 2008-2010 by Yung-Yu Chen.
"""

__docformat__ = 'restructuredtext en'

__version__ = '0.0.0+'

__description__ = "Solver Constructor: a framework to solve hyperbolic PDEs"

__all__ = ['batch', 'block', 'boundcond', 'case', 'cmdutil', 'command', 'conf',
    'conn', 'dependency', 'domain', 'gendata', 'helper', 'io', 'rpc', 'solver']

from .cmdutil import go

if __name__ == '__main__':
    go()
