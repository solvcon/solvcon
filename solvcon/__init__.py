# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2012 Yung-Yu Chen <yyc@solvcon.net>.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""
SOLVCON: A software framework to develop high-fidelity solvers of partial
differential equtions (PDEs).

SOLVCON uses the space-time `Conservation Element and Solution Element CESE
<http://www.grc.nasa.gov/WWW/microbus/>`__) method to solve generic
conservation laws.  Python is used to host code written in C, `CUDA
<http://www.nvidia.com/object/cuda_home_new.html>`__, or other programming
languages for high-performance computing (HPC).  Hybrid parallelism is achieved
by segregating share- and distributed-memory parallel computing in the
different layers of the software framework established by Python.

SOLVCON is developed by `Yung-Yu Chen <mailto:yyc@solvcon.net>`__ and
`Sheng-Tao John Yu <mailto:yu.274@osu.edu>`__, and released under `GNU GPLv2
<http://www.gnu.org/licenses/gpl-2.0.html>`__.  Please consult the web site
http://solvcon.net/ for more information.

Key Features:

- Pluggable multi-physics
- Unstructured meshes for modeling complex geometry
- Hybrid parallel computing
- Ready-to-use I/O facilities
- Parallel I/O and in situ visualization
- Automated work flow
"""

__docformat__ = 'restructuredtext en'

__version__ = '0.1.1+'

__description__ = "SOLVCON: a software framework for PDE solvers"

__all__ = ['batch', 'batch_torque', 'block', 'boundcond', 'case',
    'cmdutil', 'command', 'conf', 'connection', 'dependency', 'domain',
    'gendata', 'helper', 'io', 'kerpak', 'mpy', 'mthread', 'rpc', 'scuda',
    'solver', 'visual_vtk']

from .cmdutil import go, test

if __name__ == '__main__':
    go()
