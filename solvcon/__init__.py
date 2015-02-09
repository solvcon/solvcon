# -*- coding: UTF-8 -*-
#
# Copyright (c) 2008, Yung-Yu Chen <yyc@solvcon.net>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the SOLVCON nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
SOLVCON is a collection of `Python <http://www.python.org>`__-based
conservation-law solvers that use the space-time `Conservation Element and
Solution Element (CESE) method <http://www.grc.nasa.gov/WWW/microbus/>`__.
SOLVCON targets at solving problems that can be formulated as a system of
first-order, linear or non-linear partial differential equations (PDEs).
"""

__docformat__ = 'restructuredtext en'

__version__ = '0.1.4'

__description__ = "SOLVCON: Solvers of conservation laws"

__all__ = [
    # cmdutil
    'go', 'Command',
    # conf
    'env',
    # block
    'Block',
    # solver
    'MeshSolver',
    # case
    'MeshCase',
    # anchor
    'MeshAnchor', 'MeshAnchorList',
    # hook
    'MeshHook',
    # boundcond
    'BC', 'bctregy',
    # domain
    'Domain', 'Collective', 'Distributed',
    # helper
    'helper', 'Gmsh',
    # parcel
    'parcel',
]

from .cmdutil import Command, go
from .conf import env
from .block import Block
from .solver import MeshSolver
from .case import MeshCase
from .anchor import MeshAnchor, MeshAnchorList
from .hook import MeshHook
from .boundcond import BC, bctregy
from .domain import Domain, Collective, Distributed
from . import helper
from .helper import Gmsh
from . import parcel

if __name__ == '__main__':
    go()
