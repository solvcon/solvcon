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
Provide functionalities for unittests.
"""

import os
from unittest import TestCase
from .solver import BlockSolver

def openfile(filename, mode=None):
    """
    Open file with requested file name.  The file name contains relative path
    to 'data' directory in this directory, and uses forward slash as delimiter 
    of directory components.

    @param filename: path of file relative to 'data' directory in this
        directory.
    @type filename: str
    @keyword mode: file mode.
    @type mode: str
    @return: opened file.
    @rtype: file
    """
    import os
    from .conf import env
    path = [env.datadir] + filename.split('/')
    path = os.path.join(*path)
    if mode != None:
        return open(path, mode)
    else:
        return open(path)

def loadfile(filename):
    """
    Load file with requested file name.  The file name contains relative path
    to 'data' directory in this directory, and uses forward slash as delimiter 
    of directory components.

    @param filename: path of file relative to 'data' directory in this directory.
    @type filename: str
    @return: loaded data.
    @rtype: str
    """
    return openfile(filename).read()

def create_trivial_2d_blk():
    from solvcon.block import Block
    blk = Block(ndim=2, nnode=4, nface=6, ncell=3, nbound=3)
    blk.ndcrd[:,:] = (0,0), (-1,-1), (1,-1), (0,1)
    blk.cltpn[:] = 3
    blk.clnds[:,:4] = (3, 0,1,2), (3, 0,2,3), (3, 0,3,1)
    blk.build_interior()
    blk.build_boundary()
    blk.build_ghost()
    return blk

def get_blk_from_sample_neu(fpdtype=None, use_incenter=None):
    """
    Read data from sample.neu file and convert it into Block.
    """
    from .io.gambit import GambitNeutral
    from .boundcond import bctregy
    kw = {'fpdtype': fpdtype}
    if use_incenter is not None:
        kw['use_incenter'] = use_incenter
    return GambitNeutral(loadfile('sample.neu')).toblock(**kw)

def get_blk_from_oblique_neu(fpdtype=None, use_incenter=None):
    """
    Read data from oblique.neu file and convert it into Block.
    """
    from .io.gambit import GambitNeutral
    from .boundcond import bctregy
    bcname_mapper = {
        'inlet': (bctregy.unspecified, {}),
        'outlet': (bctregy.unspecified, {}),
        'wall': (bctregy.unspecified, {}),
        'farfield': (bctregy.unspecified, {}),
    }
    kw = {'fpdtype': fpdtype, 'bcname_mapper': bcname_mapper}
    if use_incenter is not None:
        kw['use_incenter'] = use_incenter
    return GambitNeutral(loadfile('oblique.neu')).toblock(**kw)
