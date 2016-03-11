#!/usr/bin/env python3.5
# Copyright (C) 2016 Taihsiang Ho <tai271828@gmail.com>.
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
'''
This example shows how to:
1. convert .neu to a block, which is a general SOLVCON block object.
2. convert a block to .vtk, so we could view
   the geometry of the block by paraview

Please also refer the following for the details.
1. http://scdev.readthedocs.org/en/latest/mesh.html
2. get_blk_from_sample_neu of solvcon/testing.py

Usage:
    1. ./blk-to-vtk.py
    2. paraview blok-to-vtk.vtu
    3. click "apply" and then select "wireframe" to see the geometry
       in the paraview GUI.
'''
from solvcon.io.gambit import GambitNeutral
from solvcon.io.vtkxml import VtkXmlUstGridWriter


def openfile(filename, mode=None):
    """
    Open file with requested file name.

    The file name contains relative path
    to 'data' directory in this directory,
    and uses forward slash as delimiter
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
    from solvcon.conf import env
    path = [env.datadir] + filename.split('/')
    path = os.path.join(*path)
    if mode is not None:
        return open(path, mode)
    else:
        return open(path)


def loadfile(filename):
    """
    Load file with requested file name.

    The file name contains relative path
    to 'data' directory in this directory, and uses forward slash as delimiter
    of directory components.

    @param filename: path of file relative to 'data' directory
                     in this directory.
    @type filename: str
    @return: loaded data.
    @rtype: str
    """
    with openfile(filename) as fobj:
        data = fobj.read()
    return data

kw = {'fpdtype': None}
kw['use_incenter'] = None
block_sample = GambitNeutral(loadfile('sample.neu')).toblock(**kw)
iodev = VtkXmlUstGridWriter(block_sample)
iodev.write('block-to-vtk.vtu')
