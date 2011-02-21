# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2011 Yung-Yu Chen <yyc@solvcon.net>.
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
Logic for using external compiled libraries.
"""

def str_of(dtype):
    """
    Determine the string representation of a dtype of floating-point.
    """
    import numpy as np
    for dtypestr in 'float64', 'float32':
        if dtype == getattr(np, dtypestr):
            return dtypestr
    raise TypeError

cdllcache = dict()
def loadcdll(location, libname):
    """
    Load shared objects using ctypes.  Loaded dll objects are cached to prevent
    duplicated loading.

    @param location: location of the ctypes library.
    @type location: str
    @param libname: full basename of library.
    @type libname: str
    @return: ctypes library.
    @rtype: ctypes.CDLL
    """
    import sys, os
    from ctypes import CDLL
    # initialize solver function.
    tmpl = '%s.dll' if sys.platform.startswith('win') else 'lib%s.so'
    libdir = os.path.abspath(location)
    if not os.path.isdir(libdir):
        libdir = os.path.dirname(libdir)
    libpath = os.path.join(libdir, tmpl%libname)
    return cdllcache.setdefault(libpath, CDLL(libpath))
def getcdll(libname, location=None, raise_on_fail=True):
    """
    Load shared objects at the default location.

    @param libname: main basename of library without sc_ prefix.
    @type libname: str
    @keyword location: location of the library.
    @type location: str
    @keyword raise_on_fail: raise the error on failing to load. Default True.
    @type raise_on_fail: bool
    @return: ctypes library.
    @rtype: ctypes.CDLL
    """
    import os
    from .conf import env
    location = env.libdir if location is None else location
    try:
        lib = loadcdll(location, 'sc_'+libname)
    except OSError:
        if raise_on_fail:
            raise
        else:
            lib = None
    return lib

_clib_solvcon_d = getcdll('solvcon_d')
_clib_solvcon_s = getcdll('solvcon_s')

def _clib_solvcon_of(dtype):
    import numpy as np
    if dtype == np.float32:
        return _clib_solvcon_s
    elif dtype == np.float64:
        return _clib_solvcon_d
    else:
        raise TypeError

# use SCOTCH-5.1 whenever possible.
_clib_metis = None
from ctypes import CDLL
for name in 'scotchmetis-5.1', 'scotchmetis':
    try:
        _clib_metis = CDLL('lib%s.so'%name)
    except OSError:
        pass
    if _clib_metis is not None:
        break
del CDLL
_clib_metis = getcdll('metis') if _clib_metis is None else _clib_metis
