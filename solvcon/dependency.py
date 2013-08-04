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

def guess_dllname(dllname):
    """
    Guess the name for a shared object file based on the platform the code is
    running on.

    @param dllname: the original name of the shared object file.
    @type dllname: str
    @return: the guessed full name of the shared object file.
    @rtype: str
    """
    import sys
    if sys.platform.startswith('win'):
        tmpl = '%s.dll'
    elif sys.platform == 'darwin':
        tmpl = 'lib%s.dylib'
    else:
        tmpl = 'lib%s.so'
    return tmpl % dllname
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
    import os
    from ctypes import CDLL
    if None is location:
        return None
    libname = guess_dllname(libname)
    libdir = os.path.abspath(location)
    if not os.path.isdir(libdir):
        libdir = os.path.dirname(libdir)
    libpath = os.path.join(libdir, libname)
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

try:
    _clib_solvcon_d = getcdll('solvcon_d')
    _clib_solvcon_s = getcdll('solvcon_s')
except OSError:
    pass

def _clib_solvcon_of(dtype):
    import numpy as np
    if dtype == np.float32:
        return _clib_solvcon_s
    elif dtype == np.float64:
        return _clib_solvcon_d
    else:
        raise TypeError
