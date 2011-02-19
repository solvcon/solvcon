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

from ctypes import byref, c_int, c_float, c_double, POINTER, Structure
# TODO: del byref
del c_int, c_float, c_double, POINTER
from .conf import env

def str_of(dtype):
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

# use scotch whenever possible.
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

class FortranType(Structure):
    """
    A modified ctypes Structure that can generate text for FORTRAN TYPE 
    definition.
    
    @note: ctypes is magic!
      1. You can't just override the base constructor. Somehow ctypes.Structure
         doesn't like it and gives wrong memory address.  If you do override,
         potential bug is there.  Use absorb() method to mimic customized
         constructor.
      2. Metaclassing Structure is dangerous.  Don't do that.
      3. Don't subclass TWICE.  Problems were experienced when there's another
         layer of inheritance.

    @cvar _fortran_name_: FORTRAN TYPE name.
    @ctype _fortran_name_: str
    @cvar typemapper: map from ctypes type to FORTRAN declaration type string.
    @ctype typemapper: dict
    """

    _fortran_name_ = None
    from ctypes import c_int, c_double
    typemapper = {
        c_int: 'integer*4',
        c_double: 'real*8',
    }
    del c_int, c_double

    def __str__(self):
        assert self._fortran_name_
        lst = []
        lst.append('type %s' % self._fortran_name_)
        for name, vartype in self._fields_:
            lst.append('    %s :: %s = %s' % (
                self.typemapper[vartype], name, getattr(self, name)))
        lst.append('end type %s' % self._fortran_name_)
        outdata = '\n'.join(lst)
        return outdata

    def to_text(self, out=None):
        """
        @keyword out: output file.  Can be None (output to no file).
        @type out: file
        @return: converted text.
        @rtype: str
        """
        assert self._fortran_name_

        lst = []
        lst.append('type %s' % self._fortran_name_)
        for name, vartype in self._fields_:
            lst.append('    %s :: %s' % (self.typemapper[vartype], name))
        lst.append('end type %s' % self._fortran_name_)
        outdata = '\n'.join(lst)

        if isinstance(out, str):
            out = open(out, 'w')
        if out:
            out.write(outdata)
        return outdata

    def absorb(self, another):
        """
        Absorb the fields value from another ctypes structure.

        @param another: another ftype/ctypes structure.
        @type another: FortranType
        @return: nothing.
        """
        myfields = self._fields_
        ivar = 0
        for key, vartype in another._fields_:
            # check.
            myname, mytype = myfields[ivar]
            assert myname == key
            assert mytype == vartype
            # set.
            val = getattr(another, key)
            setattr(self, key, val)
            ivar += 1

    def become(self, othertype):
        """
        Make object of another subclass and absort self to it.

        @param othertype: a subclass of self.
        @type othertype: type
        @return: absorbed other object.
        @rtype: FortranType
        """
        other = othertype()
        other.absorb(self)
        return other

del Structure
