#!/usr/bin/env python2.6
# -*- coding: UTF-8 -*-
#
# Copyright (C) 2011 Yung-Yu Chen <yyc@solvcon.net>.
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

_libs = dict()
def get_lib(path):
    from ctypes import CDLL
    if path in _libs:
        lib = _libs[path]
    else:
        lib = _libs[path] = CDLL('libnetcdf.so')
    return lib

class NetCDF(object):
    """
    Wrapper for the netCDF library by using ctypes.  Mainly designed for
    reading.  Native functions remains to be nc_*.  NC_* are class members for
    constants defined in the header file.
    """

    # constants.
    NC_NOWRITE = 0
    NC_GLOBAL = -1

    NC_NOERR = 0
    NC_ENOTATT = -43

    def __init__(self, path=None, omode=None):
        """
        @keyword path: the file to open.
        @type path: str
        @keyword omode: opening mode.
        @type omode: int
        """
        from ctypes import c_char_p
        self.lib = get_lib('libnetcdf.so')
        self.ncid = None
        # set up return type.
        self.nc_strerror.restype = c_char_p
        self.nc_inq_libvers.restype = c_char_p
        # open file.
        if path is not None:
            self.open_file(path, omode)
    def __getattr__(self, key):
        if key.startswith('nc_'):
            return getattr(self.lib, key)

    def open_file(self, path, omode=None):
        """
        Open a NetCDF file.

        @keyword path: the file to open.
        @type path: str
        @keyword omode: opening mode.
        @type omode: int
        @return: ncid
        @rtype: int
        """
        from ctypes import c_int, byref
        omode = self.NC_NOWRITE if omode is None else omode
        if self.ncid is not None:
            raise IOError('ncid %d has been opened'%self.ncid)
        ncid = c_int()
        retval = self.nc_open(path, omode, byref(ncid))
        if retval != self.NC_NOERR:
            raise IOError(self.nc_strerror(retval))
        self.ncid = ncid.value
        return self.ncid
    def close_file(self):
        """
        Close the associated NetCDF file.

        @return: nothing
        """
        retval = self.nc_close(self.ncid)
        if retval != self.NC_NOERR:
            raise IOError(self.nc_strerror(retval))
        self.ncid = None

    def get_dim(self, name):
        """
        Get the dimension of the given name.

        @param name: the name of the dimension.
        @type name: str
        @return: the dimension (length).
        @rtype: int
        """
        from ctypes import c_int, c_long, byref
        dimid = c_int()
        length = c_long()
        retval = self.nc_inq_dimid(self.ncid, name, byref(dimid))
        if retval != self.NC_NOERR:
            raise IOError(self.nc_strerror(retval))
        retval = self.nc_inq_dimlen(self.ncid, dimid, byref(length))
        if retval != self.NC_NOERR:
            raise IOError(self.nc_strerror(retval))
        return length.value

    def get_array(self, name, shape, dtype):
        """
        Load ndarray from netCDF file.

        @param name: the data to be loaded.
        @type name: str
        @param shape: the shape of ndarray.
        @type shape: tuple
        @param dtype: the dtype of ndarray.
        @type dtype: str
        @return: the loaded ndarray.
        @rtype: numpy.ndarray
        """
        from ctypes import POINTER, c_int, c_double, byref
        from numpy import empty
        # get value ID.
        varid = c_int()
        retval = self.nc_inq_varid(self.ncid, name, byref(varid))
        if retval != self.NC_NOERR:
            raise IOError(self.nc_strerror(retval))
        # prepare array and loader.
        arr = empty(shape, dtype=dtype)
        if dtype == 'int32':
            func = self.nc_get_var_int
            ptr = POINTER(c_int)
        elif dtype == 'float64':
            func = self.nc_get_var_double
            ptr = POINTER(c_double)
        else:
            raise TypeError('now surrport only int and double')
        # load array.
        retval = func(self.ncid, varid, arr.ctypes.data_as(ptr))
        if retval != self.NC_NOERR:
            raise IOError(self.nc_strerror(retval))
        return arr

def load():
    from ctypes import POINTER, c_double, c_int, byref
    from numpy import empty

    cdf = NetCDF('../../tmp/brick_with_hole.g')
    print cdf.nc_inq_libvers()

    ndim = cdf.get_dim('num_dim')
    nnode = cdf.get_dim('num_nodes')
    ncell = cdf.get_dim('num_elem')
    ndcrd = cdf.get_array('coord', (ndim, nnode), 'float64').T.copy()
    print ndim, nnode, ncell
    print ndcrd

    elem_map = cdf.get_array('elem_map', (ncell,), 'int32')
    print elem_map

    nelk = cdf.get_dim('num_el_blk')
    for ielk in range(nelk):
        nelm = cdf.get_dim('num_el_in_blk%d'%(ielk+1))
        clnnd = cdf.get_dim('num_nod_per_el%d'%(ielk+1))
        elems = cdf.get_array('connect%d'%(ielk+1), (nelm, clnnd), 'int32')
        print ielk+1, ':', nelm, clnnd, elems

    cdf.close_file()

if __name__ == '__main__':
    load()
