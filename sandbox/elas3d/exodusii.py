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

def load():
    from ctypes import POINTER, c_double, c_int, byref
    from numpy import empty
    from solvcon.io.netcdf import NetCDF

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
    import sys
    sys.path.insert(0, '../..')
    load()
