# -*- coding: UTF-8 -*-
#
# Copyright (c) 2011, Yung-Yu Chen <yyc@solvcon.net>
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
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
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
This is a dumb wrapper to netCDF4 Python binding.  The purpose of this module
is to provide a compatibility layer to an old wrapper to netCDF C library,
which is removed.

For more information about netCDF, please refer to 
http://www.unidata.ucar.edu/software/netcdf/index.html
"""


from __future__ import absolute_import, division, print_function


import numpy as np

import solvcon as sc
sc.import_module_may_fail('netCDF4')

class NetCDF(object):
    """
    Wrapper for the netCDF library by using ctypes.  Mainly designed for
    reading.  Native functions remains to be nc_*.  NC_* are class members for
    constants defined in the header file.
    """

    def __init__(self, path=None, omode="r"):
        """
        :keyword path: the file to open.
        :type path: str
        :keyword omode: opening mode.
        :type omode: int
        """
        self.rootgroup = None
        if path is not None:
            self.open_file(path, omode)

    def open_file(self, path, omode="r"):
        """
        Open a NetCDF file.

        :keyword path: the file to open.
        :type path: str
        :keyword omode: opening mode.
        :type omode: str
        :return: Root group from the opened data set.
        :rtype: netCDF4.Dataset
        """
        self.root_group = netCDF4.Dataset(path, omode)
        return self.root_group

    def close_file(self):
        """
        Close the associated NetCDF file.

        :return: Nothing
        """
        self.root_group.close()

    def get_dim(self, name):
        """
        Get the dimension of the given name.

        :param name: the name of the dimension.
        :type name: str
        :return: the dimension (length).
        :rtype: int
        """
        return len(self.root_group.dimensions[name])

    def get_array(self, name, shape, dtype):
        """
        Load ndarray from netCDF file.

        :param name: the data to be loaded.
        :type name: str
        :param shape: the shape of ndarray.
        :type shape: tuple
        :param dtype: the dtype of ndarray.
        :type dtype: str
        :return: the loaded ndarray.
        :rtype: numpy.ndarray
        """
        try:
            var = self.root_group[name]
            arr = var[...]
        except IndexError:
            sc.helper.info("Could not find the elem_map variable in the mesh file\n")
            sc.helper.info("Setting elem_map Array to 0\n")
            arr = np.zeros(shape, dtype=dtype)
        assert isinstance(arr, np.ndarray)
        assert str(arr.dtype) == str(dtype)
        return arr

    def get_lines(self, name, shape):
        """
        Load string from netCDF file.

        :param name: the data to be loaded.
        :type name: str
        :param shape: the shape of ndarray.  Must be 1 or 2.
        :type shape: tuple
        :return: The loaded strings.
        :rtype: list of str
        """
        # Load variable.
        var = self.root_group[name]
        arr = var[...]
        assert isinstance(arr, np.ndarray)
        if len(shape) > 2:
            raise IndexError('array should have no more than two dimension')
        assert arr.shape == shape
        # Convert to list of strings.
        lines = list()
        for line in arr:
            idx = np.argwhere(line == b'').min()
            line = line[:idx].tobytes()
            lines.append(line.decode())
        return lines

    def get_attr(self, name, varname=None):
        """
        Get the attribute attached to an variable.  If *varname* is None
        (default), get the global attribute.

        :param name: name of the attribute.
        :type name: str
        :param varname: name of the variable.
        :type varname: str
        :return: the attribute.
        """
        if varname is None:
            return self.root_group.getncattr(name)
        else:
            return self.root_group[varname].getncattr(name)

    get_attr_int = get_attr
    get_attr_text = get_attr
