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
This is a simple wrapper to netCDF C library.  This wrapper is designed to be
self-sufficient.  That is, this should be kept to be an one-file module.

The module is designed for reading rather than writing.  Only limited Pythonic
API is implemented.  All constants are taken from the netcdf.h file in the
official distribution (4.1.1).

For more information about netCDF, please refer to 
http://www.unidata.ucar.edu/software/netcdf/index.html
"""

_libs = dict()
def get_lib(path):
    import sys, os
    from ctypes import CDLL
    from ..dependency import guess_dllname
    if path in _libs:
        lib = _libs[path]
    else:
        fn = guess_dllname('netcdf')
        try:
            lib = CDLL(fn)
        except OSError as e:
            # XXX: not a good practice to assume library location. Take it as a
            # dirty workaround in case netcdf library isn't at where it should
            # be.
            fn = os.path.join(sys.exec_prefix, 'lib', fn)
            lib = CDLL(fn)
        _libs[path] = lib
    return lib

class NetCDF(object):
    """
    Wrapper for the netCDF library by using ctypes.  Mainly designed for
    reading.  Native functions remains to be nc_*.  NC_* are class members for
    constants defined in the header file.
    """

    # netcdf external data types
    NC_NAT = 0  # Not A Type.
    NC_BYTE = 1	# signed 1 byte integer.
    NC_CHAR = 2 # ISO/ASCII character.
    NC_SHORT = 3    # signed 2 byte integer.
    NC_INT = 4 # signed 4 byte integer.
    NC_LONG = NC_INT    # deprecated, but required for backward compatibility.
    NC_FLOAT = 5    # single precision floating point number.
    NC_DOUBLE = 6   # double precision floating point number.
    NC_UBYTE = 7    # unsigned 1 byte int.
    NC_USHORT = 8   # unsigned 2-byte int.
    NC_UINT = 9 # unsigned 4-byte int.
    NC_INT64 = 10   # signed 8-byte int.
    NC_UINT64 = 11  # unsigned 8-byte int.
    NC_STRING = 12  # string.
    # used internally in support of user-defines types. They are also the class
    # returned by nc_inq_user_type.
    NC_VLEN = 13    # used internally for vlen types.
    NC_OPAQUE = 14  # used internally for opaque types.
    NC_ENUM = 15    # used internally for enum types.
    NC_COMPOUND = 16    # used internally for compound types.

    # Default fill values.
    NC_FILL_BYTE = -127
    NC_FILL_CHAR = 0
    NC_FILL_SHORT = -32767
    NC_FILL_INT = -2147483647L
    NC_FILL_FLOAT = 9.9692099683868690e+36 # near 15 * 2^119.
    NC_FILL_DOUBLE = 9.9692099683868690e+36
    NC_FILL_UBYTE = 255
    NC_FILL_USHORT = 65535
    NC_FILL_UINT = 4294967295
    NC_FILL_INT64 = -9223372036854775806
    NC_FILL_UINT64 = 18446744073709551614
    NC_FILL_STRING = ''

    # max and min.
    NC_MAX_BYTE = 127
    NC_MIN_BYTE = -NC_MAX_BYTE - 1
    NC_MAX_CHAR = 255
    NC_MAX_SHORT = 32767
    NC_MIN_SHORT = -NC_MAX_SHORT - 1
    NC_MAX_INT = 2147483647
    NC_MIN_INT = -NC_MAX_INT - 1
    NC_MAX_FLOAT = 3.402823466e+38
    NC_MIN_FLOAT = -NC_MAX_FLOAT
    NC_MAX_DOUBLE = 1.7976931348623157e+308
    NC_MIN_DOUBLE = -NC_MAX_DOUBLE
    NC_MAX_UBYTE = NC_MAX_CHAR
    NC_MAX_USHORT = 65535
    NC_MAX_UINT = 4294967295
    NC_MAX_INT64 = 9223372036854775807
    NC_MIN_INT64 = -9223372036854775807 - 1
    NC_MAX_UINT64 = 18446744073709551615
    X_INT64_MAX = 9223372036854775807
    X_INT64_MIN = -X_INT64_MAX - 1
    X_UINT64_MAX = 18446744073709551615

    # for fill.
    _FillValue = '_FillValue'
    NC_FILL = 0
    NC_NOFILL = 0x100

    # mode for nc_open
    NC_NOWRITE = 0  # default read-only.
    NC_WRITE = 0x0001   # read and write.
    # mode for nc_create
    NC_CLOBBER = 0
    NC_NOCLOBBER = 0x0004   # don't descroy existing file.
    NC_64BIT_OFFSET = 0x0200    # use 64-bit (large) file offsets.
    NC_NETCDF4 = 0x1000 # use netCDF-4/HDF5 format.
    NC_CLASSICAL_MODE = 0x0100  # enforce classic model with NC_NETCDF4.
    # mode for both nc_open and nc_create.
    NC_SHARE = 0x0800   # share updates, limit cache.
    NC_MPIIO = 0x2000
    NC_MPIPOSIX = 0x4000
    NC_PNETXCDF = 0x8000
    # future mode for nc_open and nc_create
    NC_LOCK = 0x0400

    # formats; there are more than one format since 3.6.
    NC_FORMAT_CLASSIC = 1
    NC_FORMAT_64BIT = 2
    NC_FORMAT_NETCDF4 = 3
    NC_FORMAT_NETCDF4_CLASSIC = 4

    # for nc__open and nc__create.
    NC_SIZEHINT_DEFAULT = 0

    # in nc__enddef, align to chunk size.
    NC_ALIGN_CHUNK = -1

    # size argument to ncdimdef for unlimit.
    NC_UNLIMITED = 0

    # attribute ID for global attributes.
    NC_GLOBAL = -1

    # interfacial max; nothing to do with netCDF internal implementation.
    NC_MAX_DIMS = 1024  # max dimensions per file.
    NC_MAX_ATTRS = 8192 # max global or per variable attributes.
    NC_MAX_VARS = 8192  # max variables per file.
    NC_MAX_NAME = 256   # max length of a name.
    NC_MAX_VAR_DIMS = NC_MAX_DIMS   # max per variable dimensions.

    # endianness for HDF5 files.
    NC_ENDIAN_NATIVE = 0
    NC_ENDIAN_LITTLE = 1
    NC_ENDIAN_BIG = 2
    # chunk or continuous for HDF5 files.
    NC_CHUNKED = 0
    NC_CONTIGUOUS = 1
    # checksum for HDF5 files.
    NC_NOCHECKSUM = 0
    NC_FLETCHER32 = 1
    # shuffle for HDF5 files.
    NC_NOSHUFFLE = 0
    NC_SHUFFLE = 1

    # errors.
    NC_NOERR = 0    # no error.

    NC2_ERR = -1    # for all errors in the v2 API.
    NC_EBADID = -33 # Not a netcdf id.
    NC_ENFILE = -34 # Too many netcdfs open.
    NC_EEXIST = -35 # netcdf file exists && NC_NOCLOBBER.
    NC_EINVAL = -36 # Invalid Argument.
    NC_EPERM = -37  # Write to read only.
    NC_ENOTINDEFINE = -38   # Operation not allowed in data mode.
    NC_EINDEFINE = -39  # Operation not allowed in define mode.
    NC_EINVALCOORDS = -40   # Index exceeds dimension bound.
    NC_EMAXDIMS = -41   # NC_MAX_DIMS exceeded.
    NC_ENAMEINUSE = -42 # String match to name in use.
    NC_ENOTATT = -43    # Attribute not found.
    NC_EMAXATTS = -44   # NC_MAX_ATTRS exceeded.
    NC_EBADTYPE = -45   # Not a netcdf data type.
    NC_EBADDIM = -46    # Invalid dimension id or name.
    NC_EUNLIMPOS = -47  # NC_UNLIMITED in the wrong index.
    NC_EMAXVARS = -48   # NC_MAX_VARS exceeded.
    NC_ENOTVAR = -49    # Variable not found.
    NC_EGLOBAL = -50    # Action prohibited on NC_GLOBAL varid.
    NC_ENOTNC = -51 # Not a netcdf file.
    NC_ESTS = -52   # In Fortran, string too short.
    NC_EMAXNAME = -53   # NC_MAX_NAME exceeded.
    NC_EUNLIMIT = -54   # NC_UNLIMITED size already in use.
    NC_ENORECVARS = -55 # nc_rec op when there are no record vars.
    NC_ECHAR = -56  # Attempt to convert between text & numbers.
    NC_EEDGE = -57  # Start+count exceeds dimension bound.
    NC_ESTRIDE = -58    # Illegal stride.
    NC_EBADNAME = -59   # Attribute or variable name contains illegal
                        # characters.

    # following must match value in ncx.h
    NC_ERANGE = -60 # Math result not representable.
    NC_ENOMEM = -61 # Memory allocation (malloc) failure.
    NC_EVARSIZE = -62   # One or more variable sizes violate format
                        # constraints.
    NC_EDIMSIZE = -63   # Invalid dimension size.
    NC_ETRUNC = -64 # File likely truncated or possibly corrupted.
    NC_EAXISTYPE = -65  # Unknown axis type.

    # following errors are added for DAP.
    NC_EDAP = -66   # Generic DAP error.
    NC_ECURL = -67  # Generic libcurl error.
    NC_EIO = -68    # Generic IO error.
    NC_ENODATA = -69    # Attempt to access variable with no data.
    NC_EDAPSVC = -70    # DAP Server side error.
    NC_EDAS = -71   # Malformed or inaccessible DAS.
    NC_EDDS = -72   # Malformed or inaccessible DDS.
    NC_EDATADDS = -73   # Malformed or inaccessible DATADDS.
    NC_EDAPURL = -74    # Malformed DAP URL.
    NC_EDAPCONSTRAINT = -75 # Malformed DAP Constraint.

    # following was added in support of netcdf-4. Make all netcdf-4 error codes
    # < -100 so that errors can be added to netcdf-3 if needed.
    NC4_FIRST_ERROR = -100
    NC_EHDFERR = -101   # Error at HDF5 layer.
    NC_ECANTREAD = -102 # Can't read.
    NC_ECANTWRITE = -103    # Can't write.
    NC_ECANTCREATE = -104   # Can't create.
    NC_EFILEMETA = -105 # Problem with file metadata.
    NC_EDIMMETA = -106  # Problem with dimension metadata.
    NC_EATTMETA = -107  # Problem with attribute metadata.
    NC_EVARMETA = -108  # Problem with variable metadata.
    NC_ENOCOMPOUND = -109   # Not a compound type.
    NC_EATTEXISTS = -110    # Attribute already exists.
    NC_ENOTNC4 = -111   # Attempting netcdf-4 operation on netcdf-3 file.
    NC_ESTRICTNC3 = -112    # Attempting netcdf-4 operation on strict nc3
                            # netcdf-4 file.
    NC_ENOTNC3 = -113   # Attempting netcdf-3 operation on netcdf-4 file.
    NC_ENOPAR = -114    # Parallel operation on file opened for non-parallel
                        # access.
    NC_EPARINIT = -115  # Error initializing for parallel access.
    NC_EBADGRPID = -116 # Bad group ID.
    NC_EBADTYPID = -117 # Bad type ID.
    NC_ETYPDEFINED = -118   # Type has already been defined and may not be
                            # edited.
    NC_EBADFIELD = -119 # Bad field ID.
    NC_EBADCLASS = -120 # Bad class.
    NC_EMAPTYPE = -121  # Mapped access for atomic types only.
    NC_ELATEFILL = -122 # Attempt to define fill value when data already
                        # exists.
    NC_ELATEDEF = -123  # Attempt to define var properties, like deflate, after
                        # enddef.
    NC_EDIMSCALE = -124 # Probem with HDF5 dimscales.
    NC_ENOGRP = -125    # No group found.
    NC_ESTORAGE = -126  # Can't specify both contiguous and chunking.
    NC_EBADCHUNK = -127 # Bad chunksize.
    NC_ENOTBUILT = -128 # Attempt to use feature that was not turned on when
                        # netCDF was built.
    NC4_LAST_ERROR = -128

    def __init__(self, path=None, omode=None, libname='libnetcdf.so'):
        """
        @keyword path: the file to open.
        @type path: str
        @keyword omode: opening mode.
        @type omode: int
        """
        from ctypes import c_char_p
        self.lib = get_lib(libname)
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
        from ctypes import POINTER, c_int, c_float, c_double, byref
        from numpy import empty, zeros
        from solvcon.helper import info
        # get value ID.
        varid = c_int()
        retval = self.nc_inq_varid(self.ncid, name, byref(varid))
        if retval != self.NC_NOERR:
            if retval == self.NC_ENOTVAR and name == "elem_map":
                ## Quick Fix for Pointwise:
                ## Poitwise does not provide an elem_map but as this is not
                ## required under normal use, it should not terminate the
                ## simulation
                info("Could not find the elem_map variable in the mesh file\n")
                info("Setting elem_map Array to 0\n")
                arr = zeros(shape, dtype=dtype)
                return arr
            else:
                raise IOError(self.nc_strerror(retval))
        
        # prepare array and loader.
        arr = empty(shape, dtype=dtype)
        if dtype == 'int32':
            func = self.nc_get_var_int
        elif dtype == 'float32':
            func = self.nc_get_var_float
        elif dtype == 'float64':
            func = self.nc_get_var_double
        else:
            raise TypeError('now support only int, float, double, and char')
        # load array.
        retval = func(self.ncid, varid, arr.ctypes._as_parameter_)
        if retval != self.NC_NOERR:
            raise IOError(self.nc_strerror(retval))
        return arr

    def get_lines(self, name, shape):
        """
        Load string from netCDF file.

        @param name: the data to be loaded.
        @type name: str
        @param shape: the shape of ndarray.  Must be 1 or 2.
        @type shape: tuple
        @return: the loaded ndarray.
        @rtype: numpy.ndarray
        """
        from ctypes import POINTER, c_int, c_char, byref
        from numpy import empty
        if len(shape) > 2:
            raise IndexError('array should have no more than two dimension')
        # get value ID.
        varid = c_int()
        retval = self.nc_inq_varid(self.ncid, name, byref(varid))
        if retval != self.NC_NOERR:
            raise IOError(self.nc_strerror(retval))
        # prepare array and loader.
        arr = empty(shape, dtype='byte')
        # load string data.
        retval = self.nc_get_var_text(self.ncid, varid,
            arr.ctypes._as_parameter_)
        if retval != self.NC_NOERR:
            raise IOError(self.nc_strerror(retval))
        # convert to string.
        shape = (1, shape[0]) if len(shape) < 2 else shape
        arr = arr.reshape(shape)
        lines = []
        for ii in range(shape[0]):
            for ij in range(shape[1]):
                if arr[ii,ij] == 0:
                    lines.append(arr[ii,:ij].tostring())
                    break
            if len(lines) <= ii:
                lines.append(arr[ii,:].tostring())
        return lines
