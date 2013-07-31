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
MPY stands for "MPi for pYthon", is a single-module wrapper for any MPI
library.  Just copy the mpy.py then it works (needless to say, after you have
MPI installed on a cluster).  MPY has no external dependency other than a
working MPI installation and a standard Python.  The MPI installation has to
be built with shared object since MPY uses ctypes as interface.

All the functions in the MPI library can be accessed by
MPI().Name_Without_Leading_MPI(), and you must manually convert the arguments
from Python objects to ctypes objects.  Shorthand APIs and Pythonic APIs are
also provided.  Shorthand APIs use Python objects and can return Python
objects, with the same naming convention as MPI, but all lower-cased.  Pythonic
APIs are Pythonic.

You can verify the "installation" of MPY by running::

  $ mpiexec -n 2 python mpy.py

"""

__version__ = '0.1+'

__description__ = """MPI for Python."""

_libs = dict()
def get_lib(path):
    from ctypes import CDLL
    if path in _libs:
        lib = _libs[path]
    else:
        lib = _libs[path] = CDLL('libmpich.so')
    return lib

class MPI(object):
    """
    Wrapper for MPI library.  The leading 'MPI_' is stripped off from the
    name of all MPI entities.  Shorthand and Pythonic APIs are provided with
    all lower-cased name.  All MPI constants are Python int() and filled
    according to the mpi.h of MVAPICH2-1.5.  ctypes is used for calling
    dynamically linked MPI libraries.
    """

    # Null objects.
    COMM_NULL = 0x04000000
    OP_NULL = 0x18000000
    GROUP_NULL = 0x08000000
    DATATYPE_NULL = 0x0c000000
    REQUEST_NULL = 0x2c000000
    ERRHANDLER_NULL = 0x14000000

    # Results of the compare operations.
    IDENT = 0
    CONGRUENT = 1
    SIMILAR = 2
    UNEQUAL = 3

    # Data types.
    CHAR = 0x4c000101
    SIGNED_CHAR = 0x4c000118
    UNSIGNED_CHAR = 0x4c000102
    BYTE = 0x4c00010d
    WCHAR = 0x4c00040e
    SHORT = 0x4c000203
    UNSIGNED_SHORT = 0x4c000204
    INT = 0x4c000405
    UNSIGNED = 0x4c000406
    LONG = 0x4c000807
    UNSIGNED_LONG = 0x4c000808
    FLOAT = 0x4c00040a
    DOUBLE = 0x4c00080b
    LONG_DOUBLE = 0x4c00080c
    LONG_LONG_INT = 0x4c000809
    UNSIGNED_LONG_LONG = 0x4c000819
    LONG_LONG = LONG_LONG_INT
    #
    PACKED = 0x4c00010f
    LB = 0x4c000010
    UB = 0x4c000011
    #
    FLOAT_INT = 0x8c000000
    DOUBLE_INT = 0x8c000001
    LONG_INT = 0x8c000002
    SHORT_INT = 0x8c000003
    _2INT = 0x4c000816
    LONG_DOUBLE_INT = 0x8c000004
    # FORTRAN types.
    COMPLEX = 1275070494
    DOUBLE_COMPLEX = 1275072546
    LOGICAL = 1275069469
    REAL = 1275069468
    DOUBLE_PRECISION = 1275070495
    INTEGER = 1275069467
    _2INTEGER = 1275070496
    _2COMPLEX = 1275072548
    _2DOUBLE_COMPLEX = 1275076645
    _2REAL = 1275070497
    _2DOUBLE_PRECISION = 1275072547
    CHARACTER = 1275068698
    # Size-specific data types.
    REAL4 = 0x4c000427
    REAL8 = 0x4c000829
    REAL16 = DATATYPE_NULL
    COMPLEX8 = 0x4c000828
    COMPLEX16 = 0x4c00102a
    COMPLEX32 = DATATYPE_NULL
    INTEGER1 = 0x4c00012d
    INTEGER2 = 0x4c00022f
    INTEGER4 = 0x4c000430
    INTEGER8 = 0x4c000831
    INTEGER16 = DATATYPE_NULL
    # C99 fixed-width data types.
    INT8_T = 0x4c000137
    INT16_T = 0x4c000238
    INT32_T = 0x4c000439
    INT64_T = 0x4c00083a
    UINT8_T = 0x4c00013b
    UINT16_T = 0x4c00023c
    UINT32_T = 0x4c00043d
    UINT64_T = 0x4c00083e
    # Other C99 data types.
    C_BOOL = 0x4c00013f
    C_FLOAT_COMPLEX = 0x4c000840
    C_COMPLEX = C_FLOAT_COMPLEX
    C_DOUBLE_COMPLEX = 0x4c001041
    C_LONG_DOUBLE_COMPLEX = 0x4c001042
    # Address types.
    AINT = 0x4c000843
    OFFSET = 0x4c000844
    # Type classes.
    TYPECLASS_REAL = 1
    TYPECLASS_INTEGER = 2
    TYPECLASS_COMPLEX = 3

    # Communicators.
    COMM_WORLD = 0x44000000
    COMM_SELF = 0x44000001

    # Groups.
    GROUP_EMPTY = 0x48000000

    # RMA and Windows.
    WIN_NULL = 0x20000000

    # File.
    FILE_NULL = 0

    # Collective operations.
    MAX = 0x58000001
    MIN = 0x58000002
    SUM = 0x58000003
    PROD = 0x58000004
    LAND = 0x58000005
    BAND = 0x58000006
    LOR = 0x58000007
    BOR = 0x58000008
    LXOR = 0x58000009
    BXOR = 0x5800000a
    MINLOC = 0x5800000b
    MAXLOC = 0x5800000c
    REPLACE = 0x5800000d

    # Permanent key values.
    TAG_UB = 0x64400001
    HOST = 0x64400003
    IO = 0x64400005
    WTIME_IS_GLOBAL = 0x64400007
    UNIVERSE_SIZE = 0x64400009
    LASTUSEDCODE = 0x6440000b
    APPNUM = 0x6440000d

    # The 3 predefined window attributes for every window.
    WIN_BASE = 0x66000001
    WIN_SIZE = 0x66000003
    WIN_DISP_UNIT = 0x66000005

    # Guessed values.
    MAX_PROCESSOR_NAME = 128
    MAX_ERROR_STRING = 1024
    MAX_PORT_NAME = 256
    MAX_OBJECT_NAME = 128

    # Predefined constants.
    UNDEFINED = -32766
    KEYVAL_INVALID = 0x24000000

    # Upper bound on the overhead in bsend for each message buffer.
    BSEND_OVERHEAD = 88

    # Topology types:
    BOTTOM = 0
    UNWEIGHTED = 0

    PROC_NULL = -1
    ANY_SOURCE = -2
    ROOT = -3
    ANY_TAG = -1

    LOCK_EXCLUSIVE = 234
    LOCK_SHARED = 235

    # Built in error handlers.
    ERRORS_ARE_FATAL = 0x54000000
    ERRORS_RETURN = 0x54000001

    # MPI-1.
    NULL_COPY_FN = 0
    NULL_DELETE_FN = 0
    # MPI-2.
    COMM_NULL_COPY_FN = 0
    COMM_NULL_DELETE_FN = 0
    WIN_NULL_COPY_FN = 0
    WIN_NULL_DELETE_FN = 0
    TYPE_NULL_COPY_FN = 0
    TYPE_NULL_DELETE_FN = 0

    # Info.
    INFO_NULL = 0x1c000000
    MAX_INFO_KEY = 255
    MAX_INFO_VAL = 1024

    # Subarray and darray constructors.
    ORDER_C = 56
    ORDER_FORTRAN = 57
    DISTRIBUTE_BLOCK = 121
    DISTRIBUTE_CYCLIC = 122
    DISTRIBUTE_NONE = 123
    DISTRIBUTE_DFLT_DARG = -49767

    IN_PLACE = -1

    # Asserts for one-sided communication.
    MODE_NOCHECK = 1024
    MODE_NOSTORE = 2048
    MODE_NOPUT = 4096
    MODE_NOPRECEDE = 8192
    MODE_NOSUCCEED = 16384

    STATUS_IGNORE = 1
    STATUSES_IGNORE = 1
    ERRCODES_IGNORE = 0

    ARGV_NULL = 0
    ARGVS_NULL = 0

    # Supported thread levels.
    THREAD_SINGLE = 0
    THREAD_FUNNELED = 1
    THREAD_SERIALIZED = 2
    THREAD_MULTIPLE = 3

    # Error classes.
    SUCCESS = 0
    # Communication argument parameters.
    ERR_BUFFER = 1
    ERR_COUNT = 2
    ERR_TYPE = 3
    ERR_TAG = 4
    ERR_COMM = 5
    ERR_RANK = 6
    ERR_ROOT = 7
    ERR_TRUNCATE = 14
    # MPI objects.
    ERR_GROUP = 8
    ERR_OP = 9
    ERR_REQUEST = 19
    # Special topology argument parameters.
    ERR_TOPOLOGY = 10
    ERR_DIMS = 11
    # All other arguments.
    ERR_ARG = 12
    # Other erros.
    ERR_OTHER = 15
    ERR_UNKNOWN = 13
    ERR_INTERN = 16
    # Multiple completion.
    ERR_IN_STATUS = 17
    ERR_PENDING = 18
    # New MPI-2 error classes.
    ERR_FILE = 27
    ERR_ACCESS = 20
    ERR_AMODE = 21
    ERR_BAD_FILE = 22
    ERR_FILE_EXISTS = 25
    ERR_FILE_IN_USE = 26
    ERR_NO_SPACE = 36
    ERR_NO_SUCH_FILE = 37
    ERR_IO = 32
    ERR_READ_ONLY = 40
    ERR_CONVERSION = 23
    ERR_DUP_DATAREP = 24
    ERR_UNSUPPORTED_DATAREP = 43
    # Info (oversight?).
    ERR_INFO = 28
    ERR_INFO_KEY = 29
    ERR_INFO_VALUE = 30
    ERR_INFO_NOKEY = 31
    #
    ERR_NAME = 33
    ERR_NO_MEM = 34
    ERR_NOT_SAME = 35
    ERR_PORT = 38
    ERR_QUOTA = 39
    ERR_SERVICE = 41
    ERR_SPAWN = 42
    ERR_UNSUPPORTED_OPERATION = 44
    ERR_WIN = 45
    #
    ERR_BASE = 46
    ERR_LOCKTYPE = 47
    ERR_KEYVAL = 48
    ERR_RMA_CONFLICT = 49
    ERR_RMA_SYNC = 50
    ERR_SIZE = 51
    ERR_DISP = 52
    ERR_ASSERT = 53
    #
    ERR_LASTCODE = 0x3fffffff
    #
    CONVERSION_FN_NULL = 0

    def __init__(self, initlib=True):
        self.lib = get_lib('libmpich.so')
        if initlib:
            self.Init(None, None)
    def __getattr__(self, key):
        return getattr(self.lib, 'MPI_'+key)
    @classmethod
    def _make_comm(cls, comm):
        """
        Make up communicator c_int based on input.
        """
        from ctypes import c_int
        return c_int(cls.COMM_WORLD if comm == None else comm)
    ############################################################################
    # Shorthand API.
    ############################################################################
    def comm_rank(self, comm=None):
        from ctypes import c_int, byref
        val = c_int(-1)
        self.Comm_rank(self._make_comm(comm), byref(val))
        return val.value
    def comm_size(self, comm=None):
        from ctypes import c_int, byref
        val = c_int(-1)
        self.Comm_size(self._make_comm(comm), byref(val))
        return val.value
    ############################################################################
    # Pythonic API.
    ############################################################################
    @property
    def initialized(self):
        from ctypes import c_int, byref
        val = c_int(-1)
        self.Initialized(byref(val))
        return bool(val.value)
    @property
    def rank(self):
        from ctypes import c_int, byref
        val = c_int(-1)
        self.Comm_rank(self._make_comm(None), byref(val))
        return val.value
    @property
    def size(self, comm=None):
        from ctypes import c_int, byref
        val = c_int(-1)
        self.Comm_size(self._make_comm(None), byref(val))
        return val.value

    def send(self, obj, dst, tag, comm=None):
        from cPickle import dumps
        from ctypes import c_int, c_char_p, byref
        comm = self.COMM_WORLD if comm is None else comm
        # dump obj.
        dat = dumps(obj, -1)
        self.Send(byref(c_int(len(dat))), c_int(1), c_int(self.INT),
            c_int(dst), c_int(tag), c_int(comm))
        self.Send(c_char_p(dat), c_int(len(dat)), c_int(self.BYTE),
            c_int(dst), c_int(tag), c_int(comm))
    def recv(self, src, tag, comm=None):
        from cPickle import loads
        from ctypes import c_int, byref, create_string_buffer
        comm = self.COMM_WORLD if comm is None else comm
        # load obj.
        dlen = c_int()
        status = c_int(0)
        self.Recv(byref(dlen), c_int(1), c_int(self.INT),
            c_int(src), c_int(tag), c_int(comm), byref(status))
        status.value = 0
        dat = create_string_buffer(dlen.value)
        self.Recv(byref(dat), dlen, c_int(self.BYTE),
            c_int(src), c_int(tag), c_int(comm), byref(status))
        obj = loads(dat.raw)
        return obj

    def sendarr(self, arr, dst, tag, comm=None):
        from ctypes import c_int, c_void_p
        comm = self.COMM_WORLD if comm is None else comm
        self.Send(arr.ctypes.data_as(c_void_p),
            c_int(arr.nbytes), c_int(self.BYTE),
            c_int(dst), c_int(tag), c_int(comm))
    def recvarr(self, arr, src, tag, comm=None):
        from ctypes import c_int, c_void_p, byref
        comm = self.COMM_WORLD if comm is None else comm
        status = c_int(0)
        self.Recv(arr.ctypes.data_as(c_void_p),
            c_int(arr.nbytes), c_int(self.BYTE),
            c_int(src), c_int(tag), c_int(comm), byref(status))

def main():
    import os, sys
    from random import choice, randint
    from socket import gethostname
    mpi = MPI()
    sys.stdout.write('rank %d/(%d-1) on %s %s\n' % (
        mpi.rank, mpi.size, gethostname(),
        'initialized' if mpi.initialized else 'uninitialized'))
    if mpi.rank == 0:
        msg = ''.join([choice('abcdefghijklmnopqrstuvwxyz')
            for it in range(randint(2, 20))])
        sys.stdout.write('rank %d on %s send: %s\n' % (
            mpi.rank, gethostname(), msg))
        mpi.send(msg, 1, 99)
    elif mpi.rank == 1:
        buf = mpi.recv(0, 99)
        sys.stdout.write('rank %d on %s recv: %s\n' % (
            mpi.rank, gethostname(), buf))
    else:
        sys.stdout.write('rank %d on %s do nothing\n' % (
            mpi.rank, gethostname()))
    mpi.Finalize()

if __name__ == '__main__':
    main()
