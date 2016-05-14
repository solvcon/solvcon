# Copyright (c) 2012, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Allow the C-based :py:mod:`.mesh` module to call the C++ code in
:py:mod:`.march`.
"""

# Why do I have this module, and do not cimport march.pyx to mesh.pyx directly?
# Because I don't know how to (easily) make distutils to build src/*.c when
# mesh.pyx is in C++ mode.  Forcifully building src/*.c with a C++ compiler
# results into missing symbols because the difference of calling convention.  I
# probably can resolve it by monkey-patching distutils, or adding proper extern
# "C" to the Cython-generated .cpp file, but why do it so hackishly?  Therefore
# I add this intermediate wrapper to "shield" the C++ header from march.pyx, so
# that mesh.pyx can be built using a plain C compiler.  Once we finish porting
# solvcon.Block to C++, we can get rid of this shielding module.

from __future__ import absolute_import, division, print_function

from .march cimport Table
from ._march_bridge cimport Table
import numpy as np
cimport numpy as np

# Initialize NumPy.
np.import_array()


def check_block_pointers(blk):
    cdef Table _table
    cdef np.ndarray _cnda
    # not checking ghost arrays because they aren't contiguous and thus
    # their addresses can't match.
    cdef np.ndarray _sharr
    cdef np.ndarray _arr
    for name in blk.TABLE_NAMES:
        table = getattr(blk, 'tb'+name)
        sharr = getattr(blk, 'sh'+name)
        assert sharr.flags.c_contiguous
        arr = getattr(blk, name)
        _table = table
        _cnda = _table._nda
        _sharr = sharr
        _arr = arr
        msgs = []
        # shared.
        vals = <size_t>(_cnda.data), <size_t>(_sharr.data)
        if vals[0] != vals[1]:
            msgs.append('shared(%d,%d)' % vals)
        # body.
        vals = <size_t>(_table._core.row(0)), <size_t>(_arr.data)
        if vals[0] != vals[1]:
            msgs.append('body(%d,%d)' % vals)
        if msgs:
            tmpl = '%s array mismatch: %s'
            raise AttributeError(tmpl % (name, ', '.join(msgs)))

cdef void* get_table_bodyaddr(table):
    cdef Table _table = table
    if 0 == table.size:
        return NULL
    else:
        return <void*>(_table._core.row(0))

# vim: set fenc=utf8 ft=pyrex ff=unix nobomb et sw=4 ts=4 tw=79:
