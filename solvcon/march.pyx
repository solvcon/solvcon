# Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING

from __future__ import absolute_import, division, print_function

from libc.stdint cimport intptr_t
from libcpp.vector cimport vector
import numpy as np
cimport numpy as np
from .march cimport LookupTableCore
np.import_array()

cdef class Table:
    """
    Lookup table that allows ghost entity.
    """

    def __cinit__(self, *args, **kw):
        self._core = <LookupTableCore *>(NULL)

    def __dealloc__(self):
        if self._core != NULL:
            del self._core
            self._core = NULL

    def __init__(self, nghost, nbody, *args, **kw):
        cdef int elsize
        cdef np.dtype dtype = np.dtype(kw.pop("dtype", "int32"))
        cdef vector[int] dims = [nghost+nbody] + list(args)
        creator_name = kw.pop("creation", "empty")
        if creator_name not in ("empty", "zeros"):
            raise ValueError("invalid creation type")
        # Create the ndarray.
        create = getattr(np, creator_name)
        self._nda = create([nghost+nbody]+list(args), dtype=dtype)
        if not self._nda.flags.c_contiguous:
            raise ValueError("not C Contiguous")
        ndim = len(self._nda.shape)
        if ndim == 0:
            raise ValueError("zero dimension is not allowed")
        # Assign pointers.
        cdef np.ndarray cnda = self._nda
        self._core = new LookupTableCore(
            nghost, nbody, dims, dtype.itemsize, cnda.data)
        cdef vector[np.intp_t] pydims = [nghost+nbody] + list(args)
        if self._core.nbyte() != self._nda.nbytes:
            raise ValueError("nbyte mismatch")

    def __getattr__(self, name):
        return getattr(self._nda, name)

    @property
    def nghost(self):
        return self._core.nghost()

    @property
    def nbody(self):
        return self._core.nbody()

    @property
    def offset(self):
        """
        Element offset from the head of the ndarray to where the body starts.
        """
        return self._core.nghost() * self._core.ncolumn()

    @property
    def _ghostaddr(self):
        return <intptr_t>(self._core.data())

    @property
    def _bodyaddr(self):
        return <intptr_t>(self._core.row(0))

    property F:
        """Full array."""
        def __get__(self):
            return self._nda[...]
        def __set__(self, arr):
            self._nda[...] = arr

    property G:
        """Ghost-part array."""
        def __get__(self):
            if self.nghost:
                return self._nda[self.nghost-1::-1,...]
            else:
                return self._nda[0:0,...]
        def __set__(self, arr):
            if self.nghost:
                self._nda[self.nghost-1::-1,...] = arr
            else:
                raise IndexError('nghost is zero')

    @property
    def _ghostpart(self):
        """Ghost-part array without setter."""
        if self.nghost:
            return self._nda[self.nghost-1::-1,...]
        else:
            return self._nda[0:0,...]

    property B:
        """Body-part array."""
        def __get__(self):
            return self._nda[self.nghost:,...]
        def __set__(self, arr):
            self._nda[self.nghost:,...] = arr

    @property
    def _bodypart(self):
        """Body-part array without setter."""
        return self._nda[self.nghost:,...]

# vim: set fenc=utf8 ft=pyrex ff=unix nobomb et sw=4 ts=4 tw=79:
