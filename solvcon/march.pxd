# Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING

from libcpp.vector cimport vector

cdef extern from "march/march.hpp" namespace "march::mesh" nogil:

    cdef cppclass LookupTableCore:
        LookupTableCore(int, int, vector[int]&, int, char *) except+
        int nghost()
        int nbody()
        int ncolumn()
        size_t nbyte()
        char * row(int loc)
        char * data()

cdef class Table:
    cdef LookupTableCore * _core
    cdef readonly object _nda

# vim: set fenc=utf8 ft=pyrex ff=unix nobomb et sw=4 ts=4 tw=79:
