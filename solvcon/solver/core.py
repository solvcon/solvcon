# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2009 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
Core components for solvers.
"""

from ..gendata import TypeWithBinder

class BaseSolver(object):
    """
    Generic solver definition.  It is an abstract class and should not be used
    to any concrete simulation case.  The concrete solver sub-classes should
    override the empty init and final methods for initialization and 
    finalization, respectively.

    @ivar _fpdtype: dtype for the floating point data in the block instance.
    """

    __metaclass__ = TypeWithBinder

    _pointers_ = [] # for binder.

    def __init__(self, **kw):
        """
        @keyword fpdtype: dtype for the floating point data.
        """
        from ..conf import env
        self._fpdtype = kw.pop('fpdtype', env.fpdtype)
        self._fpdtype = env.fpdtype if self._fpdtype==None else self._fpdtype

    @property
    def fpdtype(self):
        import numpy
        _fpdtype = self._fpdtype
        if isinstance(_fpdtype, str):
            return getattr(numpy, _fpdtype)
        else:
            return self._fpdtype
    @property
    def fpdtypestr(self):
        from ..dependency import str_of
        return str_of(self.fpdtype)
    @property
    def fpptr(self):
        from ..dependency import pointer_of
        return pointer_of(self.fpdtype)
    @property
    def _clib_solvcon(self):
        from ..dependency import _clib_solvcon_of
        return _clib_solvcon_of(self.fpdtype)

    def init(self, **kw):
        """
        An empty initializer for the solver object.

        @return: nothing.
        """
        pass

    def final(self):
        """
        An empty finalizer for the solver object.

        @return: nothing.
        """
        pass
