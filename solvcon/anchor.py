# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2013 Yung-Yu Chen <yyc@solvcon.net>.
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
The family of :py:class:`MeshAnchor` classes are attached to
:py:class:`MeshSolver <solvcon.solver.MeshSolver>`.
"""


# import legacy.
from .anchor_legacy import(
    Anchor, AnchorList,
    MarchSaveAnchor, VtkAnchor, RuntimeStatAnchor, MarchStatAnchor,
    TpoolStatAnchor, FillAnchor, GlueAnchor)


class MeshAnchor(object):
    """
    Anchor that called by solver objects at various stages.

    @ivar svr: the solver object to be attached to.
    @itype svr: solvcon.solver.Solver
    @ivar kws: excessive keywords.
    @itype kws: dict
    """

    def __init__(self, svr, **kw):
        from . import solver # work around cyclic importation.
        assert isinstance(svr, solver.MeshSolver)
        self.svr = svr
        self.kws = dict(kw)

    def provide(self):
        pass
    def preloop(self):
        pass
    def premarch(self):
        pass
    def prefull(self):
        pass
    def presub(self):
        pass
    def postsub(self):
        pass
    def postfull(self):
        pass
    def postmarch(self):
        pass
    def postloop(self):
        pass
    def exhaust(self):
        pass

class MeshAnchorList(list):
    """
    Anchor container and invoker.

    @ivar svr: solver object.
    @itype svr: solvcon.solver.BaseSolver
    """

    def __init__(self, svr, *args, **kw):
        self.svr = svr
        self.names = dict()
        super(MeshAnchorList, self).__init__(*args, **kw)

    def append(self, obj, **kw):
        name = kw.pop('name', None)
        if isinstance(name, int):
            raise ValueError('name can\'t be integer')
        if isinstance(obj, type):
            obj = obj(self.svr, **kw)
        super(MeshAnchorList, self).append(obj)
        if name != None:
            self.names[name] = obj

    def __getitem__(self, key):
        if key in self.names:
            return self.names[key]
        else:
            return super(MeshAnchorList, self).__getitem__(key)

    def __call__(self, method):
        """
        Invoke the specified method for each anchor.
        
        @param method: name of the method to run.
        @type method: str
        @return: nothing
        """
        runanchors = self.svr.runanchors
        if method == 'postloop' or method == 'exhaust':
            runanchors = reversed(runanchors)
        for anchor in runanchors:
            func = getattr(anchor, method, None)
            if func != None:
                func()
