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
    Callback class to be invoked by :py:class:`MeshSolver
    <solvcon.solver.MeshSolver>` objects at various stages.
    """

    def __init__(self, svr, **kw):
        from . import solver # work around cyclic importation.
        if not isinstance(svr, solver.MeshSolver):
            raise TypeError('%s must be a %s' % (
                str(svr), str(solver.MeshSolver)))
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
    Sequence container for :py:class:`MeshAnchor` instances.
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
