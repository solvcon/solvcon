# -*- coding: UTF-8 -*-
#
# Copyright (C) 2014, Yung-Yu Chen <yyc@solvcon.net>
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
Gas-dynamics solvers.

The following example shows the content of this module:

>>> from solvcon.parcel import gas
>>> len(gas.__all__)
16
>>> [getattr(gas, nm) for nm in gas.__all__] # doctest: +NORMALIZE_WHITESPACE
[<class 'solvcon.parcel.gas.case.GasCase'>,
 <bound method CaseInfoMeta.register_arrangement of
  <class 'solvcon.parcel.gas.case.GasCase'>>,
 <class 'solvcon.parcel.gas.solver.GasSolver'>,
 <class 'solvcon.parcel.gas.boundcond.GasBC'>,
 <class 'solvcon.parcel.gas.boundcond.GasNonrefl'>,
 <class 'solvcon.parcel.gas.boundcond.GasWall'>,
 <class 'solvcon.parcel.gas.boundcond.GasInlet'>,
 <class 'solvcon.parcel.gas.probe.ProbeHook'>,
 <class 'solvcon.parcel.gas.physics.DensityInitAnchor'>,
 <class 'solvcon.parcel.gas.physics.PhysicsAnchor'>,
 <class 'solvcon.parcel.gas.inout.MeshInfoHook'>,
 <class 'solvcon.parcel.gas.inout.ProgressHook'>,
 <class 'solvcon.parcel.gas.inout.FillAnchor'>,
 <class 'solvcon.parcel.gas.inout.CflHook'>,
 <class 'solvcon.parcel.gas.inout.PMarchSave'>,
 <class 'solvcon.parcel.gas.oblique_shock.ObliqueShockRelation'>]
"""


import importlib


__all__ = []

def _include(names=None, frommod=None, fromobj=None, toall=True):
    """
    >>> _include(names=[], frommod=True, fromobj=True)
    Traceback (most recent call last):
        ...
    ValueError: only one of frommod or fromobj can be set
    >>> _include(names=[])
    Traceback (most recent call last):
        ...
    ValueError: either frommod or fromobj needs to be set
    """
    if frommod and fromobj:
        raise ValueError('only one of frommod or fromobj can be set')
    elif frommod:
        src = importlib.import_module(frommod, package='solvcon.parcel.gas')
    elif fromobj:
        src = fromobj
    else:
        raise ValueError('either frommod or fromobj needs to be set')
    for name in [] if names is None else names:
        globals()[name] = getattr(src, name)
        if toall and name not in __all__:
            __all__.append(name)

_include(names=['GasCase'], frommod='.case')
_include(names=['register_arrangement'], fromobj=GasCase)
_include(names=['GasSolver'], frommod='.solver')
_include(names=['GasBC', 'GasNonrefl', 'GasWall', 'GasInlet'],
         frommod='.boundcond')
_include(names=['ProbeHook'], frommod='.probe')
_include(names=['DensityInitAnchor', 'PhysicsAnchor'], frommod='.physics')
_include(names=['MeshInfoHook', 'ProgressHook', 'FillAnchor', 'CflHook',
                'PMarchSave'], frommod='.inout')
_include(names=['ObliqueShockRelation'], frommod='.oblique_shock')

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
