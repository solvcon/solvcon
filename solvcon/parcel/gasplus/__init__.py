# -*- coding: UTF-8 -*-
#
# Copyright (c) 2016, Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Gas-dynamics solvers based on new C++ implementation.
"""


from __future__ import absolute_import, division, print_function


__all__ = []


import importlib


def _include(names=None, frommod=None, fromobj=None, toall=True):
    """
    >>> _include(names=[], frommod=True, fromobj=True)
    Traceback (most recent call last):
        ...
    ValueError: either frommod or fromobj is allowed
    >>> _include(names=[])
    Traceback (most recent call last):
        ...
    ValueError: need either frommod or fromobj
    """
    if frommod and fromobj:
        raise ValueError('either frommod or fromobj is allowed')
    elif frommod:
        src = importlib.import_module(frommod, package='solvcon.parcel.gasplus')
    elif fromobj:
        src = fromobj
    else:
        raise ValueError('need either frommod or fromobj')
    for name in [] if names is None else names:
        globals()[name] = getattr(src, name)
        if toall and name not in __all__:
            __all__.append(name)

_include(names=['GasPlusCase'], frommod='.case')
_include(names=['register_arrangement'], fromobj=GasPlusCase)
_include(names=['GasPlusSolver'], frommod='.solver')
_include(names=['GasPlusBC', 'GasPlusNonRefl', 'GasPlusSlipWall', 'GasPlusInlet'],
         frommod='.boundcond')
_include(names=['ProbeHook'], frommod='.probe')
_include(
    names=[
        'InitByDensityTemperatureAnchor', 'DensityInitAnchor', 'PhysicsAnchor'
    ],
    frommod='.physics')
_include(names=['MeshInfoHook', 'ProgressHook', 'FillAnchor', 'CflHook',
                'PMarchSave'], frommod='.inout')
_include(names=['ObliqueShockRelation'], frommod='.oblique_shock')

# vim: set ff=unix fenc=utf8 ft=python nobomb et sw=4 ts=4 tw=79:
