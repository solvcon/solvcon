# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Pilot features for the multi-dimensional Euler solver: the oblique-shock
sample meshes.
"""

from .. import _pilot_core as _pcore

if _pcore.enable:
    from . import _oblique

    ObliqueShockMesh = _oblique.ObliqueShockMesh
else:
    _oblique = None
    ObliqueShockMesh = None

__all__ = [
    'ObliqueShockMesh',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
