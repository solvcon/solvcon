# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Pilot features for the multi-dimensional Euler solver: the oblique-shock
sample meshes and driver, the shared scalar-field rendering, and the
interactive Euler solver panel.
"""

from .. import _pilot_core as _pcore

if _pcore.enable:
    from . import _field_render  # noqa: F401
    from . import _oblique
    from . import _solution_info

    ObliqueShockMesh = _oblique.ObliqueShockMesh
    SolutionInfo = _solution_info.SolutionInfo
else:
    _field_render = None
    _oblique = None
    _solution_info = None
    ObliqueShockMesh = None
    SolutionInfo = None

__all__ = [
    'ObliqueShockMesh',
    'SolutionInfo',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
