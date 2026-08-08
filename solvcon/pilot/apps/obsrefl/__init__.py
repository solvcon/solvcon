# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The oblique-shock reflection solved with the multi-dimensional Euler solver.

The computing domain is a plain rectangle.  A uniform supersonic stream
enters horizontally from the left, the top boundary holds the state behind
an incident oblique shock, the bottom slip wall reflects that shock, and the
flow leaves through the non-reflective outflow on the right.

The driver, the analysis, and the session carry the problem itself and import
unconditionally, so a script or a no-Qt build runs and judges the reflection
without the GUI that rides behind the pilot toggle.
"""

from ... import _pilot_core as _pcore
from ._analytic import Reflection
from ._driver import ObliqueShock, ObliqueShockMesher, ObliqueShockRelation
from ._session import ReflectionSession

if _pcore.enable:
    from ._app import ObliqueShockApp
    from ._mesh_sample import ObliqueShockMesh
    from ._panel import SolutionPanel
else:
    # Bind only the public names: a None module attribute would shadow the
    # real submodule import in no-GUI builds.
    ObliqueShockApp = None
    ObliqueShockMesh = None
    SolutionPanel = None

__all__ = [
    'ObliqueShock',
    'ObliqueShockApp',
    'ObliqueShockMesh',
    'ObliqueShockMesher',
    'ObliqueShockRelation',
    'Reflection',
    'ReflectionSession',
    'SolutionPanel',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
