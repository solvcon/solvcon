# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The oblique-shock reflection solved with the multi-dimensional Euler solver.

The computing domain is a plain rectangle.  A uniform supersonic stream
enters horizontally from the left, the top boundary holds the state behind
an incident oblique shock, the bottom slip wall reflects that shock, and the
flow leaves through the non-reflective outflow on the right.

The driver, the setup, the session, the analysis, the view model, and the
presenter import without Qt, so a script runs, judges, and reads out the
reflection without the GUI.
"""

from ... import pilot

__all__, __getattr__, __dir__ = pilot.lazy_exports(__name__, {
    '_analytic': ['Reflection'],
    '_app': ['ObsreflApp'],
    '_column': ['ControlColumn'],
    '_control': ['ReflectionController'],
    '_driver': ['ObliqueShock', 'ObliqueShockRelation'],
    '_figure': ['ReflectionFigure'],
    '_presenter': ['ReflectionPresenter'],
    '_session': ['ReflectionSession'],
    '_setup': ['ReflectionSetup', 'max_deflection'],
    '_viewmodel': ['Display', 'FIELD_LABELS'],
    '_window': ['ReflectionWindow'],
}, ungated=('_analytic', '_driver', '_presenter', '_session', '_setup',
            '_viewmodel'))

if pilot.enable:
    from . import _app
    pilot.register_app("obsrefl", _app.build_app)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
