# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Sod's shock tube problem.

A diaphragm at the middle of a straight tube separates two gases at rest of
different pressures and densities.  Bursting it sends a shock wave and a
contact surface into the low-pressure side and an expansion wave into the
high-pressure side, and the analytic solution judges the numerical one.

The setup, the session, and the analysis import without Qt, so a script
runs and judges a shock tube without the GUI.
"""

from ... import pilot

__all__, __getattr__, __dir__ = pilot.lazy_exports(__name__, {
    '_analytic': ['TubeAnalysis'],
    '_app': ['ShockTubeApp'],
    '_riemann': ['GasState', 'Wave', 'ExactRiemann', 'QUANTITIES', 'derive'],
    '_session': ['ShockTubeSession'],
    '_setup': ['ShockTubeSetup'],
    '_shocktube': ['ShockTube'],
}, ungated=('_analytic', '_riemann', '_session', '_setup', '_shocktube'))

if pilot.enable:
    from . import _app
    pilot.register_app("shocktube", _app.build_app)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
