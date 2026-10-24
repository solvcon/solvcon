# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The 2D drawing surface: the canvas window and the SVG import dialog. The
Painter that draws on it lives in :mod:`solvcon.pilot.painter`.
"""

from .. import _pilot_core as _pcore

__all__, __getattr__, __dir__ = _pcore.lazy_exports(__name__, {
    '_canvas_gui': ['Canvas', 'Save2DCanvasDialog'],
    '_naca_airfoil': ['Naca4', 'Naca4Sampler'],
    '_svg_gui': ['SVGFileDialog'],
}, ungated=('_naca_airfoil',))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
