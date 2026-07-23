# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The 2D drawing surface: the canvas window, the draw-tool painter, and the
SVG import dialog.
"""

from .. import _pilot_core as _pcore

if _pcore.enable:
    from . import _canvas_gui
    from . import _painter_gui
    from . import _svg_gui

    Canvas = _canvas_gui.Canvas
    Save2DCanvasDialog = _canvas_gui.Save2DCanvasDialog
    Painter = _painter_gui.Painter
    SVGFileDialog = _svg_gui.SVGFileDialog
else:
    # Bind only the public names: a None module attribute would shadow the
    # real submodule import in no-GUI builds.
    Canvas = None
    Save2DCanvasDialog = None
    Painter = None
    SVGFileDialog = None

__all__ = [
    'Canvas',
    'Painter',
    'SVGFileDialog',
    'Save2DCanvasDialog',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
