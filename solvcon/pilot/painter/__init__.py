# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The Painter: the draw tool selector and the inspector that edits what a 2D
canvas holds, with the pages, rows, icons, and palette-derived styling they
are built from.
"""

from .. import _pilot_core as _pcore

__all__, __getattr__, __dir__ = _pcore.lazy_exports(__name__, {
    '_gui': ['Painter', 'PainterPanel'],
})

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
