# Copyright (c) 2024, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Airfoil shape
"""

from . import _naca
from .. import _pilot_core as _pcore
if _pcore.enable:
    from . import _airfoil_gui
else:
    _airfoil_gui = None

Naca4 = _naca.Naca4
Naca4Sampler = _naca.Naca4Sampler
Naca4Airfoil = _airfoil_gui.Naca4Airfoil if _airfoil_gui else None

__all__ = [
    'Naca4',
    'Naca4Sampler',
    'Naca4Airfoil',
]


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
