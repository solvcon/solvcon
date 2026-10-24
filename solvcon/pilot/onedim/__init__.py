# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
One-dimensional CESE solver demos: the Euler shock tube, Burgers' equation,
and the linear scalar wave.
"""

from .. import _pilot_core as _pcore

__all__, __getattr__, __dir__ = _pcore.lazy_exports(__name__, {
    '_burgers1d': ['Burgers1DApp'],
    '_euler1d': ['Euler1DApp'],
    '_linear_wave': ['LinearWave1DApp'],
})

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
