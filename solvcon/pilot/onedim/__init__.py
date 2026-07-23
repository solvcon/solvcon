# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
One-dimensional CESE solver demos: the Euler shock tube, Burgers' equation,
and the linear scalar wave.
"""

from .. import _pilot_core as _pcore

if _pcore.enable:
    from . import _burgers1d
    from . import _euler1d
    from . import _linear_wave

    Burgers1DApp = _burgers1d.Burgers1DApp
    Euler1DApp = _euler1d.Euler1DApp
    LinearWave1DApp = _linear_wave.LinearWave1DApp
else:
    # Bind only the public names: a None module attribute would shadow the
    # real submodule import in no-GUI builds.
    Burgers1DApp = None
    Euler1DApp = None
    LinearWave1DApp = None

__all__ = [
    'Burgers1DApp',
    'Euler1DApp',
    'LinearWave1DApp',
]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
