# Copyright (c) 2022, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
One-dimensional Solver for the Euler Equations
"""

import numpy as np

try:
    from _solvcon import onedim as _impl  # noqa: F401
except ImportError:
    from ..._solvcon import onedim as _impl  # noqa: F401

__all__ = [
    'Euler1DCore',
    'Euler1DSolver',
]

Euler1DCore = _impl.Euler1DCore


class Euler1DSolver:
    """
    Numerical solver for the one-dimensional Euler equation by using the CESE
    method.
    """

    def __init__(self, xmin, xmax, ncoord, time_increment=0.05,
                 keep_edge=False):
        self._core = self.init_solver(xmin, xmax, ncoord, time_increment,
                                      gamma=1.4)
        # gamma is 1.4 for air.
        _ = self.ncoord - 1
        start = 0 if keep_edge else 2
        stop = _ if keep_edge else (_ - 2)
        num = (stop - start) // 2 + 1
        self.xindices = np.linspace(start, stop, num, dtype='int32')

    def __getattr__(self, name):
        return getattr(self._core, name)

    @staticmethod
    def init_solver(xmin, xmax, ncoord, time_increment, gamma):
        # Create the solver object.
        svr = Euler1DCore(ncoord=ncoord, time_increment=time_increment)

        # Initialize spatial grid.
        svr.coord[...] = np.linspace(xmin, xmax, num=ncoord)

        # Initialize field.
        svr.cfl.fill(0)
        svr.gamma.fill(gamma)
        svr.so0[...] = 0.0
        svr.so1[...] = 0.0

        return svr

    @staticmethod
    def calc_u2(gamma, density, velocity, pressure):
        ie = 1. / (gamma - 1) * pressure / density
        ke = velocity * velocity / 2
        return density * (ie + ke)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
