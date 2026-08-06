# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Body-cell views of the multi-dimensional Euler solution.
"""

import numpy as np

__all__ = [
    'EulerField',
]


class EulerField(object):
    """Read the body-cell fields of one Euler run.

    A solver table and a mesh table both prepend the ghost rows and count
    them in :attr:`SimpleArray.nghost`, so a body-cell view is the table
    from that row on.  Pairing the two tables here lets a caller ask for a
    quantity instead of assembling it from a solver, a mesh, and an offset.

    :ivar svr: The :class:`~solvcon.core.EulerCore` holding the solution.
    :ivar mesh: The :class:`~solvcon.core.StaticMesh` the solver runs on.
    """

    def __init__(self, svr, mesh):
        self.svr = svr
        self.mesh = mesh

    def conserved(self):
        """Return the newest order-0 solution, ``[ncell, neq]``.

        :meth:`EulerCore.march` leaves the newest step in ``so0n``.  The
        view aliases the solver memory, so writing into it seeds the field.
        """
        arr = self.svr.so0n
        return arr.ndarray[arr.nghost:]

    def density(self):
        """Return the newest density over the body cells, ``[ncell]``."""
        return self.conserved()[:, 0]

    def centroid(self):
        """Return the body-cell centroid, ``[ncell, ndim]``."""
        arr = self.mesh.clcnd
        return arr.ndarray[arr.nghost:]

    def volume(self):
        """Return the body-cell volume, ``[ncell]``."""
        arr = self.mesh.clvol
        return arr.ndarray[arr.nghost:]

    def total_mass(self):
        """Return the density integrated over the whole domain."""
        return float(np.sum(self.density() * self.volume()))

    def residual(self):
        r"""Return the root-mean-square density change per unit time.

        .. math::

            r = \frac{2}{\Delta t}
                \sqrt{\frac{1}{N} \sum_{i=1}^{N}
                      \left(\rho^{n}_{i} - \rho^{c}_{i}\right)^{2}}

        A CESE substep advances half a :attr:`EulerCore.time_increment`
        :math:`\Delta t` and leaves the state it started from in ``so0c``,
        so the density difference against ``so0n`` spans that half step.
        Averaging over the :math:`N` body cells keeps meshes of different
        size comparable, and dividing by the half step makes a rate that a
        run marching toward a steady state drives to zero.
        """
        old = self.svr.so0c
        change = self.density() - old.ndarray[old.nghost:, 0]
        return float(np.sqrt(np.mean(change * change))
                     / (self.svr.time_increment / 2.0))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
