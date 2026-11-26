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

    FIELDS = ('density', 'velx', 'vely', 'speed', 'pressure', 'mach',
              'total_energy', 'cfl')

    #: Momentum components in axis order; the solution table carries the
    #: first ``ndim`` of them.
    MOMENTA = ('momx', 'momy', 'momz')

    def __init__(self, svr, mesh):
        self.svr = svr
        self.mesh = mesh

    @property
    def conserveds(self):
        """The conserved variables in the order :attr:`conserved` holds
        them."""
        return (('density',) + self.MOMENTA[:self.svr.ndim]
                + ('total_energy',))

    @property
    def conserved(self):
        """The newest order-0 solution, ``[ncell, neq]``.

        :meth:`EulerCore.march` leaves the newest step in ``so0n``.  The
        view aliases the solver memory, so writing into it seeds the field.
        """
        arr = self.svr.so0n
        return arr.ndarray[arr.nghost:]

    @property
    def density(self):
        """The newest density over the body cells, ``[ncell]``."""
        return self.conserved[:, 0]

    @property
    def gamma(self):
        """The per-cell ratio of specific heats, ``[ncell]``."""
        arr = self.svr.gamma
        return arr.ndarray[arr.nghost:]

    @property
    def centroid(self):
        """The body-cell centroid, ``[ncell, ndim]``."""
        arr = self.mesh.clcnd
        return arr.ndarray[arr.nghost:]

    @property
    def volume(self):
        """The body-cell volume, ``[ncell]``."""
        arr = self.mesh.clvol
        return arr.ndarray[arr.nghost:]

    @property
    def vel(self):
        """The body-cell velocity, ``[ncell, ndim]``."""
        cons = self.conserved
        return cons[:, 1:1 + self.svr.ndim] / cons[:, 0][:, None]

    @property
    def velx(self):
        """The x component of velocity, ``[ncell]``."""
        return self.vel[:, 0]

    @property
    def vely(self):
        """The y component of velocity, ``[ncell]``."""
        return self.vel[:, 1]

    @property
    def speed(self):
        """The velocity magnitude, ``[ncell]``."""
        return np.sqrt((self.vel ** 2).sum(axis=1))

    @property
    def total_energy(self):
        """The total energy per unit volume, ``[ncell]``."""
        return self.conserved[:, 1 + self.svr.ndim]

    @property
    def pressure(self):
        """The pressure of the ideal gas, ``[ncell]``."""
        return (self.gamma - 1.0) * (self.total_energy
                                     - 0.5 * self.density * self.speed ** 2)

    @property
    def mach(self):
        """The Mach number against the local speed of sound, ``[ncell]``."""
        return self.speed / np.sqrt(self.gamma * self.pressure / self.density)

    @property
    def cfl(self):
        """The CFL number the last step ran at, ``[ncell]``.

        The solver's unclamped ``cflo``: the wave speed over half a step
        against the shortest distance out of the cell.  Above 1 a cell
        marched past its own domain of dependence.
        """
        arr = self.svr.cflo
        return arr.ndarray[arr.nghost:]

    @property
    def solution_point(self):
        """The solution point of each body cell, ``[ncell, ndim]``.

        It is the centroid of the conservation element of the cell, where
        :attr:`conserved` holds, and not the cell centroid: the two part on
        a triangle.
        """
        arr = self.svr.cecnd
        return arr.ndarray[arr.nghost:, :self.svr.ndim]

    @property
    def gradient(self):
        """The newest spatial gradient of :attr:`conserved`,
        ``[ncell, neq, ndim]``."""
        arr = self.svr.so1n
        return arr.ndarray[arr.nghost:]

    def expand(self, cells, points):
        """Return the conserved variables the solution elements of
        ``cells`` give at ``points``, ``[n, neq]``.

        The solution element of a cell is its order-0 value plus the
        gradient times the offset from the solution point.

        :param cells: The body-cell indices, ``[n]``.
        :param points: The points to read at, ``[n, ndim]``, one per cell.
        :return: The conserved variables, ``[n, neq]``.
        """
        cells = np.asarray(cells, dtype='int64')
        offset = (np.asarray(points, dtype='float64')
                  - self.solution_point[cells])
        return (self.conserved[cells]
                + np.einsum('ned,nd->ne', self.gradient[cells], offset))

    def field_at(self, name, cells, points=None):
        """Return the named field of ``cells``, read at ``points`` by their
        solution elements, ``[n]``.

        A derived field takes the expanded conserved variables, since the
        solution element expands those and not the field.  The CFL number
        has no solution element and reads the cell value.

        :param name: One of :attr:`FIELDS`.
        :param cells: The body-cell indices, ``[n]``.
        :param points: The points, ``[n, ndim]``, or None for the solution
            points, where the reading is the cell value.
        :return: The field value of each cell, ``[n]``.
        :raise ValueError: For a name that is not in :attr:`FIELDS`.
        """
        if name not in self.FIELDS:
            raise ValueError(f"unknown field '{name}'")
        cells = np.asarray(cells, dtype='int64')
        if points is None:
            conserved = self.conserved[cells]
        else:
            conserved = self.expand(cells, points)
        return _RowField(self, cells, conserved).field(name)

    def field(self, name):
        """Return the scalar field named by one of :attr:`FIELDS`.

        The name is the property that derives the field, so a caller
        holding a name (a control, a report, a plot axis) reaches the same
        reader a caller holding the attribute does.
        """
        if name not in self.FIELDS:
            raise ValueError(f"unknown field '{name}'")
        return getattr(self, name)

    def calc_overall_mass(self):
        """Return the density integrated over the whole domain."""
        return float(np.sum(self.density * self.volume))

    def calc_cfl_range(self):
        """Return the smallest and the largest CFL number in the domain.

        A mean would hide the one cell that outran the stable step.
        """
        cfl = self.cfl
        return float(cfl.min()), float(cfl.max())


class _RowField(EulerField):
    """The fields of chosen rows of conserved variables, derived the way
    :class:`EulerField` derives them over the body cells.

    Only :meth:`field` and the fields it names hold for the rows; the
    geometry and the sums, such as :attr:`centroid` or
    :meth:`calc_overall_mass`, still read the whole mesh.

    :param base: The :class:`EulerField` the rows come from.
    :param cells: The body cells the rows stand for, ``[n]``.
    :param conserved: The conserved variables, ``[n, neq]``.
    """

    def __init__(self, base, cells, conserved):
        super().__init__(base.svr, base.mesh)
        self._base = base
        self._cells = cells
        self._conserved = conserved

    @property
    def conserved(self):
        """The rows this reader was given, ``[n, neq]``."""
        return self._conserved

    @property
    def gamma(self):
        """The ratio of specific heats of the cells, ``[n]``."""
        return self._base.gamma[self._cells]

    @property
    def cfl(self):
        """The CFL number of the cells, ``[n]``."""
        return self._base.cfl[self._cells]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
