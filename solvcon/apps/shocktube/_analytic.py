# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Judge a shock-tube run against the exact solution of its Riemann
problem.

The run holds each quantity at its solution points.  :class:`TubeAnalysis`
holds the exact solution on the same points and the cells around them, and
measures the run: the star state the gas between the outer waves reaches,
where the shocks and the contact stand, and how far the run strays from
the exact cell means.
"""

import math
import typing

import numpy as np

from . import _riemann

__all__ = [
    'FrontFit',
    'NAMES',
    'TubeAnalysis',
    'ZoneValue',
    'quantities',
]

#: The quantities a run holds and the exact solution gives, named as the
#: fields of the solver.
NAMES = ('density', 'velocity', 'pressure', 'temperature',
         'internal_energy', 'entropy')
#: The gas constant the solver takes the temperature with.
GAS_CONSTANT = 8.31446261815324


def quantities(sol, gamma):
    """Return every quantity of :data:`NAMES` of an exact solution.

    :param sol: ``{'rho', 'u', 'p'}`` arrays, as
        :meth:`~._riemann.ExactRiemann.sample` gives them.
    :param gamma: The ratio of specific heats.
    """
    rho, vel, pre = sol['rho'], sol['u'], sol['p']
    with np.errstate(divide='ignore', invalid='ignore'):
        temperature = pre / (rho * GAS_CONSTANT)
    return {'density': rho, 'velocity': vel, 'pressure': pre,
            'temperature': temperature,
            'internal_energy': _riemann.derive('e', rho, vel, pre, gamma),
            'entropy': _riemann.derive('s', rho, vel, pre, gamma)}


class ZoneValue(typing.NamedTuple):
    """A quantity of the star state, the run's mean against the exact
    value."""

    #: What the value is, such as ``'p*'``.
    label: str
    #: The mean of the run over the cells clear of every front, or NaN.
    computed: float
    #: The exact value.
    exact: float
    #: ``computed / exact - 1``, or NaN without cells or an exact 0.
    error: float
    #: The cells the mean is over.
    count: int


class FrontFit(typing.NamedTuple):
    """Where the run puts a front against where the exact solution does."""

    #: What the front is, such as ``'shock x'``.
    label: str
    #: Where the run crosses the value midway across the front, or NaN.
    computed: float
    #: Where the exact front stands.
    exact: float
    #: The miss over the distance the front has run from the diaphragm,
    #: the relative error of its speed, or NaN.
    error: float
    #: Whether the front has run far enough from the diaphragm to be told
    #: apart from it.
    formed: bool


class TubeAnalysis(object):
    """Measure a run on the solution points ``x`` against ``exact``.

    Each point stands for a cell around it.  A mean of the star state
    leaves out the cells within :attr:`MARGIN` cells of a front, where the
    scheme smears the jump.

    :param exact: The :class:`~._riemann.ExactRiemann` of the run.
    :param x: The solution points, ascending.
    :param xlim: ``(xmin, xmax)``, the ends of the tube.
    """

    #: The cells either side of a front that a zone mean leaves out.
    MARGIN = 2

    def __init__(self, exact, x, xlim):
        self.exact = exact
        self.x = np.asarray(x, dtype='float64')
        self.xlim = (float(xlim[0]), float(xlim[1]))
        # A cell spans the faces midway to the neighbours; the end cells
        # reach as far out as the inner ones, but not past the tube.
        middle = 0.5 * (self.x[:-1] + self.x[1:])
        faces = np.concatenate([2.0 * self.x[:1] - middle[:1], middle,
                                2.0 * self.x[-1:] - middle[-1:]])
        #: The faces of the cells around :attr:`x`.
        self.faces = np.clip(faces, *self.xlim)
        #: The distance between two solution points.
        self.spacing = float(self.x[1] - self.x[0]) if len(self.x) > 1 \
            else 0.0

    def sample(self, coord, time):
        """Return the exact solution at ``coord`` at ``time`` by the name
        of each quantity of :data:`NAMES`.

        :param coord: The points to sample at.
        :param time: The time to sample at.
        """
        return quantities(self.exact.sample(coord, time), self.exact.gamma)

    def means(self, time):
        """Return the exact solution averaged over each cell at ``time``,
        one value per solution point, by quantity.

        :param time: The time to average at.
        """
        return quantities(self.exact.averages(self.faces, time),
                          self.exact.gamma)

    def steps(self, time):
        """Return ``(x, values)`` of the exact cell means at ``time`` drawn
        as steps that span each cell."""
        means = self.means(time)
        return (np.repeat(self.faces, 2)[1:-1],
                {name: np.repeat(value, 2) for name, value in means.items()})

    def mass(self, density):
        """Return the mass the cells hold of ``density`` at the points."""
        return float(np.dot(np.asarray(density, dtype='float64'),
                            np.diff(self.faces)))

    def l1_error(self, values, means):
        """Return how far ``values`` stray from ``means``: the sum of the
        misses over the sum of the means, both weighted by the cell widths,
        or NaN where the means sum to 0."""
        widths = np.diff(self.faces)
        miss = float(np.dot(np.abs(values - means), widths))
        total = float(np.dot(np.abs(means), widths))
        return miss / total if total else math.nan

    def _edges(self, time):
        """Return where the left wave ends, the contact stands, and the
        right wave starts at ``time``, or None with vacuum."""
        exact = self.exact
        if exact.vacuum:
            return None
        left, contact, right = exact.waves
        return (exact.x0 + time * max(left.speeds),
                exact.x0 + time * contact.speeds[0],
                exact.x0 + time * min(right.speeds))

    def star_zones(self, values, time):
        """Return a :class:`ZoneValue` of each quantity of the star state:
        the pressure and the velocity between the outer waves, and the
        density either side of the contact.  Vacuum has no star state, and
        gives none.

        :param values: The run's quantities at the points, by name.
        :param time: The time the run stands at.
        """
        edges = self._edges(time)
        if edges is None:
            return ()
        left, contact, right = edges
        room = self.MARGIN * self.spacing
        x = self.x
        on_left = (x > left + room) & (x < contact - room)
        on_right = (x > contact + room) & (x < right - room)
        exact = self.exact
        rows = (('p*', 'pressure', on_left | on_right, exact.p_star),
                ('u*', 'velocity', on_left | on_right, exact.u_star),
                ('rho* left', 'density', on_left, exact.rho_star_left),
                ('rho* right', 'density', on_right, exact.rho_star_right))
        out = []
        for label, name, mask, target in rows:
            count = int(mask.sum())
            computed = (float(np.mean(values[name][mask])) if count
                        else math.nan)
            error = (computed / target - 1.0 if count and target
                     else math.nan)
            out.append(ZoneValue(label, computed, float(target), error,
                                 count))
        return tuple(out)

    def front_fits(self, values, time):
        """Return a :class:`FrontFit` of each shock and of the contact,
        left to right.

        A shock is placed where the pressure crosses the mean of the
        pressures either side of it, and the contact where the density
        does, each searched between the fronts beside it.  A front closer
        to the diaphragm than :attr:`MARGIN` cells is not formed.

        :param values: The run's quantities at the points, by name.
        :param time: The time the run stands at.
        """
        edges = self._edges(time)
        if edges is None:
            return ()
        exact = self.exact
        left, contact, right = edges
        xmin, xmax = self.xlim
        found = []
        if 'shock' == exact.waves[0].kind:
            found.append(("left shock x" if 'shock' == exact.waves[2].kind
                          else "shock x", 'pressure', left,
                          (exact.left.p, exact.p_star), (xmin, contact)))
        if exact.strengths[1]:
            found.append(("contact x", 'density', contact,
                          (exact.rho_star_left, exact.rho_star_right),
                          (left, right)))
        if 'shock' == exact.waves[2].kind:
            found.append(("right shock x" if 'shock' == exact.waves[0].kind
                          else "shock x", 'pressure', right,
                          (exact.p_star, exact.right.p), (contact, xmax)))
        out = []
        for label, name, target, sides, span in found:
            run = target - exact.x0
            formed = abs(run) >= self.MARGIN * self.spacing
            computed = (self._crossing(values[name], 0.5 * sum(sides),
                                       target, span) if formed
                        else math.nan)
            error = ((computed - target) / run if math.isfinite(computed)
                     else math.nan)
            out.append(FrontFit(label, computed, float(target), error,
                                formed))
        return tuple(out)

    def _crossing(self, values, level, near, span):
        """Return where ``values`` cross ``level`` inside ``span``, the
        crossing nearest ``near`` by linear interpolation, or NaN."""
        x = self.x
        inside = np.nonzero((x > span[0]) & (x < span[1]))[0]
        if len(inside) < 2:
            return math.nan
        lo, hi = inside[0], inside[-1] + 1
        diff = np.asarray(values[lo:hi], dtype='float64') - level
        flips = np.nonzero(np.signbit(diff[:-1]) != np.signbit(diff[1:]))[0]
        if 0 == len(flips):
            return math.nan
        xs = x[lo:hi]
        spots = xs[flips] + diff[flips] / (diff[flips] - diff[flips + 1]) \
            * (xs[flips + 1] - xs[flips])
        return float(spots[np.argmin(np.abs(spots - near))])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
