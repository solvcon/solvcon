# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The point probe of the reflection: one field read at a point inside a
cell in three ways, and against the analytic solution."""

import typing

import numpy as np

__all__ = [
    'Probe',
    'ProbeReading',
    'mean_value_weights',
]


def mean_value_weights(corners, point):
    """Return the mean value coordinates of ``point`` over a polygon.

    The weights sum to one, reproduce a linear field on any convex polygon,
    and reduce to the barycentric coordinates on a triangle, so one rule
    serves every cell type.  A point on a corner takes that corner alone,
    and a point on a side is cut linearly between its two ends, where the
    general formula divides by zero.

    :param corners: The polygon corners in order, ``[k, 2]``.
    :param point: The point, ``(x, y)``.
    :return: The weights, ``[k]``.
    """
    gaps = np.asarray(corners, dtype='float64') - np.asarray(
        point, dtype='float64')
    count = len(gaps)
    radius = np.hypot(gaps[:, 0], gaps[:, 1])
    weights = np.zeros(count, dtype='float64')
    scale = radius.max()
    on = np.nonzero(radius <= 1.e-12 * scale)[0]
    if len(on):
        weights[on[0]] = 1.0
        return weights
    after = np.roll(gaps, -1, axis=0)
    radius_after = np.roll(radius, -1)
    cross = gaps[:, 0] * after[:, 1] - gaps[:, 1] * after[:, 0]
    dot = (gaps * after).sum(axis=1)
    flat = np.abs(cross) <= 1.e-12 * scale * scale
    side = np.nonzero(flat & (dot < 0.0))[0]
    if len(side):
        it = side[0]
        frac = radius[it] / (radius[it] + radius_after[it])
        weights[it] = 1.0 - frac
        weights[(it + 1) % count] = frac
        return weights
    # tan(a/2) of the angle the side from corner i to corner i+1 spans at
    # the point, through (1 - cos a) / sin a.  A point on the line of a
    # side but past its ends sees it under no angle, where the formula
    # reads 0 / 0.
    half = np.zeros(count, dtype='float64')
    half[~flat] = (radius * radius_after - dot)[~flat] / cross[~flat]
    weights = (half + np.roll(half, 1)) / radius
    return weights / weights.sum()


class ProbeReading(typing.NamedTuple):
    """One field read at one point of the domain.

    :ivar name: The field read.
    :ivar cell: The body cell holding the point.
    :ivar point: The point read at, ``(x, y)``.
    :ivar solution_point: The solution point of the cell, ``(x, y)``.
    :ivar at_solution_point: The value of the cell, which holds at its
        solution point and is what the cell is colored by.
    :ivar by_element: The value the solution element of the cell gives at
        the point.
    :ivar by_mesh: The value interpolated at the point from the node values
        of the cell, which the profile cut reads too.
    :ivar exact: The analytic value at the point, or None for a field
        without one.
    """

    name: str
    cell: int
    point: tuple
    solution_point: tuple
    at_solution_point: float
    by_element: float
    by_mesh: float
    exact: typing.Optional[float]


class Probe(object):
    """Read a field of a reflection run at a point inside a cell.

    :ivar analysis: The :class:`~._analytic.Reflection` of the run.
    """

    def __init__(self, analysis):
        self.analysis = analysis

    def read(self, cell, point, name='density'):
        """Return the :class:`ProbeReading` of ``name`` at ``point`` in
        ``cell``.

        :param cell: The body cell index, as a pick reports it.
        :param point: The point, ``(x, y)``, inside the cell.
        :param name: One of :attr:`~solvcon.multidim.euler.EulerField.FIELDS`.
        :return: The :class:`ProbeReading` of the field at the point.
        """
        field = self.analysis.field
        x, y = float(point[0]), float(point[1])
        nodes, corners = self.analysis.cell_corners(cell)
        weights = mean_value_weights(corners, (x, y))
        by_mesh = float(weights @ self.analysis.node_field(name)[nodes])
        try:
            exact = float(self.analysis.analytic_profile([x], y, name)[0])
        except ValueError:
            exact = None
        return ProbeReading(
            name=name, cell=int(cell), point=(x, y),
            solution_point=tuple(float(v)
                                 for v in field.solution_point[cell]),
            at_solution_point=float(field.field_at(name, [cell])[0]),
            by_element=float(field.field_at(name, [cell], [(x, y)])[0]),
            by_mesh=by_mesh, exact=exact)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
