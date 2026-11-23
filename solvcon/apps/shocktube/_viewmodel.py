# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""What the widgets of the shock-tube app draw.

The presenter (:mod:`._presenter`) fills the records, and the widgets only
read them.
"""

import typing

__all__ = [
    'Display',
    'ProfileFrame',
    'QUANTITY_LABELS',
    'Results',
    'XtFrame',
]

# The problem is dimensionless, so a label carries no unit.
#: Display names of :data:`~._analytic.NAMES`, in the order of the panels.
QUANTITY_LABELS = {
    'density': "density",
    'velocity': "velocity",
    'pressure': "pressure",
    'temperature': "temperature",
    'internal_energy': "internal energy",
    'entropy': "entropy",
}


class Display(typing.NamedTuple):
    """What the figure shows of a run; none of it marches the run."""

    #: The quantities drawn, a panel each, in the order of
    #: :data:`QUANTITY_LABELS`.
    quantities: tuple = ('density', 'velocity', 'pressure')
    #: How the exact solution is drawn: ``'points'`` through the points
    #: of a fine grid, or ``'cells'`` averaged over the cells of the run.
    exact: str = 'points'
    #: Whether the wave fronts are guided.
    fronts: bool = True
    #: Whether a panel whose values are all positive spans decades.
    log: bool = False
    #: Whether a strip under each panel draws the run minus the exact
    #: cell means.
    residual: bool = False
    #: Whether the x-t diagram shows under the panels.
    xt: bool = True


class Results(typing.NamedTuple):
    """What the results card reads."""

    #: A :class:`~solvcon.pilot.panel.ResultRow` per quantity of the star
    #: state; empty with vacuum.
    states: tuple
    #: A :class:`~solvcon.pilot.panel.ResultRow` per shock and contact.
    fronts: tuple
    #: ``(label, text)`` of the run figures, such as the mass drift.
    rows: tuple
    #: The tooltip of the head: how the errors are taken.
    caption: str
    #: ``(text, tone)`` under the tables, or None.
    note: tuple = None


class ProfileFrame(typing.NamedTuple):
    """The profile panels of one frame."""

    #: ``(xmin, xmax)``, the ends of the tube.
    xlim: tuple
    #: The quantities drawn, a panel each.
    quantities: tuple
    #: ``(x, values)`` of the exact solution, ``values`` by quantity.
    exact: tuple
    #: ``(x, values)`` of the run at its points.
    points: tuple
    #: The exact cell means at the points by quantity, which the residual
    #: is taken against.
    reference: dict
    #: ``(label, x)`` of each wave front, left first.
    fronts: tuple
    #: Where the diaphragm stood at t = 0.
    x0: float
    #: ``(kind, label)`` of each curve the legend names, ``kind``
    #: ``'line'`` or ``'marker'``.
    legend: tuple
    #: The caption over the panels.
    caption: str


class XtFrame(typing.NamedTuple):
    """The x-t diagram of one frame: the history of a quantity of the run
    under the wave fronts of the exact solution."""

    #: ``(xmin, xmax)``, the ends of the tube.
    xlim: tuple
    #: The points of the run.
    x: object
    #: The time of each row of :attr:`rows`, ascending.
    times: object
    #: A row of values over :attr:`x` per time.
    rows: object
    #: ``(kind, speeds)`` of each wave of the exact solution.
    waves: tuple
    #: Where the waves start.
    x0: float
    #: The time at the step cap, the top of the diagram.
    t_end: float
    #: What the colors are of.
    label: str

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
