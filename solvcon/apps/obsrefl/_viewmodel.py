# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""What the widgets of the reflection app draw, and the words they show.

The presenter (:mod:`._presenter`) fills the records, and the widgets only
read them.
"""

import typing

import numpy as np

__all__ = [
    'Display',
    'FIELD_LABELS',
    'FieldFrame',
    'ProbeFrame',
    'ProfileFrame',
    'Results',
]

# The problem is dimensionless, so a label carries no unit.
#: Display names of :attr:`~solvcon.multidim.euler.EulerField.FIELDS`.
FIELD_LABELS = {
    'density': 'density',
    'velx': 'x velocity',
    'vely': 'y velocity',
    'speed': 'speed',
    'pressure': 'pressure',
    'mach': 'Mach number',
    'total_energy': 'total energy',
    'cfl': 'CFL number',
}


class Results(typing.NamedTuple):
    """What the results card reads."""

    #: The display label of the field the zones compare.
    field: str
    #: A :class:`~solvcon.pilot.panel.ResultRow` per zone, ``'zone 1'``
    #: first; empty for a field the analytic solution gives no value of.
    zones: tuple
    #: The :class:`~solvcon.pilot.panel.ResultRow` of the fitted ``'shock
    #: angle'``, in degrees, and of the ``'wall x'`` of the reflection
    #: point.
    fits: tuple
    #: ``(label, text)`` of the run figures, such as the mass drift and the
    #: CFL range.
    rows: tuple
    #: The tooltip of the head: how the errors are taken.
    caption: str
    #: ``(text, tone)`` under the tables, or None.
    note: tuple = None


class Display(typing.NamedTuple):
    """What the figure shows of a run; none of it marches the run."""

    #: The name of the colored field, a key of :data:`FIELD_LABELS`.
    field: str = 'density'
    #: Whether the color bar shows.
    bar: bool = True
    #: Whether the cells the profile samples are outlined.
    mark_cells: bool = False
    #: Whether the line of the profile is drawn.
    mark_line: bool = False
    #: The height of the profile, a fraction of the domain height.
    cut: float = 0.5
    #: Whether hovering over the domain probes the field.
    probe: bool = True


class FieldFrame(typing.NamedTuple):
    """The colored field of one frame of the domain view and its color
    bar."""

    #: The field name, a key of :data:`FIELD_LABELS`.
    name: str
    #: Its display label.
    label: str
    #: The value per body cell.
    values: np.ndarray
    #: The low end of the color range, pinned to cover the exact zone
    #: values so a color reads alike over the run.
    lo: float
    #: The high end of the color range.
    hi: float
    #: The caption over the view.
    caption: str
    #: The cells to outline, an array of cell indices, or None.
    gauge_cells: object = None
    #: The height of the profile line, or None.
    gauge_height: float = None


class ProbeFrame(typing.NamedTuple):
    """The readout of the probe under the cursor."""

    #: The line under the caption: the cell, the point, and the readings.
    text: str
    #: The tooltip: what each reading is.
    tip: str


class ProfileFrame(typing.NamedTuple):
    """The profile plot of one frame: the field along a horizontal cut."""

    #: The abscissae, ascending, shared by both series.
    x: np.ndarray
    #: The computed field at ``x``.
    computed: np.ndarray
    #: The analytic field at ``x``.
    exact: np.ndarray
    #: The title of the value axis.
    label: str
    #: What the frame is a frame of, compared with ``==``: a frame of
    #: another key resets the zoom, and no other change does.
    key: object = None


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
