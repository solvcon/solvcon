# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""What the widgets of the reflection app draw, and the words they show.

The presenter (:mod:`._presenter`) fills the records, and the widgets only
read them.
"""

import math
import typing

import numpy as np

__all__ = [
    'Cell',
    'Display',
    'FIELD_LABELS',
    'FieldFrame',
    'ProfileFrame',
    'ResultRow',
    'Results',
    'STATES',
    'StripState',
    'clock',
    'format_error',
    'format_number',
    'format_sci',
    'format_signed',
    'tone_of',
]

#: The states a run can be in, in the order a run passes them.
STATES = ('ready', 'running', 'paused', 'finished', 'failed')
#: Display names of :attr:`~solvcon.multidim.euler.EulerField.FIELDS`; the
#: problem is dimensionless, so there is no unit.
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


class Cell(typing.NamedTuple):
    """One value of the results card."""

    #: What the cell reads.
    text: str
    #: 'good', 'bad', 'muted', or empty for the plain ink.
    tone: str = ''
    #: The tooltip, or empty for none.
    tip: str = ''


class ResultRow(typing.NamedTuple):
    """One row of a table of the results card: a computed value against
    the exact one, and the error between them."""

    #: What the row is of, such as ``'zone 1'``.
    label: str
    #: The :class:`Cell` of the computed value.
    computed: Cell
    #: The :class:`Cell` of the exact value.
    exact: Cell
    #: The :class:`Cell` of the error between them.
    error: Cell


class Results(typing.NamedTuple):
    """What the results card reads."""

    #: The display label of the field the zones compare.
    field: str
    #: A :class:`ResultRow` per zone, ``'zone 1'`` first; empty for a field
    #: the analytic solution gives no value of.
    zones: tuple
    #: The :class:`ResultRow` of the fitted ``'shock angle'``, in degrees,
    #: and of the ``'wall x'`` of the reflection point.
    fits: tuple
    #: ``(label, text)`` of the run figures, such as the mass drift and the
    #: CFL range.
    rows: tuple
    #: The tooltip of the head: how the errors are taken.
    caption: str
    #: ``(text, tone)`` under the tables, or None.
    note: tuple = None


class StripState(typing.NamedTuple):
    """What the run strip shows of a run."""

    #: One of :data:`STATES`.
    state: str
    #: The steps marched.
    step: int
    #: The step cap of the run.
    max_steps: int
    #: The time marched, ``step`` times the time increment.
    time: float
    #: The time at the step cap.
    t_end: float
    #: What made the run fail, or empty.
    failure: str = ''
    #: Whether the parameters hold edits the run has not taken.
    dirty: bool = False
    #: Whether the edits are valid, so a reset can rebuild the run from
    #: them.
    can_reset: bool = True


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


def format_sci(value, digits=3):
    """Return ``value`` in short scientific notation, ``2.71e-3`` rather
    than ``2.71e-03``; a value that is not finite reads "undefined"."""
    if not math.isfinite(value):
        return "undefined"
    if 0 == value:
        return "0"
    mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def format_number(value, digits=4):
    """Return ``value`` to ``digits`` significant digits, in scientific
    notation where a fixed one would run long, and "undefined" like
    :func:`format_sci`."""
    if not math.isfinite(value):
        return format_sci(value)
    magnitude = abs(value)
    if magnitude and (magnitude >= 1e5 or magnitude < 1e-3):
        return format_sci(value, digits)
    return f"{value:.{digits}g}"


def format_signed(value, digits=2):
    """Return ``value`` like :func:`format_sci` with its sign spelled out,
    so a drift reads which way it went."""
    if not math.isfinite(value) or 0 == value:
        return format_sci(value, digits)
    return ("+" if value > 0 else "") + format_sci(value, digits)


def format_error(value):
    """Return the relative error ``value`` in percent with its sign,
    ``0.0123`` as ``+1.23%``.

    An error too small or too large for two decimals of a percent goes
    scientific rather than reading ``+0.00%`` or running long; a value that
    is not finite reads "undefined".
    """
    if not math.isfinite(value):
        return "undefined"
    percent = 100.0 * value
    if 0 == percent:
        return "0%"
    if abs(percent) < 0.01 or abs(percent) >= 1e4:
        return format_signed(percent) + "%"
    return f"{percent:+.2f}%"


def tone_of(error, good=0.01, poor=0.05):
    """Return the tone of a relative error: 'good' within ``good``, 'bad'
    beyond ``poor``, plain between, and 'muted' where it is undefined."""
    if not math.isfinite(error):
        return 'muted'
    magnitude = abs(error)
    if magnitude <= good:
        return 'good'
    return 'bad' if magnitude > poor else ''


def clock(time, t_end):
    """Return the time readout ``t = time / t_end``, the time to the
    decimals four digits of the end time take and padded to the width of
    the end time, so its width holds while the time runs."""
    if not (math.isfinite(t_end) and t_end > 0):
        return f"t = {time:.4g} / {t_end:.4g}"
    decimals = max(0, 3 - math.floor(math.log10(t_end)))
    width = len(f"{t_end:.{decimals}f}")
    return f"t = {time:{width}.{decimals}f} / {t_end:.4g}"

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
