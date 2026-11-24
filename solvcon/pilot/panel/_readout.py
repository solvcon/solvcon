# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""What the run strip and the results card of an app window read of a run,
and the words they print.

This module imports no Qt, so a presenter fills its records headless too.
"""

import math
import typing

__all__ = [
    'Cell',
    'RUN_STATES',
    'ResultRow',
    'StripState',
    'clock',
    'format_error',
    'format_number',
    'format_sci',
    'format_signed',
    'tone_of',
]

#: The states a run can be in, in the order a run passes them.
RUN_STATES = ('ready', 'running', 'paused', 'finished', 'failed')


class Cell(typing.NamedTuple):
    """One value of the results card."""

    #: What the cell reads.
    text: str
    #: ``'good'``, ``'bad'``, ``'muted'``, or empty for the plain ink.
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


class StripState(typing.NamedTuple):
    """What the run strip shows of a run."""

    #: One of :data:`RUN_STATES`.
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


def format_sci(value, digits=3):
    """Return ``value`` in short scientific notation, ``2.71e-3`` rather
    than ``2.71e-03``, or "undefined" where it is not finite.

    :param value: The number.
    :param digits: The significant digits.
    """
    if not math.isfinite(value):
        return "undefined"
    if 0 == value:
        return "0"
    mantissa, exponent = f"{value:.{digits - 1}e}".split("e")
    return f"{mantissa}e{int(exponent)}"


def format_number(value, digits=4):
    """Return ``value`` to ``digits`` significant digits, or "undefined"
    like :func:`format_sci`.

    :param value: The number.
    :param digits: The significant digits.
    """
    if not math.isfinite(value):
        return format_sci(value)
    # Scientific where a fixed notation would run long.
    magnitude = abs(value)
    if magnitude and (magnitude >= 1e5 or magnitude < 1e-3):
        return format_sci(value, digits)
    return f"{value:.{digits}g}"


def format_signed(value, digits=2):
    """Return ``value`` like :func:`format_sci` with its sign spelled out,
    ``+1.2e-9``.

    :param value: The number.
    :param digits: The significant digits.
    """
    if not math.isfinite(value) or 0 == value:
        return format_sci(value, digits)
    return ("+" if value > 0 else "") + format_sci(value, digits)


def format_error(value):
    """Return the relative error ``value`` in percent with its sign,
    ``0.0123`` as ``+1.23%``, or "undefined" where it is not finite.

    :param value: The relative error.
    """
    if not math.isfinite(value):
        return "undefined"
    percent = 100.0 * value
    if 0 == percent:
        return "0%"
    # Scientific where two decimals would read +0.00% or run long.
    if abs(percent) < 0.01 or abs(percent) >= 1e4:
        return format_signed(percent) + "%"
    return f"{percent:+.2f}%"


def tone_of(error, good=0.01, poor=0.05):
    """Return the tone of a relative error: ``'good'`` within ``good``,
    ``'bad'`` beyond ``poor``, empty between, and ``'muted'`` where it is
    not finite.

    :param error: The relative error.
    :param good: The largest magnitude that reads good.
    :param poor: The largest magnitude that does not read bad.
    """
    if not math.isfinite(error):
        return 'muted'
    magnitude = abs(error)
    if magnitude <= good:
        return 'good'
    return 'bad' if magnitude > poor else ''


def clock(time, t_end):
    """Return the time readout ``t = time / t_end``, which keeps one width
    while the time runs.

    :param time: The time marched.
    :param t_end: The time at the step cap.
    """
    if not (math.isfinite(t_end) and t_end > 0):
        return f"t = {time:.4g} / {t_end:.4g}"
    # The decimals of four significant digits of the end time, with the
    # time padded to the width of the end time.
    decimals = max(0, 3 - math.floor(math.log10(t_end)))
    width = len(f"{t_end:.{decimals}f}")
    return f"t = {time:{width}.{decimals}f} / {t_end:.4g}"

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
