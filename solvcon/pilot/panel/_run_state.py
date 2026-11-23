# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""What the run strip of an app window reads of a run.

This module imports no Qt, so a presenter fills its records headless too.
"""

import math
import typing

__all__ = [
    'RUN_STATES',
    'StripState',
    'clock',
]

#: The states a run can be in, in the order a run passes them.
RUN_STATES = ('ready', 'running', 'paused', 'finished', 'failed')


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
