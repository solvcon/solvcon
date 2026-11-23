# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Read a shock-tube run out into what the widgets of the app show.

The functions read a :class:`~._session.ShockTubeSession`, its setup, or
its display into a record of :mod:`._viewmodel` or a text, and
:class:`ShockTubePresenter` hands the records to the window and the
control column.
"""

import math

import numpy as np

from ...pilot import panel

from . import _setup
from . import _viewmodel

__all__ = [
    'MARCH_LABEL',
    'ShockTubePresenter',
    'display_summary',
    'numerics_summary',
    'problem_summary',
    'profile_frame',
    'results',
    'strip_state',
    'xt_frame',
]

#: What the legend and the x-t diagram call the run.
MARCH_LABEL = "CESE"
#: The points the exact solution is drawn through.
EXACT_POINTS = 801


def _star_rows(session):
    """Return a :class:`~solvcon.pilot.panel.ResultRow` per quantity of the
    star state."""
    margin = session.analysis.MARGIN
    rows = []
    for zone in session.star_zones():
        exact = panel.Cell(panel.format_number(zone.exact))
        if 0 == zone.count:
            computed = panel.Cell(
                "no cells", 'muted',
                f"no cell of the star region stands {margin} cells clear "
                f"of the fronts yet")
            error = panel.Cell("-", 'muted')
        else:
            computed = panel.Cell(panel.format_number(zone.computed), '',
                                  f"mean over {zone.count} cells")
            tip = ("the exact value is 0, so no error is relative to it"
                   if 0 == zone.exact else "")
            error = panel.Cell(panel.format_error(zone.error),
                               panel.tone_of(zone.error), tip)
        rows.append(panel.ResultRow(zone.label, computed, exact, error))
    return tuple(rows)


def _front_rows(session):
    """Return a :class:`~solvcon.pilot.panel.ResultRow` per shock and
    contact."""
    margin = session.analysis.MARGIN
    rows = []
    for fit in session.front_fits():
        exact = panel.Cell(panel.format_number(fit.exact), '',
                           "where the exact front stands")
        if not fit.formed:
            computed = panel.Cell(
                "not formed", 'muted',
                f"the front has run less than {margin} cells from the "
                f"diaphragm")
            error = panel.Cell("-", 'muted')
        elif not math.isfinite(fit.computed):
            computed = panel.Cell("not found", 'muted',
                                  "the run crosses no value midway across "
                                  "the front near it")
            error = panel.Cell("-", 'muted')
        else:
            computed = panel.Cell(panel.format_number(fit.computed), '',
                                  "where the run crosses the value midway "
                                  "across the front")
            error = panel.Cell(panel.format_error(fit.error),
                               panel.tone_of(fit.error),
                               "the miss over the distance the front has "
                               "run")
        rows.append(panel.ResultRow(fit.label, computed, exact, error))
    return tuple(rows)


def _run_rows(session):
    """Return the ``(label, text)`` rows of the run figures: the mass drift
    since step 0, signed, the L1 error of the density against the exact
    cell means, and the largest CFL number of the last chunk."""
    last = session.records[-1]
    l1 = session.l1_error('density')
    return (('mass drift', panel.format_signed(session.mass_drift())),
            ('L1 density', panel.format_error(l1).lstrip("+")),
            ('CFL', f"{last.cfl:.2f}" if session.step else "-"))


def _escaped(session):
    """Return whether a wave of the exact solution has run out of the
    tube."""
    # Past that, the solver's ends no longer hold the gas at rest.
    xmin, xmax = session.xlim
    exact, time = session.exact, session.time
    speeds = [speed for wave in exact.waves for speed in wave.speeds]
    return any(not xmin <= exact.x0 + speed * time <= xmax
               for speed in speeds)


def results(session):
    """Return the :class:`~._viewmodel.Results` the card reads of
    ``session``.

    An error is toned by :func:`~solvcon.pilot.panel.tone_of`.  The note
    says when the gases part into vacuum, and, in the bad tone, when a
    wave has run out of the tube.

    :param session: The :class:`~._session.ShockTubeSession` to read.
    """
    # The card reads the numbers TubeAnalysis measures, so the card and a
    # script judge a run alike.  A wave out of the tube leaves the run
    # valid but voids the comparison.
    note = None
    if session.exact.vacuum:
        note = ("The gases part into vacuum, so there is no star state.",
                'muted')
    elif _escaped(session):
        note = ("A wave has run out of the tube, so the exact solution no "
                "longer applies.", 'bad')
    margin = session.analysis.MARGIN
    caption = (f"A star value is the mean of the cells between the outer "
               f"waves, leaving out {margin} cells either side of a front, "
               f"and its error is relative to the exact value.  A front "
               f"stands where the run crosses the value midway across it, "
               f"and its error is the miss over the distance it has run.")
    return _viewmodel.Results(states=_star_rows(session),
                              fronts=_front_rows(session),
                              rows=_run_rows(session), caption=caption,
                              note=note)


def profile_frame(session, display, problem="Sod"):
    """Return the :class:`~._viewmodel.ProfileFrame` of ``session`` that
    the :class:`~._viewmodel.Display` ``display`` asks for.

    :param session: The :class:`~._session.ShockTubeSession` to read.
    :param display: The :class:`~._viewmodel.Display` to read it as.
    :param problem: The name of the problem the caption leads with.
    """
    analysis, time = session.analysis, session.time
    xmin, xmax = session.xlim
    if 'cells' == display.exact:
        exact = analysis.steps(time)
        line = "exact cell means"
    else:
        coord = np.linspace(xmin, xmax, EXACT_POINTS, dtype='float64')
        exact = (coord, analysis.sample(coord, time))
        line = "exact"
    caption = (f"{problem}, step {session.step}, t = {time:.4g}, waves "
               f"{session.exact.pattern}")
    return _viewmodel.ProfileFrame(
        xlim=(xmin, xmax), quantities=tuple(display.quantities),
        exact=exact, points=(session.x, session.values()),
        reference=analysis.means(time),
        fronts=tuple(session.exact.positions(time)),
        x0=session.exact.x0,
        legend=(('line', line), ('marker', MARCH_LABEL)), caption=caption)


def xt_frame(session):
    """Return the :class:`~._viewmodel.XtFrame` of the density history of
    ``session``, up to the time of its step cap.

    :param session: The :class:`~._session.ShockTubeSession` to read.
    """
    exact = session.exact
    return _viewmodel.XtFrame(
        xlim=session.xlim, x=session.x,
        times=np.array([time for time, _ in session.density_rows],
                       dtype='float64'),
        rows=np.array([row for _, row in session.density_rows],
                      dtype='float64'),
        waves=tuple((wave.kind, wave.speeds) for wave in exact.waves
                    if 'vacuum' != wave.kind),
        x0=exact.x0, t_end=session.t_end,
        label=f"density, {MARCH_LABEL}")


def strip_state(session, state, failure='', dirty=False, can_reset=True):
    """Return the :class:`~solvcon.pilot.panel.StripState` of ``session``
    in ``state``.

    :param session: The :class:`~._session.ShockTubeSession` to read, or
        None, which reads step 0 of 0.
    :param state: One of :data:`~solvcon.pilot.panel.RUN_STATES`.
    :param failure: What made the run fail, or empty.
    :param dirty: Whether the parameters hold edits the run has not taken.
    :param can_reset: Whether a reset can rebuild the run from the edits.
    """
    if session is None:
        return panel.StripState(state, 0, 0, 0.0, 0.0, failure, dirty,
                                can_reset)
    return panel.StripState(
        state=state, step=session.step, max_steps=session.max_steps,
        time=session.time, t_end=session.t_end, failure=failure,
        dirty=dirty, can_reset=can_reset)


def _pending(text, dirty, invalid):
    """Return ``text``, marked when its edits wait for Reset or hold a
    value the setup rejects."""
    if invalid:
        return text + "  (invalid)"
    return text + "  (pending)" if dirty else text


def problem_summary(setup, dirty, invalid=False):
    """Return the summary of the problem group of ``setup``, ``p 1 | 0.1,
    rho 1 | 0.125``, marked when its edits wait for Reset or are wrong.

    :param setup: The :class:`~._setup.ShockTubeSetup` the group holds.
    :param dirty: Whether its edits wait for Reset.
    :param invalid: Whether it holds a value the setup rejects.
    """
    text = _setup.number_text
    return _pending(f"p {text(setup.p_left)} | {text(setup.p_right)}, rho "
                    f"{text(setup.rho_left)} | {text(setup.rho_right)}",
                    dirty, invalid)


def numerics_summary(setup, dirty, invalid=False):
    """Return the summary of the numerics group of ``setup``, ``201 on
    [-10, 10], dt 0.05``, marked when its edits wait for Reset or are
    wrong.

    :param setup: The :class:`~._setup.ShockTubeSetup` the group holds.
    :param dirty: Whether its edits wait for Reset.
    :param invalid: Whether it holds a value the setup rejects.
    """
    text = _setup.number_text
    return _pending(f"{text(setup.ncoord)} on [{text(setup.xmin)}, "
                    f"{text(setup.xmax)}], dt {text(setup.time_increment)}",
                    dirty, invalid)


def display_summary(display):
    """Return the summary of the display group, ``3 panels, fronts, x-t``:
    how many quantities are drawn, or the one drawn, and the options on.

    :param display: The :class:`~._viewmodel.Display` to sum up.
    """
    count = len(display.quantities)
    parts = [f"{count} panels" if count > 1
             else _viewmodel.QUANTITY_LABELS[display.quantities[0]]]
    parts += [word for word, on in (("cell means", 'cells' == display.exact),
                                    ("fronts", display.fronts),
                                    ("log", display.log),
                                    ("residual", display.residual),
                                    ("x-t", display.xt)) if on]
    return ", ".join(parts)


class ShockTubePresenter(object):
    """Read a run out into the figure of the window and the results card
    of the control column.

    The presenter holds no run: each read-out takes the session and the
    display the controller decided to show.  Either target may be None,
    so a test or a script drives the half it has.
    """

    def __init__(self, window, column):
        #: What holds the figure, or None.
        self.window = window
        #: What holds the results card, or None.
        self.column = column

    def show(self, session, display, problem="Sod"):
        """Read ``session`` out as the :class:`~._viewmodel.Display`
        ``display`` asks: the panels and the x-t diagram into the figure,
        the results into the card.

        :param session: The :class:`~._session.ShockTubeSession` to read.
        :param display: The :class:`~._viewmodel.Display` to read it as.
        :param problem: The name of the problem the caption leads with.
        """
        if self.window is not None:
            figure = self.window.figure
            figure.show_profile(profile_frame(session, display, problem),
                                display)
            figure.show_xt(xt_frame(session) if display.xt else None)
        if self.column is not None:
            self.column.card.show_results(results(session))

    def blank(self):
        """Clear the figure and the card, for no run to show."""
        if self.window is not None:
            self.window.figure.show_profile(None, None)
            self.window.figure.show_xt(None)
        if self.column is not None:
            self.column.card.show_results(None)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
