# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Read a reflection run out into what the widgets of the app show.

The presenter stands between the run and the widgets, as in the
model-view-presenter pattern.  It reads a
:class:`~._session.ReflectionSession` into the plain records of
:mod:`._viewmodel` and hands each widget its record, so a widget draws
what it is handed and holds none of the logic that decides what a run
shows.
"""

import math

import numpy as np

from . import _viewmodel

__all__ = [
    'display_summary',
    'field_frame',
    'freestream_summary',
    'numerics_summary',
    'profile_frame',
    'results',
    'strip_state',
]

#: The rows of the fits table, in the order the card shows them.
FIT_LABELS = ('shock angle', 'wall x')


def _no_cells(zone):
    """Return the cells of a zone the margin leaves empty."""
    return (_viewmodel.Cell("no cells", 'muted',
                            f"no cell of zone {zone} stands clear of the "
                            f"shocks by the margin"),
            _viewmodel.Cell("-", 'muted'))


def _zone_rows(infos):
    """Return a :class:`~._viewmodel.ResultRow` per zone of ``infos``."""
    rows = []
    for info in infos:
        exact = _viewmodel.Cell(_viewmodel.format_number(info.analytic))
        if 0 == info.count:
            computed, error = _no_cells(info.zone)
        else:
            computed = _viewmodel.Cell(
                _viewmodel.format_number(info.computed), '',
                f"mean over {info.count} cells")
            tip = ("the exact value is 0, so no error is relative to it"
                   if 0 == info.analytic else "")
            error = _viewmodel.Cell(_viewmodel.format_error(info.error),
                                    _viewmodel.tone_of(info.error), tip)
        rows.append(_viewmodel.ResultRow(f"zone {info.zone}", computed,
                                         exact, error))
    return tuple(rows)


def _fit_rows(session):
    """Return the rows of the fitted shock angle and its wall abscissa.

    Both come off the one line fitted through the density crossings, so
    a line that has not formed leaves both without a value.
    """
    fit = session.fit_incident_angle()
    point = session.reflection_point()
    formed = fit.npoint >= 2 and math.isfinite(fit.degree)
    unformed = _viewmodel.Cell(
        "not formed", 'muted',
        f"the fit ran through {fit.npoint} density "
        f"{'crossing' if 1 == fit.npoint else 'crossings'}; a shock "
        f"descending to the wall takes two or more")
    rows = []
    for label, value, target, error, unit in (
            (FIT_LABELS[0], fit.degree, fit.analytic, fit.error,
             "in degrees from the free stream"),
            (FIT_LABELS[1], point.x, point.analytic, point.error,
             "where the incident shock meets the wall")):
        if math.isfinite(target):
            exact = _viewmodel.Cell(_viewmodel.format_number(target), '',
                                    unit)
        else:
            exact = _viewmodel.Cell("none", 'muted',
                                    "the domain is too short to hold the "
                                    "reflection")
        if not (formed and math.isfinite(value)):
            rows.append(_viewmodel.ResultRow(label, unformed, exact,
                                             _viewmodel.Cell("-", 'muted')))
            continue
        rows.append(_viewmodel.ResultRow(
            label, _viewmodel.Cell(_viewmodel.format_number(value), '',
                                   f"{unit}; fitted through {fit.npoint} "
                                   f"crossings"),
            exact, _viewmodel.Cell(_viewmodel.format_error(error),
                                   _viewmodel.tone_of(error))))
    return tuple(rows)


def _run_rows(session):
    """Return the ``(label, text)`` rows of the run figures: the mass drift
    since the first record, signed, and the CFL range of the last one."""
    records = session.history.records
    if not records:
        return (('mass drift', "-"), ('CFL', "-"))
    first, last = records[0], records[-1]
    drift = (last.mass - first.mass) / first.mass
    return (('mass drift', _viewmodel.format_signed(drift)),
            ('CFL', f"{last.cfl_min:.2f}-{last.cfl_max:.2f}"))


def results(session, name):
    """Return the :class:`~._viewmodel.Results` the card reads of
    ``session`` for the field ``name``.

    Every number the card reads is the one :class:`~._analytic.Reflection`
    measures, so the card and a script judge a run alike.
    A zone error reads good, plain, or bad by
    :func:`~._viewmodel.tone_of`; a zone the margin leaves empty reads "no
    cells", and a fit through fewer than two crossings "not formed", both
    muted.  A field without an exact value has no zone rows, and the fits,
    which always read the density, stay.  A reflected shock through the top
    inlet leaves the run parameters valid but the comparison void, which
    the note says in the bad tone.
    """
    analysis = session.analysis
    label = _viewmodel.FIELD_LABELS[name]
    infos = session.zone_info(name)
    zones = _zone_rows(infos)
    note = None
    if not infos:
        note = (f"The {label} has no exact value to compare.", 'muted')
    elif not analysis.has_reflection:
        note = ("The domain is too short to hold the reflection: there is "
                "no zone 3 and no wall x.", 'muted')
    elif session.shock.reflected_shock_hits_top():
        note = ("The reflected shock reaches the top inlet, so the "
                "three-zone analytic solution does not apply.", 'bad')
    else:
        empty = [str(info.zone) for info in infos if 0 == info.count]
        if empty:
            note = (f"The margin leaves zone {', '.join(empty)} without "
                    f"cells; a finer mesh fills it.", 'muted')
    caption = (f"A zone mean leaves out the cells within "
               f"{analysis.margin:.0%} of the domain height of a shock, and "
               f"its error is relative to the exact zone value.  The fits "
               f"run a line through where the density crosses its mid value "
               f"upstream of the wall.")
    return _viewmodel.Results(field=label, zones=zones,
                              fits=_fit_rows(session),
                              rows=_run_rows(session), caption=caption,
                              note=note)


def field_frame(session, display):
    """Return the :class:`~._viewmodel.FieldFrame` of ``session`` that the
    :class:`~._viewmodel.Display` ``display`` asks for.

    The colors span :meth:`~._analytic.Reflection.color_range`, so a color
    reads the same value over the run; the caption reads the range of the
    frame itself.
    """
    analysis = session.analysis
    name = display.field
    label = _viewmodel.FIELD_LABELS[name]
    values = np.array(session.field.field(name), dtype='float64')
    vmin, vmax = float(values.min()), float(values.max())
    lo, hi = analysis.color_range(name, vmin, vmax)
    height = analysis.cut_height(display.cut)
    cells = None
    if display.mark_cells:
        cells = np.nonzero(analysis.profile_cells(height))[0].astype('int64')
    caption = (f"{label}, step {session.step}: "
               f"{_viewmodel.format_number(vmin, 3)} to "
               f"{_viewmodel.format_number(vmax, 3)}")
    return _viewmodel.FieldFrame(
        name=name, label=label, values=values, lo=lo, hi=hi,
        caption=caption, gauge_cells=cells,
        gauge_height=height if display.mark_line else None)


class _Unpaired(object):
    """Stand in for a :class:`~._analytic.Reflection` whose field has no
    analytic value, so :meth:`~._analytic.Reflection.profile` cuts the
    computed field and pairs it with NaN.

    The analysis raises for a field it holds no zone value of, and the cut
    is otherwise the same; a fold-back lets the profile skip the pairing.
    """

    def __init__(self, analysis):
        self._analysis = analysis

    def __getattr__(self, name):
        return getattr(self._analysis, name)

    def analytic_profile(self, xs, height, name='density'):
        return np.full(len(xs), np.nan, dtype='float64')


def profile_frame(session, display):
    """Return the :class:`~._viewmodel.ProfileFrame` of ``session`` along
    the cut ``display`` places.

    Both series come off one cut, so they share their abscissae; a field
    without an analytic value pairs the computed one with NaN.  The frame
    is keyed by the field, so the plot keeps its zoom while the run
    marches and drops it for another field.
    """
    analysis = session.analysis
    name = display.field
    label = _viewmodel.FIELD_LABELS[name]
    height = analysis.cut_height(display.cut)
    if name in analysis.zone_fields():
        cut = analysis.profile(height, name)
    else:
        cut = type(analysis).profile(_Unpaired(analysis), height, name)
    return _viewmodel.ProfileFrame(
        x=np.array(cut.x, dtype='float64'),
        computed=np.array(cut.computed, dtype='float64'),
        exact=np.array(cut.analytic, dtype='float64'), label=label,
        key=name)


def strip_state(session, state, failure='', dirty=False, can_reset=True):
    """Return the :class:`~._viewmodel.StripState` of ``session`` in
    ``state``; the time is the steps marched times the time increment, and
    the end time that of the step cap.  Without a session the strip reads
    step 0 of 0."""
    if session is None:
        return _viewmodel.StripState(state, 0, 0, 0.0, 0.0, failure, dirty,
                                     can_reset)
    dt = float(session.shock.svr.time_increment)
    return _viewmodel.StripState(
        state=state, step=session.step, max_steps=session.max_steps,
        time=session.step * dt, t_end=session.max_steps * dt,
        failure=failure, dirty=dirty, can_reset=can_reset)


def _pending(text, dirty):
    """Return ``text``, marked when its edits wait for Reset."""
    return text + "  (pending)" if dirty else text


def freestream_summary(setup, dirty):
    """Return the summary of the free-stream group of ``setup``, ``Mach 3,
    10 deg, gamma 1.4``, and whether its edits wait for Reset."""
    return _pending(f"Mach {setup.mach:g}, {setup.angle:g} deg, gamma "
                    f"{setup.gamma:g}", dirty)


def numerics_summary(setup, dirty):
    """Return the summary of the numerics group of ``setup``, ``64 x 16
    unstructured, dt 0.002``, and whether its edits wait for Reset."""
    return _pending(f"{setup.nx:g} x {setup.ny:g} {setup.cell_type}, dt "
                    f"{setup.time_increment:g}", dirty)


def display_summary(display):
    """Return the summary of the display group, ``density, bar, cut
    0.50``: the field, the marks that show, and the cut height."""
    parts = [_viewmodel.FIELD_LABELS[display.field]]
    parts += [word for word, on in (("bar", display.bar),
                                    ("cells", display.mark_cells),
                                    ("line", display.mark_line)) if on]
    parts.append(f"cut {display.cut:.2f}")
    return ", ".join(parts)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
