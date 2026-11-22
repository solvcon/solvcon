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

from . import _viewmodel

__all__ = [
    'results',
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

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
