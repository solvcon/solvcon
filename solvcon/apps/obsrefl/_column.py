# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The control column of the reflection app."""

from ...pilot import panel

from . import _forms

__all__ = [
    'ControlColumn',
    'ResultsCard',
]


class ResultsCard(panel.ResultsCard):
    """The verdict of the run: the field computed in each zone against the
    analytic value and the error between them, the fitted reflected shock
    against the exact one, and the figures of the run, such as the mass
    drift and the CFL range.
    """

    SECTIONS = (('zones', "Zones"), ('fits', "Shock fit"))

    def head_text(self, section, title, results):
        if 'zones' == section:
            return f"{title}  {results.field}"
        return title

    def empty_text(self, section, results):
        if 'zones' == section:
            return f"No exact {results.field} to compare with"
        return "Not formed yet"


class ControlColumn(panel.ControlColumn):
    """The control column: the results card over the groups of the free
    stream and the numerics, which wait for Reset, and the display, which
    applies at once.
    """

    def __init__(self, parent=None):
        self.freestream = _forms.FreeStreamGroup()
        self.numerics = _forms.NumericsGroup()
        self.display = _forms.DisplayGroup()
        super().__init__(ResultsCard(), (self.freestream, self.numerics,
                                         self.display), self.display, parent)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
