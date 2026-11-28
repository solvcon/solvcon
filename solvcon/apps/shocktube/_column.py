# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The control column of the shock-tube app."""

from ...pilot import panel

from . import _forms

__all__ = [
    'ControlColumn',
    'ResultsCard',
]


class ResultsCard(panel.ResultsCard):
    """The verdict of the run: the star state the run reaches between the
    outer waves and where it puts the shocks and the contact, each against
    the exact solution with the error between them, and the figures of the
    run, such as the mass drift and the CFL number.
    """

    SECTIONS = (('states', "Star state"), ('fronts', "Fronts"))

    def empty_text(self, section, results):
        if 'states' == section:
            return "No star state between the waves"
        return "No shock or contact to place"


class ControlColumn(panel.ControlColumn):
    """The control column: the results card over the groups of the
    problem and the numerics, which wait for Reset, and the display, which
    applies at once.
    """

    def __init__(self, parent=None):
        self.problem = _forms.ProblemGroup()
        self.numerics = _forms.NumericsGroup()
        self.display = _forms.DisplayGroup()
        super().__init__(ResultsCard(), (self.problem, self.numerics,
                                         self.display), self.display, parent)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
