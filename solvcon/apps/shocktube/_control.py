# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Drive one shock-tube run between the window and the control column."""

import numpy as np

from ...pilot import panel

from . import _presenter
from . import _setup

__all__ = [
    'ShockTubeController',
]


class ShockTubeController(panel.RunController):
    """March one shock-tube run into the window and its control column on
    the frame timer, as :class:`~solvcon.pilot.panel.RunController` does.

    The problem and the numerics wait for Reset; a density that is no
    longer finite fails the run.

    :param window: The :class:`~._window.ShockTubeWindow` to drive.
    :param column: The :class:`~._column.ControlColumn` beside it.
    :param setup: The :class:`~._setup.ShockTubeSetup` to start from, or
        None for Sod's problem.
    """

    def __init__(self, window, column, setup=None):
        #: The :class:`~._presenter.ShockTubePresenter`.
        self.presenter = _presenter.ShockTubePresenter(window, column)
        super().__init__(window, column, _setup.ShockTubeSetup()
                         if setup is None else setup)

    def setup_groups(self):
        column = self.column
        return ((column.problem, _presenter.problem_summary),
                (column.numerics, _presenter.numerics_summary))

    def check_march(self, session):
        # Nothing drawn after a density gone non-finite would mean anything.
        if not np.isfinite(session.values()['density']).all():
            raise FloatingPointError("the density is no longer finite")

    def show_session(self, session, display):
        self.presenter.show(session, display, self.setup.problem())

    def blank(self):
        self.presenter.blank()

    def strip_state(self, session, state, failure, dirty, can_reset):
        return _presenter.strip_state(session, state, failure, dirty,
                                      can_reset)

    def display_summary(self, display):
        return _presenter.display_summary(display)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
