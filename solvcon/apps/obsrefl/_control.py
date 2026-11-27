# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Drive one reflection run between the window and the control
column."""

import numpy as np

from ...pilot import panel

from . import _presenter
from . import _setup

__all__ = [
    'ReflectionController',
]


class ReflectionController(panel.RunController):
    """March one reflection run into the window and its control column on
    the frame timer, as :class:`~solvcon.pilot.panel.RunController` does.

    The free stream and the numerics wait for Reset; a density that is no
    longer finite fails the run.

    :param window: The :class:`~._window.ReflectionWindow` to drive.
    :param column: The :class:`~._column.ControlColumn` beside it.
    :param setup: The :class:`~._setup.ReflectionSetup` to start from, or
        None for the defaults.
    """

    def __init__(self, window, column, setup=None):
        #: The :class:`~._presenter.ReflectionPresenter`.
        self.presenter = _presenter.ReflectionPresenter(window, column)
        super().__init__(window, column, _setup.ReflectionSetup()
                         if setup is None else setup)
        window.figure.probe_moved.connect(self._on_probe_moved)

    def setup_groups(self):
        column = self.column
        return ((column.freestream, _presenter.freestream_summary),
                (column.numerics, _presenter.numerics_summary))

    def take_session(self, session):
        shock = session.shock
        self.window.figure.set_mesh(shock.mesh, shock.shock_path())

    def _on_probe_moved(self):
        self.presenter.show_probe(self.session, self.display)

    def check_march(self, session):
        # Nothing drawn after a density gone non-finite would mean anything.
        if not np.isfinite(session.field.field('density')).all():
            raise FloatingPointError("the density is no longer finite")

    def show_session(self, session, display):
        self.presenter.show(session, display)

    def blank(self):
        self.presenter.blank()

    def strip_state(self, session, state, failure, dirty, can_reset):
        return _presenter.strip_state(session, state, failure, dirty,
                                      can_reset)

    def display_summary(self, display):
        return _presenter.display_summary(display)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
