# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The sub-window holding the analysis plots."""

from ...pilot import base

from ._plots import AnalysisLinePlots

__all__ = [  # noqa: F822
    'LinePlotViewer',
]


class LinePlotViewer(object):
    """Own the sub-window holding the analysis plots.

    The plots are built once and outlive the window: a reopened window is
    handed the same :class:`~._plots.AnalysisLinePlots` back rather than a
    fresh one that dropped what it was showing.

    :ivar lineplots: The :class:`~._plots.AnalysisLinePlots` widget,
        always live.
    """

    SIZE = (520, 620)

    def __init__(self, mgr):
        self._mgr = mgr
        # Owner-supplied callback fired when the sub-window closes from any
        # source.
        self.closed = None
        self.lineplots = AnalysisLinePlots()
        self._subwin = None
        self._close_filter = None

    @property
    def is_open(self):
        return self._subwin is not None

    def open(self):
        if self._subwin is not None:
            return
        # A parented widget is deleted with its window, so the plots are
        # unparented on a close and handed to the new sub-window here.  The
        # manager shows the window and marks it delete-on-close.
        self.lineplots.setParent(None)
        self._subwin = self._mgr.addSubWindow(self.lineplots)
        self._subwin.setWindowTitle("reflection analysis")
        self._subwin.resize(*self.SIZE)
        self._mgr.addSubWindowGrip(self._subwin)
        self._close_filter = base.SubWindowCloseFilter(
            self._on_subwin_closed, self._subwin)
        self._subwin.installEventFilter(self._close_filter)

    def close(self):
        """Close the sub-window; its close event fires :attr:`closed`."""
        if self._subwin is not None:
            self._subwin.close()
        else:
            self._on_subwin_closed()

    def _on_subwin_closed(self):
        # Take the plots back out before Qt deletes the window around them,
        # then drop the window and tell the owner.
        self.lineplots.setParent(None)
        self._subwin = None
        self._close_filter = None
        if self.closed is not None:
            self.closed()

    def show_run(self, session, name, height):
        if self._subwin is None:
            return
        self.lineplots.show_run(session, name, height)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
