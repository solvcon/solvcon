# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The plot that judges a reflection run against the analytic answer.

:class:`AnalysisLinePlots` draws a horizontal cut of the field over the
analytic three-zone step it is heading for, so the gap between the curves
is the error at that station, which a color map cannot show.

The widget holds no run.  Its owner pushes a session in whenever a frame
is drawn, and the plot follows it.
"""

from PySide6.QtWidgets import QWidget, QVBoxLayout

from ...visual import _plot

__all__ = [  # noqa: F822
    'AnalysisLinePlots',
]


class AnalysisLinePlots(QWidget):
    """The line profile of one run against the analytic step."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._profile = _plot.LinePlotWidget(
            title="line profile", xlabel="x", ylabel="density")
        self._computed = self._profile.add_series("computed")
        self._analytic = self._profile.add_series("analytic")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.addWidget(self._profile)

    def show_run(self, session, name, height):
        """Draw ``session`` into the plot, or blank it without one.

        Both curves come off one :meth:`~._analytic.Reflection.profile`, so
        they are sampled at the same abscissae and the gap between them is
        the error at that station rather than an artifact of the sampling.
        """
        if None is session:
            self._computed.clear_data()
            self._analytic.clear_data()
        else:
            cut = session.profile(height, name)
            self._computed.set_data(_plot._array(cut.x),
                                    _plot._array(cut.computed))
            self._analytic.set_data(_plot._array(cut.x),
                                    _plot._array(cut.analytic))
            self._profile.set_ylabel(name)
        self._profile.refresh()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
