# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The profile plot of the reflection figure: the field along a
cut."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QVBoxLayout, QWidget

from ... import pilot
from ...pilot import visual

from . import _ink

__all__ = [
    'ProfilePlot',
]


class ProfilePlot(pilot.PaletteStyle, QWidget):
    """The exact solution as a thin ink line and the computed values as
    circles in their data color, in a
    :class:`~solvcon.pilot.visual.LinePlotWidget`, so the gap between them
    is the error at that station.

    A zoomed or panned view holds over the frames of one
    :attr:`~._viewmodel.ProfileFrame.key`, a blank included, and a new key
    drops it.
    """

    EXACT_WIDTH = 1.25
    MARKER_SIZE = 5.0

    def __init__(self, parent=None):
        super().__init__(parent)
        #: The :class:`~._viewmodel.ProfileFrame` shown, or None.
        self.frame = None
        self._key = None
        #: The :class:`~solvcon.pilot.visual.LinePlotWidget` drawn.
        self.plot = visual.LinePlotWidget(title="line profile", xlabel="x")
        #: The series of the exact solution.
        self.exact = self.plot.add_series("exact")
        self.exact.line_width = self.EXACT_WIDTH
        #: The series of the computed values.
        self.computed = self.plot.add_series("computed",
                                             marker=self.MARKER_SIZE)
        # A click on the plot takes the focus off a control, so the run
        # keys of the window reach it.
        self.plot.setFocusPolicy(Qt.ClickFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.plot)
        self._apply_style()

    def _apply_style(self):
        ink = _ink.Ink(self)
        for series, color in ((self.exact, ink.ink),
                              (self.computed, ink.series('computed'))):
            series.color = pilot.PlotColor(color.red(), color.green(),
                                           color.blue(), color.alpha())
        self.plot.update()

    def show_profile(self, frame):
        """Draw a frame.

        :param frame: The :class:`~._viewmodel.ProfileFrame` to draw, or
            None to blank the plot.
        """
        self.frame = frame
        if frame is None:
            self.exact.clear_data()
            self.computed.clear_data()
        else:
            if frame.key != self._key:
                self._key = frame.key
                self.plot.reset_view()
            x = visual.plot_array(frame.x)
            self.exact.set_data(x, visual.plot_array(frame.exact))
            self.computed.set_data(x, visual.plot_array(frame.computed))
            self.plot.set_ylabel(frame.label)
        self.plot.refresh()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
