# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The figure of the reflection app: the domain over the profile."""

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette
from PySide6.QtWidgets import (QHBoxLayout, QLabel, QSplitter, QVBoxLayout,
                               QWidget)

from ... import pilot
from ...pilot import visual

from . import _domainview
from . import _profile

__all__ = [
    'ReflectionFigure',
]


class ReflectionFigure(pilot.PaletteStyle, QWidget):
    """The domain pane, with its caption and color bar, stacked over the
    profile plot in a splitter.

    The figure holds no run: its owner hands it the mesh of a run once
    through :meth:`set_mesh`, then a :class:`~._viewmodel.FieldFrame`
    through :meth:`show_field` and a :class:`~._viewmodel.ProfileFrame`
    through :meth:`show_profile`.  The cells and the gauge of the profile
    cut are drawn by the shared
    :class:`~solvcon.pilot.visual.FieldColorizer`.

    :param mgr: The pilot's :class:`~solvcon.pilot.RManager`, or None for
        its instance.
    """

    #: The share of the height the domain pane and the profile take.
    STRETCH = (3, 2)
    #: What the caption reads without a run.
    BLANK = "No run"

    def __init__(self, parent=None, mgr=None):
        super().__init__(parent)
        self.setObjectName("figure")
        # A palette event can reach the figure while it builds, ahead of
        # the caption the style sets.
        self.caption = None
        self._colorizer = None
        self._gauge = None

        #: The :class:`~._domainview.DomainView`.
        self.domain = _domainview.DomainView(mgr=mgr)
        #: The :class:`~solvcon.pilot.visual.ColorBar` beside it.
        self.bar = visual.ColorBar(vertical=True, themed=True, label="")
        #: The :class:`~._profile.ProfilePlot` under it.
        self.profile = _profile.ProfilePlot()
        #: The caption over the view.
        self.caption = QLabel(self.BLANK)
        self.caption.setObjectName("figureCaption")

        #: The domain pane: the caption, the view, and the bar.
        self.pane = QWidget()
        self.pane.setBackgroundRole(QPalette.Base)
        self.pane.setAutoFillBackground(True)
        view = QHBoxLayout()
        view.setContentsMargins(0, 0, 0, 0)
        view.setSpacing(6)
        view.addWidget(self.domain, 1)
        view.addWidget(self.bar)
        pane = QVBoxLayout(self.pane)
        pane.setContentsMargins(0, 0, 0, 0)
        pane.setSpacing(4)
        pane.addWidget(self.caption)
        pane.addLayout(view, 1)

        #: The splitter between the pane and the profile.
        self.splitter = QSplitter(Qt.Vertical)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.addWidget(self.pane)
        self.splitter.addWidget(self.profile)
        for index, stretch in enumerate(self.STRETCH):
            self.splitter.setStretchFactor(index, stretch)
        # The 3D view and the bar hint at no height, so the split starts
        # from the shares rather than from the hints.
        self.splitter.setSizes([100 * stretch for stretch in self.STRETCH])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.addWidget(self.splitter)
        self.setBackgroundRole(QPalette.Base)
        self.setAutoFillBackground(True)
        self._apply_style()

    def _apply_style(self):
        """Set the caption in the ink of the ground it sits on."""
        if self.caption is None:
            return
        ink = pilot.AppInk(self)
        palette = self.caption.palette()
        palette.setColor(QPalette.WindowText, ink.ink)
        self.caption.setPalette(palette)

    def set_mesh(self, mesh, path=None):
        """Show a mesh and color its cells from now on.

        :param mesh: The :class:`~solvcon.core.StaticMesh` of the run.
        :param path: A polyline of two or three corners to overlay, or None.
        """
        self._colorizer = visual.FieldColorizer(mesh)
        self._gauge = (None, None)
        self.domain.viewer.update_mesh(mesh)
        if path is not None:
            self.domain.viewer.draw_path(path)

    def show_field(self, frame):
        """Color the cells by a frame over its pinned range, mark its
        gauge, and scale the bar alike.

        :param frame: The :class:`~._viewmodel.FieldFrame` to show, or None
            to blank the caption and the bar.
        """
        if frame is None:
            self.caption.setText(self.BLANK)
            self.bar.show_scale(None, None)
            return
        self.caption.setText(frame.caption)
        self.bar.show_scale(frame.lo, frame.hi, frame.label)
        if self._colorizer is None:
            return
        self._set_gauge(frame.gauge_cells, frame.gauge_height)
        colorizer = self._colorizer
        self.domain.viewer.draw_field(
            colorizer.verts,
            colorizer.colors(np.asarray(frame.values, dtype='float64'),
                             frame.lo, frame.hi),
            colorizer.indices)

    def _set_gauge(self, cells, height):
        """Hand the colorizer the gauge only when it changes, since a new
        gauge repacks the vertices of every cell.

        :param cells: The cells to outline, or None.
        :param height: The height of the profile line, or None.
        """
        key = (None if cells is None else np.asarray(cells).tobytes(),
               height)
        if key == self._gauge:
            return
        self._gauge = key
        self._colorizer.set_gauge(cells, height)

    def set_bar_visible(self, on):
        """Show or hide the color bar beside the view.

        :param on: Whether the bar shows.
        """
        self.bar.setVisible(bool(on))

    def show_profile(self, frame):
        """Draw a frame in the profile plot.

        :param frame: The :class:`~._viewmodel.ProfileFrame` to draw, or
            None to blank the plot.
        """
        self.profile.show_profile(frame)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
