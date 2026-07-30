# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the Design page of the Painter inspector.
"""

import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot.base import _gui
    from solvcon.pilot.painter import _design
    from PySide6 import QtCore, QtGui, QtWidgets
except ImportError:
    pilot = None

# The offscreen platform cannot back a live window; neither can the headless
# Windows CI runner, whose WARP software rasterizer faults on one.
NO_LIVE_WINDOW = ((os.getenv('QT_QPA_PLATFORM') or '').startswith('offscreen')
                  or ('nt' == os.name and bool(os.getenv('GITHUB_ACTIONS'))))


def _click(widget, x, y):
    """Post a synthetic left-button press and release to ``widget``."""
    pos = QtCore.QPointF(x, y)
    glob = QtCore.QPointF(widget.mapToGlobal(pos.toPoint()))
    for etype, button, buttons in (
            (QtCore.QEvent.Type.MouseButtonPress,
             QtCore.Qt.LeftButton, QtCore.Qt.LeftButton),
            (QtCore.QEvent.Type.MouseButtonRelease,
             QtCore.Qt.LeftButton, QtCore.Qt.NoButton)):
        QtWidgets.QApplication.sendEvent(
            widget,
            QtGui.QMouseEvent(etype, pos, glob, button, buttons,
                              QtCore.Qt.NoModifier))


@unittest.skipIf(NO_LIVE_WINDOW or not solvcon.HAS_PILOT,
                 "pilot windows need a real window surface")
class PainterDesignCanvasTC(unittest.TestCase):
    """The page against a real canvas, bound by the Painter dock."""

    def setUp(self):
        self.mgr = _gui.controller.build()
        self.painter = _gui.controller.painter
        self.painter._ensure_dock()
        self.widget = self.mgr.add2DWidget()
        self.widget.setDrawTool("select")
        self.world = solvcon.WorldFp64()
        self.sid = self.world.add_rectangle(-2, -1, 2, 1)
        self.widget.updateWorld(self.world)
        view = solvcon.ViewTransform2dFp64()
        view.pan(100.0, 100.0)
        view.zoom = 20.0
        # Set the view before showing so the resize auto-centering, which a
        # well-formed transform disables, leaves the mapping deterministic.
        self.widget.setViewTransform(view)
        self.mgr.show()
        self.sub = self.mgr.mdiArea.subWindowList()[-1]
        self.sub.show()
        self.mgr.mdiArea.setActiveSubWindow(self.sub)
        QtWidgets.QApplication.processEvents()

    def test_the_page_follows_the_active_canvas(self):
        page = self.painter.panel.design
        # Activation is delivered through a zero timer, so let it run.
        QtWidgets.QApplication.processEvents()
        _click(self.sub.widget(), 100, 100)
        self.assertEqual(self.widget.selectedShape, self.sid)
        page.refresh()
        self.assertEqual(page._name.text(), f"Rectangle {self.sid}")
        self.assertAlmostEqual(page.fields["W"].value(), 4)

    def test_closing_the_canvas_leaves_the_page_standing(self):
        # The sub-window deletes the canvas on close, and a page that held it
        # would read freed memory on its next poll.
        page = self.painter.panel.design
        self.sub.close()
        QtWidgets.QApplication.processEvents()
        page.refresh()
        self.assertEqual(page._name.text(),
                         _design.DesignPage.EMPTY_TEXT)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
