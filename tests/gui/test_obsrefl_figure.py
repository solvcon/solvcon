# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The figure of the reflection app, drawn from frames built by hand over a
coarse run.

The domain pane hosts a 3D view, which renders only on a live window
surface, so these run in the window lane.
"""

import platform
import unittest

import numpy as np

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from PySide6.QtCore import QEvent, QPointF, Qt
        from PySide6.QtGui import QMouseEvent
        from PySide6.QtWidgets import QApplication
        from solvcon.apps.obsrefl import _figure, _session, _viewmodel
except ImportError:
    pilot = None


def _settle():
    for _ in range(20):
        QApplication.processEvents()


def _colored(widget):
    """Return the count of pixels of a grab of ``widget`` that carry a
    color rather than a grey, skipping where grabbing is unreliable."""
    if platform.system() == "Windows":
        raise unittest.SkipTest("offscreen grabbing unreliable on Windows CI")
    image = widget.grab().toImage()
    if image.isNull() or 0 == image.width():
        raise unittest.SkipTest("the window gave no frame")
    count = 0
    for iy in range(0, image.height(), 3):
        for ix in range(0, image.width(), 3):
            color = image.pixelColor(ix, iy)
            channels = (color.red(), color.green(), color.blue())
            count += max(channels) - min(channels) > 60
    return count


def _move(widget, buttons=None):
    """Post a move of the cursor to the middle of ``widget``, with
    ``buttons`` held, none by default."""
    buttons = Qt.NoButton if buttons is None else buttons
    at = QPointF(widget.width() / 2.0, widget.height() / 2.0)
    QApplication.postEvent(widget, QMouseEvent(
        QEvent.MouseMove, at, widget.mapToGlobal(at), Qt.NoButton, buttons,
        Qt.NoModifier))
    _settle()


def _field_frame(session, cells=None, height=None):
    values = np.asarray(session.field.field('density'), dtype='float64')
    vmin, vmax = float(values.min()), float(values.max())
    lo, hi = session.analysis.color_range('density', vmin, vmax)
    return _viewmodel.FieldFrame(
        name='density', label='density', values=values, lo=lo, hi=hi,
        caption=f"density, step {session.step}: {vmin:.3g} to {vmax:.3g}",
        gauge_cells=cells, gauge_height=height)


@testing.requires_live_window
class ReflectionFigureTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mgr = pilot.RManager.instance.setUp()
        window = cls.mgr.mainWindow
        window.resize(1000, 800)
        window.show()
        _settle()
        cls.session = _session.ReflectionSession(nx=8, ny=3, max_steps=10)
        cls.session.advance()

    def _figure(self):
        figure = _figure.ReflectionFigure(mgr=self.mgr)
        figure.resize(800, 600)
        self.addCleanup(testing.dispose_widget, figure)
        figure.set_mesh(self.session.shock.mesh,
                        self.session.shock.shock_path())
        return figure

    def test_a_field_colors_the_cells_and_scales_the_bar(self):
        figure = self._figure()
        figure.show()
        _settle()
        bare = _colored(figure.domain)
        frame = _field_frame(self.session)
        figure.show_field(frame)
        _settle()
        self.assertGreater(_colored(figure.domain), bare)
        self.assertEqual((frame.lo, frame.hi), (figure.bar.lo,
                                                figure.bar.hi))
        self.assertEqual(frame.caption, figure.caption.text())

    def test_a_blank_field_blanks_the_caption_and_the_bar(self):
        figure = self._figure()
        figure.show_field(_field_frame(self.session))
        figure.show_field(None)
        self.assertEqual(figure.BLANK, figure.caption.text())
        self.assertIsNone(figure.bar.lo)


@testing.requires_live_window
class ReflectionProbeTC(unittest.TestCase):
    """Hovering over the domain picks the cell and the point to probe."""

    @classmethod
    def setUpClass(cls):
        cls.mgr = pilot.RManager.instance.setUp()
        cls.session = _session.ReflectionSession(nx=8, ny=3, max_steps=10)
        cls.session.advance()

    def setUp(self):
        self.figure = _figure.ReflectionFigure(mgr=self.mgr)
        self.figure.resize(800, 600)
        self.addCleanup(testing.dispose_widget, self.figure)
        self.figure.set_mesh(self.session.shock.mesh,
                             self.session.shock.shock_path())
        self.figure.show()
        _settle()
        if self.figure.domain.widget is None:
            raise unittest.SkipTest("the pilot built no 3D view")
        self.moves = []
        self.figure.probe_moved.connect(
            lambda: self.moves.append(self.figure.probe_target))

    def test_a_hover_probes_the_cell_under_the_cursor(self):
        _move(self.figure.domain.widget)
        self.assertEqual(1, len(self.moves))
        cell, (x, y) = self.figure.probe_target
        mesher = self.session.shock.mesher
        self.assertTrue(mesher.x0 <= x <= mesher.x1)
        self.assertTrue(mesher.y0 <= y <= mesher.y1)
        # The point lies in the cell the pick reports, so its corners
        # straddle it.
        _, corners = self.session.analysis.cell_corners(cell)
        self.assertTrue(corners[:, 0].min() <= x <= corners[:, 0].max())
        self.assertTrue(corners[:, 1].min() <= y <= corners[:, 1].max())
        # The same pixel again is no move.
        _move(self.figure.domain.widget)
        self.assertEqual(1, len(self.moves))

    def test_leaving_the_view_drops_the_probe(self):
        _move(self.figure.domain.widget)
        QApplication.postEvent(self.figure.domain.widget,
                               QEvent(QEvent.Leave))
        _settle()
        self.assertIsNone(self.figure.probe_target)
        self.assertEqual(2, len(self.moves))

    def test_a_drag_navigates_and_does_not_probe(self):
        _move(self.figure.domain.widget, Qt.LeftButton)
        self.assertEqual([], self.moves)

    def test_the_switch_stops_the_probe_and_hides_its_line(self):
        _move(self.figure.domain.widget)
        moved = len(self.moves)
        self.figure.set_probe_enabled(False)
        self.assertIsNone(self.figure.probe_target)
        self.assertTrue(self.figure.readout.isHidden())
        # The switches come from the presenter, which reads the probe out
        # next; a move would call it back from inside itself.
        self.assertEqual(moved, len(self.moves))
        del self.moves[:]
        _move(self.figure.domain.widget)
        self.assertEqual([], self.moves)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
