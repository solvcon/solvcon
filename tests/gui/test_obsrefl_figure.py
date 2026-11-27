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


def _at(widget, fx=0.5, fy=0.5):
    """The point at the fractions ``(fx, fy)`` of ``widget``."""
    return QPointF(widget.width() * fx, widget.height() * fy)


def _move(widget, buttons=None, at=None):
    """Post a move of the cursor to ``at``, the middle of ``widget`` by
    default, with ``buttons`` held, none by default."""
    buttons = Qt.NoButton if buttons is None else buttons
    at = _at(widget) if at is None else at
    QApplication.postEvent(widget, QMouseEvent(
        QEvent.MouseMove, at, widget.mapToGlobal(at), Qt.NoButton, buttons,
        Qt.NoModifier))
    _settle()


def _click(widget, at=None, drag=0.0):
    """Post a press of the left button at ``at`` and its release ``drag``
    pixels to the right."""
    at = _at(widget) if at is None else at
    off = QPointF(at.x() + drag, at.y())
    for kind, where, held in ((QEvent.MouseButtonPress, at, Qt.LeftButton),
                              (QEvent.MouseButtonRelease, off, Qt.NoButton)):
        QApplication.postEvent(widget, QMouseEvent(
            kind, where, widget.mapToGlobal(where), Qt.LeftButton, held,
            Qt.NoModifier))
    _settle()


def _count_color(figure, rgb, half=60):
    """Count the pixels near the middle of the domain pane in ``rgb``."""
    want = [round(255 * c) for c in rgb]
    image = figure.domain.grab().toImage()
    cx, cy = image.width() // 2, image.height() // 2
    hits = 0
    for iy in range(cy - half, cy + half):
        for ix in range(cx - half, cx + half):
            color = image.pixelColor(ix, iy)
            got = (color.red(), color.green(), color.blue())
            hits += max(abs(g - w) for g, w in zip(got, want)) < 8
    return hits


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

    def test_a_click_pins_only_with_pinning_on(self):
        view = self.figure.domain.widget
        self.figure.set_pinning(False)
        _click(view)
        self.assertEqual([], self.moves)
        self.assertFalse(self.figure.probe_pinned)
        self.figure.set_pinning(True)
        _click(view)
        self.assertTrue(self.figure.probe_pinned)
        pinned = self.figure.probe_target
        self.assertIsNotNone(pinned)
        # Hovering and leaving leave the pinned probe alone.
        _move(view, at=_at(view, 0.4, 0.4))
        QApplication.postEvent(view, QEvent(QEvent.Leave))
        _settle()
        self.assertEqual(pinned, self.figure.probe_target)
        # A drag navigates and does not move the pin.
        _click(view, at=_at(view, 0.4, 0.4), drag=30.0)
        self.assertEqual(pinned, self.figure.probe_target)

    def test_a_click_off_the_domain_lets_the_pin_go(self):
        view = self.figure.domain.widget
        _click(view)
        self.assertTrue(self.figure.probe_pinned)
        _click(view, at=_at(view, 0.01, 0.01))
        self.assertFalse(self.figure.probe_pinned)
        self.assertIsNone(self.figure.probe_target)
        # Hovering probes again.
        _move(view)
        self.assertIsNotNone(self.figure.probe_target)

    def test_a_click_on_the_pinned_cell_lets_it_go(self):
        view = self.figure.domain.widget
        _click(view)
        self.assertTrue(self.figure.probe_pinned)
        cell = self.figure.probe_target[0]
        _click(view)
        self.assertFalse(self.figure.probe_pinned)
        # The probe stays where the cursor is, and hovering moves it again.
        self.assertEqual(cell, self.figure.probe_target[0])
        _move(view, at=_at(view, 0.4, 0.4))
        self.assertNotEqual(cell, self.figure.probe_target[0])
        # A third click pins again.
        _click(view)
        self.assertTrue(self.figure.probe_pinned)

    def test_a_click_on_another_cell_lets_the_pin_go(self):
        view = self.figure.domain.widget
        _click(view)
        cell = self.figure.probe_target[0]
        other = _at(view, 0.4, 0.4)
        _click(view, at=other)
        # The probe reads the cell clicked, but holds it only on the next
        # click.
        self.assertFalse(self.figure.probe_pinned)
        self.assertNotEqual(cell, self.figure.probe_target[0])
        _click(view, at=other)
        self.assertTrue(self.figure.probe_pinned)

    def test_a_hover_rings_the_cell_and_a_pin_thickens_the_ring(self):
        if platform.system() == "Windows":
            raise unittest.SkipTest("offscreen grabbing unreliable on Windows")
        self.figure.show_field(_field_frame(self.session))
        _settle()
        core = self.figure._colorizer.CORE_COLOR
        view = self.figure.domain.widget
        self.assertEqual(0, _count_color(self.figure, core))
        _move(view)
        _settle()
        hovered = _count_color(self.figure, core)
        self.assertGreater(hovered, 0)
        _click(view)
        _settle()
        # The mark is drawn at once, with no new frame.  The pin widens the
        # core and dots the nodes; the halo keeps its width around both, so
        # only the core tells the two apart.
        pinned = _count_color(self.figure, core)
        self.assertGreater(pinned, hovered)
        _click(view)
        _settle()
        # The release leaves the hover where the click landed.
        self.assertEqual(hovered, _count_color(self.figure, core))

    def test_turning_pinning_off_releases_the_pin(self):
        view = self.figure.domain.widget
        self.figure.set_pinning(True)
        _click(view)
        moved = len(self.moves)
        self.figure.set_pinning(False)
        self.assertFalse(self.figure.probe_pinned)
        self.assertIsNone(self.figure.probe_target)
        self.assertEqual(moved, len(self.moves))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
