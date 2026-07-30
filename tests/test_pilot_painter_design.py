# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the Design page of the Painter inspector.
"""

import math
import os
import unittest

import solvcon
from pilot_ci import SKIP_PILOT_WIDGETS

try:
    from solvcon import pilot
    from solvcon.pilot.base import _gui
    from solvcon.pilot import painter as _painter
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


class _StubCanvas:
    """Stand-in for the 2D canvas, so a test can set the selection directly.

    :class:`PainterDesignCanvasTC` covers the live canvas path, including
    selection written from outside a mouse gesture.
    """

    def __init__(self, world):
        self.world = world
        self.selectedShape = -1
        self.repaints = 0

    def requestRepaint(self):
        self.repaints += 1


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class PainterDesignPageTC(unittest.TestCase):
    """What the page shows for a selection, and what an edit writes back."""

    @classmethod
    def setUpClass(cls):
        # No window needed, only a live QGuiApplication to hold a widget.
        pilot.RManager.instance.setUp()

    def setUp(self):
        self.world = solvcon.WorldFp64()
        # A rectangle centered on the origin, 4 wide and 2 tall.
        self.sid = self.world.add_rectangle(-2, -1, 2, 1)
        self.canvas = _StubCanvas(self.world)
        self.page = _design.DesignPage()
        self.page.set_canvas_source(lambda: self.canvas)

    def _select(self, shape_id):
        self.canvas.selectedShape = shape_id
        self.page.refresh()

    def _header(self):
        return self.page._name.text()

    def _values(self):
        return {letter: field.value()
                for letter, field in self.page.fields.items()}

    def _commit(self, letter, text):
        """Type ``text`` into a field and finish the edit, as Enter does."""
        field = self.page.fields[letter]
        field.edit.setText(text)
        field.edit.textEdited.emit(text)
        field.edit.editingFinished.emit()

    def test_nothing_selected_leaves_the_page_empty(self):
        self.assertEqual(self._header(),
                         _design.DesignPage.EMPTY_TEXT)
        self.assertTrue(self.page._badge.isHidden())
        self.assertTrue(self.page._icon.isHidden())
        for letter, field in self.page.fields.items():
            with self.subTest(letter=letter):
                self.assertFalse(field.isEnabled())
                self.assertEqual(field.edit.text(), "")

    def test_selection_fills_the_header_and_the_position(self):
        self._select(self.sid)
        self.assertEqual(self._header(), f"Rectangle {self.sid}")
        self.assertFalse(self.page._badge.isHidden())
        self.assertFalse(self.page._icon.pixmap().isNull())
        self.assertEqual(self._values(), {"X": 0, "Y": 0, "W": 4, "H": 2})
        # W and H wait on the scale operation the world does not have.
        self.assertFalse(self.page.fields["X"].edit.isReadOnly())
        self.assertTrue(self.page.fields["W"].edit.isReadOnly())

    def test_the_size_is_the_shape_own_box(self):
        # A quarter turn leaves the rectangle 4 by 2; only the axis-aligned
        # span it covers swaps, and that is not what the fields show.
        self.world.rotate_shape(self.sid, 0.5 * math.pi, 0.0, 0.0)
        self._select(self.sid)
        self.assertAlmostEqual(self.page.fields["W"].value(), 4)
        self.assertAlmostEqual(self.page.fields["H"].value(), 2)

    def test_picking_another_shape_refreshes_the_page(self):
        # A pick moves no geometry, so a poll watching the world would miss it.
        other = self.world.add_circle(10, 10, 3)
        self._select(self.sid)
        self._select(other)
        self.assertEqual(self._header(), f"Circle {other}")
        self.assertEqual(self._values(), {"X": 10, "Y": 10, "W": 6, "H": 6})

    def test_editing_x_moves_the_shape_and_repaints(self):
        self._select(self.sid)
        self._commit("X", "5")
        self.assertAlmostEqual(self.page.fields["X"].value(), 5)
        self.assertAlmostEqual(self.world.shape_obb(self.sid)[0], 3)
        self.assertEqual(self.canvas.repaints, 1)
        # The whole edit is a single undo step, and the fields follow it back.
        self.world.undo()
        self.page.refresh()
        self.assertAlmostEqual(self.page.fields["X"].value(), 0)

    def test_an_edit_that_changes_nothing_writes_nothing(self):
        # Text that parses to the value already shown, and text that does
        # not parse at all.
        self._select(self.sid)
        self._commit("Y", "0")
        self._commit("X", "not a number")
        self.assertEqual(self.canvas.repaints, 0)
        self.assertEqual(self.page.fields["X"].edit.text(), "0")

    def test_a_finished_edit_does_not_commit_again(self):
        # Enter leaves the field focused and its text as typed, so the value
        # can move on underneath it; the next focus change must not write the
        # old text back over that. Nothing maps a window here, so the focus
        # the sequence turns on is stubbed.
        field = self.page.fields["X"]
        field.edit.hasFocus = lambda: True
        self._select(self.sid)
        self._commit("X", "5")
        self.world.undo()
        self.page.refresh()
        field.edit.editingFinished.emit()
        self.assertEqual(self.canvas.repaints, 1)
        self.assertAlmostEqual(self.world.shape_obb(self.sid)[0], -2)

    def test_the_page_fits_the_designed_inspector(self):
        # Four editors asking for a line of text each would open the dock
        # wider than the inspector the design draws.
        self.assertLessEqual(self.page.sizeHint().width(),
                             _painter.PainterPanel._INSPECTOR_WIDTH)

    def test_leaving_a_field_untouched_moves_nothing(self):
        # The field shows a rounded value, so a focus change on a field nobody
        # typed in would otherwise commit the rounding error as a move.
        self.world.translate_shape(self.sid, 1.0 / 3.0, 0.0)
        self._select(self.sid)
        self.page.fields["X"].edit.editingFinished.emit()
        self.assertEqual(self.canvas.repaints, 0)
        self.assertAlmostEqual(self.page.fields["X"].value(), 1.0 / 3.0)

    def test_a_non_finite_edit_never_reaches_the_shape(self):
        # A shape moved to nan or inf is not brought back by undo.
        self._select(self.sid)
        for text in ("nan", "1e400"):
            with self.subTest(text=text):
                self._commit("X", text)
                self.assertEqual(self.canvas.repaints, 0)
                self.assertAlmostEqual(self.world.shape_obb(self.sid)[0], -2)

    def test_extreme_coordinates_never_break_the_shape(self):
        # A shape this far out still reports a finite center, and a move is
        # turned away whether the step itself overflows or only the far corner
        # does once a finite step lands on it.
        far = self.world.add_rectangle(4e307, -1, 6e307, 1)
        self._select(far)
        self.assertTrue(math.isfinite(self.page.fields["X"].value()))
        for text in ("1.7e308", "-1.6e308"):
            with self.subTest(text=text):
                self._commit("X", text)
                self.assertEqual(self.canvas.repaints, 0)
                self.assertTrue(all(math.isfinite(value) for value
                                    in self.world.shape_obb(far)))

    def test_a_dead_selection_leaves_the_page_empty(self):
        # The canvas keeps the id it stored, and a query on a dead one throws.
        self._select(self.sid)
        self.world.remove_shape(self.sid)
        self.page.refresh()
        self.assertEqual(self._header(),
                         _design.DesignPage.EMPTY_TEXT)
        self.assertFalse(self.page.fields["X"].isEnabled())

    def test_closing_the_canvas_clears_the_page(self):
        # The page asks for the canvas again on every read.
        self._select(self.sid)
        self.canvas = None
        self.page.refresh()
        self.assertEqual(self._header(),
                         _design.DesignPage.EMPTY_TEXT)

    def test_sections_the_model_cannot_fill_are_greyed_out(self):
        self.assertEqual(list(self.page.placeholders),
                         ["Stroke", "Fill", "Grid & snap", "Layers"])
        for title, section in self.page.placeholders.items():
            with self.subTest(title=title):
                self.assertFalse(section.isEnabled())
                self.assertIn("needs", section.toolTip())

    def test_the_page_restyles_with_the_palette(self):
        # A color captured once would survive a theme switch as is.
        self._select(self.sid)
        before = self.page.styleSheet()
        icon = self.page._icon.pixmap().toImage()
        palette = QtGui.QPalette(self.page.palette())
        palette.setColor(QtGui.QPalette.Window, QtGui.QColor("black"))
        palette.setColor(QtGui.QPalette.WindowText, QtGui.QColor("white"))
        palette.setColor(QtGui.QPalette.Highlight, QtGui.QColor("red"))
        self.page.setPalette(palette)
        self.assertNotEqual(self.page.styleSheet(), before)
        self.assertNotEqual(self.page._icon.pixmap().toImage(), icon)


@unittest.skipIf(NO_LIVE_WINDOW or SKIP_PILOT_WIDGETS or not solvcon.HAS_PILOT,
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
