# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import tempfile
import unittest

import numpy as np

import solvcon

try:
    from solvcon import pilot
    from PySide6.QtGui import QImage
except ImportError:
    pilot = None

# The offscreen platform cannot back a live window; neither can the headless
# Windows CI runner, whose WARP software rasterizer faults on one.
NO_LIVE_WINDOW = ((os.getenv('QT_QPA_PLATFORM') or '').startswith('offscreen')
                  or ('nt' == os.name and bool(os.getenv('GITHUB_ACTIONS'))))


def _load_rgba(path):
    """Load a PNG into an (height, width, 4) RGBA uint8 array of pixels.

    The pixels are physical (device) pixels: a HiDPI capture is larger than
    the widget's logical size by the device-pixel ratio.
    """
    img = QImage(path)
    assert not img.isNull(), "QImage failed to load %s" % path
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    width, height = img.width(), img.height()
    # bytesPerLine may pad the scanline past width * 4, so reshape on the
    # full stride (in pixels) and then slice the padding off.
    stride = img.bytesPerLine() // 4
    arr = np.frombuffer(bytes(img.constBits()), dtype='uint8')
    return arr.reshape(height, stride, 4)[:, :width, :].copy()


def _send_mouse(widget, kind, x, y):
    """Post a synthetic left-button mouse event to ``widget``.
    """
    from PySide6 import QtCore, QtGui, QtWidgets
    kinds = {
        'press': (QtCore.QEvent.Type.MouseButtonPress,
                  QtCore.Qt.LeftButton, QtCore.Qt.LeftButton),
        'move': (QtCore.QEvent.Type.MouseMove,
                 QtCore.Qt.NoButton, QtCore.Qt.LeftButton),
        'release': (QtCore.QEvent.Type.MouseButtonRelease,
                    QtCore.Qt.LeftButton, QtCore.Qt.NoButton),
    }
    etype, button, buttons = kinds[kind]
    pos = QtCore.QPointF(x, y)
    glob = widget.mapToGlobal(pos.toPoint())
    event = QtGui.QMouseEvent(etype, pos, QtCore.QPointF(glob), button,
                              buttons, QtCore.Qt.NoModifier)
    QtWidgets.QApplication.sendEvent(widget, event)


@unittest.skipIf(NO_LIVE_WINDOW or not solvcon.HAS_PILOT,
                 "live-GUI interaction needs a real window surface")
class R2DWidgetSelectToolTC(unittest.TestCase):
    """Drive the select tool through select, move, rotate, and deselect."""

    @classmethod
    def setUpClass(cls):
        cls.mgr = pilot.RManager.instance.setUp()

    def setUp(self):
        from PySide6 import QtWidgets
        self.widget = self.mgr.add2DWidget()
        self.widget.setDrawTool("select")
        self.world = solvcon.WorldFp64()
        # A rectangle centered on the origin.
        self.sid = self.world.add_rectangle(-2, -1, 2, 1)
        self.widget.updateWorld(self.world)
        v = solvcon.ViewTransform2dFp64()
        v.pan(100.0, 100.0)
        v.zoom = 20.0
        # Set the view before showing so the resize auto-centering, which a
        # well-formed transform disables, leaves the mapping deterministic.
        self.widget.setViewTransform(v)
        self.mgr.show()
        self.sub = self.mgr.mdiArea.subWindowList()[-1]
        self.sub.show()
        self.mgr.mdiArea.setActiveSubWindow(self.sub)
        # The PySide6 widget wraps the same C++ object the handle above does.
        self.target = self.sub.widget()
        QtWidgets.QApplication.processEvents()

    def test_selecting_from_outside_ends_an_active_drag(self):
        """A caller that moves the selection mid-gesture leaves the canvas
        holding a drag on a shape it no longer selects; the gesture ends the
        way a release ends it."""
        from PySide6 import QtCore
        other = self.world.add_circle(20, 20, 2)
        _send_mouse(self.target, 'press', 100, 100)
        self.assertEqual(self.widget.selectedShape, self.sid)
        held = self.world.segment(0).x0

        self.widget.selectedShape = other
        _send_mouse(self.target, 'move', 160, 100)
        # The shape the gesture had hold of stays where the drag left it, and
        # the cursor the drag put on comes off.
        self.assertAlmostEqual(self.world.segment(0).x0, held)
        self.assertEqual(self.target.cursor().shape(), QtCore.Qt.ArrowCursor)
        _send_mouse(self.target, 'release', 160, 100)
        # The undo bracket the press opened closed with the gesture, so the
        # move is one step and the world is not left mid-operation.
        self.assertTrue(self.world.can_undo)

    def test_select_move_rotate_run_through(self):
        orig_x0 = self.world.segment(0).x0
        # Press on the shape to select it, then drag to move it.
        _send_mouse(self.target, 'press', 100, 100)
        _send_mouse(self.target, 'move', 140, 100)
        _send_mouse(self.target, 'release', 140, 100)
        self.assertEqual(self.widget.selectedShape, self.sid)
        moved_x0 = self.world.segment(0).x0
        self.assertNotAlmostEqual(moved_x0, orig_x0)
        # The whole move drag is a single undo step: one undo restores the
        # original position, and one redo replays the move.
        self.world.undo()
        self.assertAlmostEqual(self.world.segment(0).x0, orig_x0)
        self.world.redo()
        self.assertAlmostEqual(self.world.segment(0).x0, moved_x0)
        # Grab the rotate handle and swing it.
        hx, hy = self.widget.rotateHandleScreen
        _send_mouse(self.target, 'press', hx, hy)
        _send_mouse(self.target, 'move', hx + 30, hy + 30)
        _send_mouse(self.target, 'release', hx + 30, hy + 30)
        # The rotate drag is one undo step too: undo returns to the moved (not
        # the original) state, so the rotation alone is reverted.
        self.world.undo()
        self.assertAlmostEqual(self.world.segment(0).x0, moved_x0)
        # Switching tools drops the selection.
        self.widget.setDrawTool("circle")
        self.assertEqual(self.widget.selectedShape, -1)


@unittest.skipIf(NO_LIVE_WINDOW or not solvcon.HAS_PILOT,
                 "theme switching needs a real window surface")
class R2DWidgetThemeTC(unittest.TestCase):
    """The painted frame follows the theme, not just the palette the widget
    reports. Whether a canvas is themed can only be settled on the pixels.
    """

    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        self.widget = self.mgr.add2DWidget()

    def tearDown(self):
        # The manager is a shared singleton, so restore the default mode to
        # keep the tests independent of the order they run in.
        self.mgr.set_theme("system")

    def _dominant_color(self):
        """The most common color in an offscreen render of an empty canvas.

        On a world with no geometry the backdrop covers nearly every pixel,
        with only the grid, the axes, and the origin marker over it, so the
        winner is the backdrop the renderer filled with. Reading the most
        common color rather than a fixed pixel keeps the test off the grid
        lines, whose spacing shifts with the view.
        """
        self.widget.updateWorld(solvcon.WorldFp64())
        self.widget.resetView()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "theme.png")
            self.assertTrue(
                self.widget.saveImage(path, pilot.Overlay2dOptions()))
            pixels = _load_rgba(path)[:, :, :3].astype('uint32')
        # Pack each pixel into one integer first: np.unique over a 1-D array
        # is a plain sort, while over rows (axis=0) it takes a much slower
        # void-dtype lexsort.
        packed = ((pixels[:, :, 0] << 16)
                  | (pixels[:, :, 1] << 8) | pixels[:, :, 2]).ravel()
        values, counts = np.unique(packed, return_counts=True)
        winner = int(values[counts.argmax()])
        return (winner >> 16, (winner >> 8) & 0xff, winner & 0xff)

    def test_the_rendered_backdrop_is_the_themed_one(self):
        for mode in ("light", "dark"):
            with self.subTest(mode=mode):
                self.mgr.set_theme(mode)
                self.assertEqual(self._dominant_color(),
                                 self.widget.canvasPalette["background"])


@unittest.skipIf(NO_LIVE_WINDOW or not solvcon.HAS_PILOT,
                 "live-GUI interaction needs a real window surface")
class PainterToolboxTC(unittest.TestCase):
    """Run-through coverage of the Painter toolbox and the 'Create blank 2D
    canvas' flow.

    The painter is still a prototype, so these stay at the run-through
    level -- open the flow and drive it without crashing -- and leave
    detailed behavioral assertions for future work. They drive live widgets
    (docks, focus changes, mouse gestures), so they are skipped under the
    offscreen Qt platform like the other interactive pilot tests; the
    draw-tool API itself is covered headlessly by
    R2DWidgetWorldTC.test_draw_tool_round_trip.
    """

    @classmethod
    def setUpClass(cls):
        cls.mgr = pilot.RManager.instance.setUp()

    def test_create_blank_canvas_shows_toolbox(self):
        """'Create blank 2D canvas' opens an empty, focused canvas on the
        select tool and brings up the Painter toolbox.
        """
        from solvcon.pilot.canvas import _canvas_gui
        from solvcon.pilot import painter as _painter
        painter = _painter.Painter(mgr=self.mgr)
        canvas = _canvas_gui.Canvas(mgr=self.mgr, painter=painter)
        widget = canvas._create_blank_2d_canvas()
        self.assertIsNotNone(painter._dock)
        self.assertEqual(widget.drawTool, "select")

    def test_draw_across_blank_canvases(self):
        """The PR's manual test: create two blank canvases and rubber-band a
        circle onto each in turn, exercising tool routing and the 2D path's
        handling of multiple canvases and rapid focus changes. Surviving the
        gestures without a crash is the assertion.
        """
        import gc
        from PySide6 import QtWidgets
        from solvcon.pilot.canvas import _canvas_gui
        from solvcon.pilot import painter as _painter
        painter = _painter.Painter(mgr=self.mgr)
        canvas = _canvas_gui.Canvas(mgr=self.mgr, painter=painter)
        first = canvas._create_blank_2d_canvas()
        second = canvas._create_blank_2d_canvas()
        del first, second
        gc.collect()
        self.mgr.show()
        area = self.mgr.mdiArea
        subs = list(area.subWindowList())
        for sub in subs:
            sub.show()
        QtWidgets.QApplication.processEvents()
        self.mgr.setDrawTool("circle")
        # Select each canvas in turn and rubber-band a circle onto it.
        for _ in range(3):
            for sub in subs:
                area.setActiveSubWindow(sub)
                QtWidgets.QApplication.processEvents()
                target = sub.widget()
                _send_mouse(target, 'press', 40, 40)
                _send_mouse(target, 'move', 110, 100)
                _send_mouse(target, 'release', 110, 100)
                QtWidgets.QApplication.processEvents()
        self.assertIn(self.mgr.currentR2DWidget().drawTool,
                      ("select", "circle"))

    def test_press_then_repaint_with_circle_tool_does_not_crash(self):
        """The zero-radius preview used to crash because the painter's pen
        was uninitialized until the first paint event, so pressing without
        moving then forcing a repaint triggered a null pointer dereference.
        """
        from PySide6 import QtWidgets
        from solvcon.pilot.canvas import _canvas_gui
        from solvcon.pilot import painter as _painter
        painter = _painter.Painter(mgr=self.mgr)
        canvas = _canvas_gui.Canvas(mgr=self.mgr, painter=painter)
        canvas._create_blank_2d_canvas()
        self.mgr.show()
        sub = self.mgr.mdiArea.subWindowList()[-1]
        sub.show()
        self.mgr.setDrawTool("circle")
        target = sub.widget()
        QtWidgets.QApplication.processEvents()
        # Press without moving, then force the synchronous repaint the
        # zero-radius preview used to crash on.
        _send_mouse(target, 'press', 60, 60)
        target.repaint()
        QtWidgets.QApplication.processEvents()
        _send_mouse(target, 'release', 60, 60)
        # Surviving the repaint is the assertion; the canvas still answers.
        self.assertEqual(self.mgr.currentR2DWidget().drawTool, "circle")

    def test_each_shape_tool_commits_expected_type(self):
        """Each shape tool maps one rubber-band gesture onto the matching
        World primitive: drawing grows the canvas world by a single shape of
        the expected type. This covers the 2-point -> add_* mapping in C++
        that the headless round-trip test cannot reach.
        """
        from PySide6 import QtWidgets
        from solvcon.pilot.canvas import _canvas_gui
        from solvcon.pilot import painter as _painter
        painter = _painter.Painter(mgr=self.mgr)
        canvas = _canvas_gui.Canvas(mgr=self.mgr, painter=painter)
        canvas._create_blank_2d_canvas()
        world = canvas._blank_worlds[-1]
        self.mgr.show()
        sub = self.mgr.mdiArea.subWindowList()[-1]
        sub.show()
        self.mgr.mdiArea.setActiveSubWindow(sub)
        target = sub.widget()
        QtWidgets.QApplication.processEvents()
        # The shape tools, paired with the shape type each one commits.
        shapes = [("line", "line"), ("triangle", "triangle"),
                  ("rectangle", "rectangle"), ("ellipse", "ellipse"),
                  ("circle", "circle")]
        for index, (tool, shape) in enumerate(shapes):
            self.mgr.setDrawTool(tool)
            _send_mouse(target, 'press', 40, 40)
            _send_mouse(target, 'move', 120, 100)
            _send_mouse(target, 'release', 120, 100)
            QtWidgets.QApplication.processEvents()
            self.assertEqual(world.nshape, index + 1)
            self.assertEqual(world.shape_type_of(index), shape)


if __name__ == '__main__':
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
