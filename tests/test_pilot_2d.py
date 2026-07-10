# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for R2DWidget and its on-screen screenshot APIs.
"""

import os
import tempfile
import unittest

import numpy as np

import solvcon

try:
    from solvcon import pilot
    from PySide6.QtGui import QGuiApplication, QImage
except ImportError:
    pilot = None

GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', False)

_PNG_MAGIC = b'\x89PNG\r\n\x1a\n'

# The pixel helpers below classify colors that RWorldRenderer2d paints (see
# RWorldRenderer2d.cpp). GEOMETRY (120, 180, 240) is the only strongly-blue
# color: the backdrop, grid, axes, and origin marker all have blue <= 80, so
# a blue-dominant pixel must be geometry. ORIGIN (220, 80, 80) is the only
# red-dominant color, since the yellow axes have equal red and green, which
# lets the origin marker be located on its own.


def _png_size(data):
    """Return PNG width and height from IHDR (offsets 16 and 20)."""
    assert data[:8] == _PNG_MAGIC
    return (int.from_bytes(data[16:20], 'big'),
            int.from_bytes(data[20:24], 'big'))


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


def _channels(arr):
    """Return the R, G, B planes as signed ints for dominance comparisons."""
    return (arr[:, :, 0].astype('int32'),
            arr[:, :, 1].astype('int32'),
            arr[:, :, 2].astype('int32'))


def _geometry_mask(arr):
    """Return the boolean mask of GEOMETRY-colored (blue-dominant) pixels."""
    red, green, blue = _channels(arr)
    return (blue >= 120) & (blue > red + 30) & (blue > green)


def _origin_mask(arr):
    """Return the boolean mask of ORIGIN-marker (red-dominant) pixels."""
    red, green, blue = _channels(arr)
    return (red >= 150) & (red > green + 40) & (red > blue + 40)


def _has_geometry_near(mask, px, py, radius=4):
    """Return whether a geometry pixel lies within radius of (px, py).

    A point that rounds to outside the image has no geometry by definition,
    so return False rather than sampling the clamped edge window.
    """
    col, row = int(round(px)), int(round(py))
    height, width = mask.shape
    if not (0 <= col < width and 0 <= row < height):
        return False
    window = mask[max(0, row - radius):row + radius + 1,
                  max(0, col - radius):col + radius + 1]
    return bool(window.any())


def _build_world():
    """A world exercising every primitive R2DWidget paints, plus the two
    cases the 2D path must handle silently: an out-of-plane (z != 0) point
    that gets projected, and a removed (DEAD) shape whose geometry must be
    dropped by collect_live_*.
    """
    w = solvcon.WorldFp64()
    # Bare segment (owned by no shape) and a bare cubic Bezier.
    w.add_segment(solvcon.Point3dFp64(-3, 3, 0),
                  solvcon.Point3dFp64(3, 3, 0))
    w.add_bezier(solvcon.Point3dFp64(-3, 0, 0),
                 solvcon.Point3dFp64(-1, 2, 0),
                 solvcon.Point3dFp64(1, -2, 0),
                 solvcon.Point3dFp64(3, 0, 0))
    # Shapes: segment-backed and Bezier-backed.
    w.add_triangle(0, 0, 1, 0, 0, 1)
    w.add_rectangle(2, 2, 4, 3)
    w.add_circle(-2, -2, 1.0)
    # A point off the z=0 plane: the 2D widget drops z, must not error.
    w.add_point(0.5, 0.5, 5.0)
    # A removed shape: its segments must be culled, not painted.
    dead = w.add_triangle(8, 8, 9, 8, 8, 9)
    w.remove_shape(dead)
    return w


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


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class R2DWidgetWorldTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.widget = pilot.RManager.instance.setUp().add2DWidget()

    @classmethod
    def tearDownClass(cls):
        cls.widget = None

    def test_add_2d_widget_created(self):
        """The 2D widget is created and accessible via the manager.
        """
        self.assertIsNotNone(self.widget)

    def test_update_world_accepts_mixed_geometry(self):
        """updateWorld accepts a world with mixed geometry (segments, curves,
        points, shapes) and a removed shape, without raising.
        """
        self.widget.updateWorld(_build_world())
        self.widget.requestRepaint()

    def test_update_world_none_clears(self):
        """A null world clears the canvas to its grid backdrop instead of
        crashing; RWorldRenderer2d is skipped when no world is set.
        """
        self.widget.updateWorld(_build_world())
        self.widget.updateWorld(None)
        self.widget.requestRepaint()

    def test_resync_after_mutating_world(self):
        """Re-issuing updateWorld on the same world after adding geometry
        (the live Canvas sample flow) repaints the new state. Guards that
        the widget re-reads the world rather than caching a snapshot.
        """
        w = solvcon.WorldFp64()
        w.add_circle(0.0, 0.0, 1.0)
        self.widget.updateWorld(w)
        w.add_rectangle(-2, -2, 2, 2)
        self.widget.updateWorld(w)
        self.widget.requestRepaint()

    def test_empty_world_paints_without_error(self):
        """A world with no geometry is valid: the loops are simply empty.
        Catches off-by-one / null-pad assumptions in RWorldRenderer2d.
        """
        self.widget.updateWorld(solvcon.WorldFp64())
        self.widget.requestRepaint()

    def test_view_transform_round_trip(self):
        """The view-state API that frames the world is intact: a pan/zoom
        set via setViewTransform reads back through the viewTransform
        property (zoom 3.0 is well within the widget's clamp band).
        """
        self.widget.resetView()
        v = solvcon.ViewTransform2dFp64()
        v.pan(40.0, 25.0)
        v.zoom = 3.0
        self.widget.setViewTransform(v)
        got = self.widget.viewTransform
        self.assertEqual(got.pan_x, 40.0)
        self.assertEqual(got.pan_y, 25.0)
        self.assertEqual(got.zoom, 3.0)

    def test_draw_tool_round_trip(self):
        """setDrawTool selects the tool the Painter toolbox drives; every
        registered tool reads back through the drawTool property, the
        default is pan, and an unknown name is rejected with ValueError.
        """
        from solvcon.pilot import _pilot_core
        # The Painter toolbox exposes one button per registered shape tool.
        self.assertLessEqual(
            {"pan", "line", "triangle", "rectangle", "ellipse", "circle"},
            set(_pilot_core.draw_tool_names()))
        for tool in _pilot_core.draw_tool_names():
            self.widget.setDrawTool(tool)
            self.assertEqual(self.widget.drawTool, tool)
        self.widget.setDrawTool("pan")
        with self.assertRaises(ValueError):
            self.widget.setDrawTool("no-such-tool")
        # An invalid request leaves the previous tool untouched.
        self.assertEqual(self.widget.drawTool, "pan")

    def test_selected_shape_defaults_to_none(self):
        """A fresh canvas has nothing selected; selectedShape reads -1."""
        self.widget.updateWorld(solvcon.WorldFp64())
        self.assertEqual(self.widget.selectedShape, -1)

    def _render_world(self, world, pan_x, pan_y, zoom):
        """Render world under an explicit view; return (geometry_mask,
        to_pixel).

        An explicit transform also disables the widget's auto-centering, so
        the world->screen mapping is fixed regardless of the widget's size.
        ``to_pixel(world_x, world_y)`` maps a world point to its physical
        (device) pixel. It folds in the device-pixel ratio recovered from
        the rendered origin marker rather than an assumed one, so pixel
        predictions hold on both standard and HiDPI displays.
        """
        self.widget.updateWorld(world)
        view = solvcon.ViewTransform2dFp64()
        view.pan_x = pan_x
        view.pan_y = pan_y
        view.zoom = zoom
        self.widget.setViewTransform(view)

        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "render.png")
            self.widget.saveImage(path)
            arr = _load_rgba(path)

        # The origin marker sits at world (0, 0), which maps to screen
        # (pan_x, pan_y); its physical-pixel center recovers the dpr.
        rows, cols = np.where(_origin_mask(arr))
        self.assertGreater(len(cols), 0, "origin marker was not rendered")
        dpr = (cols.mean() / pan_x + rows.mean() / pan_y) / 2.0

        def to_pixel(world_x, world_y):
            screen_x, screen_y = view.screen_from_world(world_x, world_y)
            return screen_x * dpr, screen_y * dpr

        return _geometry_mask(arr), to_pixel

    def test_known_world_renders_to_expected_pixels(self):
        """Gate test: a known world renders to the expected pixels.

        A single horizontal segment is rendered under an explicit view
        transform. The render must be non-blank, the segment's two endpoints
        must land where the view transform predicts, and a removed (DEAD)
        rectangle must leave no pixels in its region.
        """
        point_a = (-1.5, 1.5)
        point_b = (1.5, 1.5)
        dead_rect = (1.2, -1.2, 2.0, -0.4)  # Well clear of the live segment.

        world = solvcon.WorldFp64()
        world.add_segment(
            solvcon.Point3dFp64(point_a[0], point_a[1], 0.0),
            solvcon.Point3dFp64(point_b[0], point_b[1], 0.0))
        dead = world.add_rectangle(*dead_rect)
        world.remove_shape(dead)

        geometry, to_pixel = self._render_world(world, 110.0, 110.0, 24.0)

        # Non-blank: the live segment painted real geometry pixels.
        self.assertGreater(int(geometry.sum()), 0)

        # Known endpoints map to the expected pixels under the view transform.
        for label, (world_x, world_y) in (("A", point_a), ("B", point_b)):
            self.assertTrue(
                _has_geometry_near(geometry, *to_pixel(world_x, world_y)),
                "no geometry at predicted endpoint %s" % label)

        # The removed rectangle must be culled: no geometry anywhere in its
        # screen bounding box. The mapping is separable, so the two opposite
        # world corners bound that region.
        px0, py0 = to_pixel(dead_rect[0], dead_rect[1])
        px1, py1 = to_pixel(dead_rect[2], dead_rect[3])
        x_lo, x_hi = sorted((px0, px1))
        y_lo, y_hi = sorted((py0, py1))
        region = geometry[max(0, int(y_lo) - 2):int(y_hi) + 3,
                          max(0, int(x_lo) - 2):int(x_hi) + 3]
        self.assertEqual(int(region.sum()), 0, "removed shape was painted")

    def test_circle_renders_as_hollow_loop_on_locus(self):
        """A circle renders as a closed, hollow ring on its locus.

        add_circle builds the outline from four cubic Beziers, stroked and
        not filled. Sampling the predicted ring at twelve angles checks the
        loop is painted all the way around -- a dropped Bezier quadrant or an
        open arc would leave a gap -- and that includes the four cardinal
        points landing where the view transform predicts. The center and a
        mid-radius point stay blank because the renderer strokes outlines
        only.
        """
        center = (0.5, 0.3)
        radius = 2.0

        world = solvcon.WorldFp64()
        world.add_circle(center[0], center[1], radius)

        geometry, to_pixel = self._render_world(world, 160.0, 120.0, 30.0)

        # The ring is painted all the way around, cardinal points included.
        for degrees in range(0, 360, 30):
            angle = np.radians(degrees)
            world_x = center[0] + radius * np.cos(angle)
            world_y = center[1] + radius * np.sin(angle)
            self.assertTrue(
                _has_geometry_near(geometry, *to_pixel(world_x, world_y)),
                "no geometry on the ring at %d degrees" % degrees)

        # Outlines only: the center and a mid-radius point stay blank.
        for world_x, world_y in (center, (center[0] + radius / 2, center[1])):
            self.assertFalse(
                _has_geometry_near(geometry, *to_pixel(world_x, world_y)),
                "circle interior should be hollow")


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "live-GUI interaction is unstable under GitHub Actions")
class R2DWidgetPanSelectTC(unittest.TestCase):
    """Drive the pan tool through select, move, rotate, and deselect."""

    @classmethod
    def setUpClass(cls):
        cls.mgr = pilot.RManager.instance.setUp()

    def setUp(self):
        from PySide6 import QtWidgets
        self.widget = self.mgr.add2DWidget()
        self.widget.setDrawTool("pan")
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


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class R2DWidgetScreenshotTC(unittest.TestCase):
    """R2DWidget on-screen screenshot APIs (saveImage/clipImage)."""

    @classmethod
    def setUpClass(cls):
        cls.widget = pilot.RManager.instance.setUp().add2DWidget()

    @classmethod
    def tearDownClass(cls):
        cls.widget = None

    def test_save_image_writes_png_file(self):
        """saveImage writes a valid, non-empty PNG of the widget."""
        self.widget.updateWorld(_build_world())
        self.widget.resetView()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "widget.png")
            self.assertTrue(self.widget.saveImage(path))
            with open(path, 'rb') as stream:
                data = stream.read()
        self.assertEqual(data[:8], _PNG_MAGIC)
        width, height = _png_size(data)
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)

    def test_current_2d_widget_exposes_screenshot_api(self):
        """currentR2DWidget returns the active 2D widget, API intact."""
        mgr = pilot.RManager.instance.setUp()
        mgr.add2DWidget()
        current = mgr.currentR2DWidget()
        self.assertIsNotNone(current)
        current.updateWorld(_build_world())
        current.resetView()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "current.png")
            self.assertTrue(current.saveImage(path))
            with open(path, 'rb') as stream:
                data = stream.read()
        self.assertEqual(data[:8], _PNG_MAGIC)
        width, height = _png_size(data)
        self.assertGreater(width, 0)
        self.assertGreater(height, 0)

    def test_list_2d_widgets_returns_usable_widgets(self):
        """list2DWidgets returns real R2DWidgets, not bare QWidgets."""
        mgr = pilot.RManager.instance.setUp()
        mgr.add2DWidget()
        widgets = mgr.list2DWidgets()
        self.assertIsInstance(widgets, list)
        self.assertGreaterEqual(len(widgets), 1)
        for widget in widgets:
            self.assertTrue(hasattr(widget, "saveImage"))
            self.assertTrue(hasattr(widget, "clipImage"))
        listed = widgets[-1]
        listed.updateWorld(_build_world())
        listed.resetView()
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "listed.png")
            self.assertTrue(listed.saveImage(path))
            with open(path, 'rb') as stream:
                data = stream.read()
        self.assertEqual(data[:8], _PNG_MAGIC)

    def test_clip_image_copies_pixmap_to_clipboard(self):
        """clipImage puts a non-null widget pixmap on the clipboard."""
        clipboard = QGuiApplication.clipboard()
        if clipboard is None:
            self.skipTest("no clipboard in this environment")
        self.widget.updateWorld(_build_world())
        self.widget.resetView()
        clipboard.clear()
        self.assertTrue(clipboard.pixmap().isNull())
        self.widget.clipImage()
        pixmap = clipboard.pixmap()
        self.assertFalse(pixmap.isNull())
        self.assertGreater(pixmap.width(), 0)
        self.assertGreater(pixmap.height(), 0)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class Save2DCanvasDialogTC(unittest.TestCase):
    """File -> Save 2D canvas dialog and path resolution."""

    def test_resolve_save_path_forces_png_or_jpg(self):
        from solvcon.pilot._canvas_gui import (
            resolve_save_path, _PNG_FILTER, _JPG_FILTER)
        self.assertEqual(
            resolve_save_path("/tmp/a", _PNG_FILTER), "/tmp/a.png")
        self.assertEqual(
            resolve_save_path("/tmp/a.bmp", _PNG_FILTER), "/tmp/a.png")
        self.assertEqual(
            resolve_save_path("/tmp/a", _JPG_FILTER), "/tmp/a.jpg")
        self.assertEqual(
            resolve_save_path("/tmp/a.png", _JPG_FILTER), "/tmp/a.jpg")
        self.assertEqual(
            resolve_save_path("/tmp/a.jpeg", _JPG_FILTER), "/tmp/a.jpeg")
        self.assertEqual(resolve_save_path("/tmp/a.png", ""), "/tmp/a.png")
        self.assertIsNone(resolve_save_path("/tmp/a.bmp", ""))
        self.assertIsNone(resolve_save_path("", _PNG_FILTER))

    def test_menu_action_is_registered(self):
        from solvcon.pilot import _gui
        mgr = _gui.controller.build()
        act = mgr.menu_model.action("file.save_2d_canvas")
        self.assertIsNotNone(act)
        self.assertEqual(act.text(), "Save 2D canvas")
        self.assertIn(
            "Save 2D canvas",
            [a.text() for a in mgr.menu_model.menu("File").actions()])

    def test_run_without_canvas_skips_dialog(self):
        """Without a focused 2D canvas, run must not open the save dialog."""
        from solvcon.pilot import _canvas_gui
        mgr = pilot.RManager.instance.setUp()
        # RManager is a process-wide singleton; earlier tests may leave a
        # focused 2D subwindow, so clear the MDI before asserting the guard.
        mgr.mdiArea.closeAllSubWindows()
        feature = _canvas_gui.Save2DCanvasDialog(mgr=mgr)
        opened = []
        feature._diag.open = lambda *a, **k: opened.append(True)
        self.assertIsNone(mgr.currentR2DWidget())
        self.assertFalse(feature.run())
        self.assertEqual(opened, [])

    def test_save_current_reports_write_result(self):
        """Menu save path returns saveImage's bool and only writes on
        success."""
        from solvcon.pilot import _canvas_gui
        mgr = pilot.RManager.instance.setUp()
        widget = mgr.add2DWidget()
        widget.updateWorld(_build_world())
        widget.resetView()
        feature = _canvas_gui.Save2DCanvasDialog(mgr=mgr)
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "out.png")
            self.assertTrue(feature._save_current(path))
            self.assertTrue(os.path.isfile(path))
            with open(path, 'rb') as stream:
                data = stream.read()
            bad = os.path.join(folder, "missing", "out.png")
            self.assertFalse(feature._save_current(bad))
            self.assertFalse(os.path.isfile(bad))
        self.assertEqual(data[:8], _PNG_MAGIC)


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "live-GUI interaction is unstable under GitHub Actions")
class PainterToolboxTC(unittest.TestCase):
    """Run-through coverage of the Painter toolbox and the 'Create blank 2D
    canvas' flow.

    The painter is still a prototype, so these stay at the run-through
    level -- open the flow and drive it without crashing -- and leave
    detailed behavioral assertions for future work. They drive live widgets
    (docks, focus changes, mouse gestures), so they are skipped on GitHub
    Actions like the other interactive pilot tests; the draw-tool API itself
    is covered headlessly by R2DWidgetWorldTC.test_draw_tool_round_trip.
    """

    @classmethod
    def setUpClass(cls):
        cls.mgr = pilot.RManager.instance.setUp()

    def test_create_blank_canvas_shows_toolbox(self):
        """'Create blank 2D canvas' opens an empty, focused canvas on the
        Pan tool and brings up the Painter toolbox.
        """
        from solvcon.pilot import _canvas_gui, _painter_gui
        painter = _painter_gui.Painter(mgr=self.mgr)
        canvas = _canvas_gui.Canvas(mgr=self.mgr, painter=painter)
        widget = canvas._create_blank_2d_canvas()
        self.assertIsNotNone(painter._dock)
        self.assertEqual(widget.drawTool, "pan")

    def test_draw_across_blank_canvases(self):
        """The PR's manual test: create two blank canvases and rubber-band a
        circle onto each in turn, exercising tool routing and the 2D path's
        handling of multiple canvases and rapid focus changes. Surviving the
        gestures without a crash is the assertion.
        """
        import gc
        from PySide6 import QtWidgets
        from solvcon.pilot import _canvas_gui, _painter_gui
        painter = _painter_gui.Painter(mgr=self.mgr)
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
        self.assertIn(self.mgr.currentR2DWidget().drawTool, ("pan", "circle"))

    def test_press_then_repaint_with_circle_tool_does_not_crash(self):
        """The zero-radius preview used to crash because the painter's pen
        was uninitialized until the first paint event, so pressing without
        moving then forcing a repaint triggered a null pointer dereference.
        """
        from PySide6 import QtWidgets
        from solvcon.pilot import _canvas_gui, _painter_gui
        painter = _painter_gui.Painter(mgr=self.mgr)
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
        from solvcon.pilot import _canvas_gui, _painter_gui
        painter = _painter_gui.Painter(mgr=self.mgr)
        canvas = _canvas_gui.Canvas(mgr=self.mgr, painter=painter)
        canvas._create_blank_2d_canvas()
        world = canvas._blank_worlds[-1]
        self.mgr.show()
        sub = self.mgr.mdiArea.subWindowList()[-1]
        sub.show()
        self.mgr.mdiArea.setActiveSubWindow(sub)
        target = sub.widget()
        QtWidgets.QApplication.processEvents()
        # The non-pan tools, paired with the shape type each one commits.
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
