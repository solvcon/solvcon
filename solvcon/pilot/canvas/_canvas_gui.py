# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Canvas GUI feature for pilot.
"""

import os

from PySide6 import QtCore, QtWidgets

from ... import core
from ...plot import curve, plane_layer

from .. import _gui_common

__all__ = [
    'Canvas',
    'Save2DCanvasDialog',
]

_PNG_FILTER = "PNG image (*.png)"
_JPG_FILTER = "JPEG image (*.jpg *.jpeg)"
_ALLOWED_EXTS = (".png", ".jpg", ".jpeg")


def resolve_save_path(path, name_filter):
    """Force ``path`` to a png/jpg extension from the chosen name filter.

    Returns the resolved path, or ``None`` when the result is not png/jpg.
    """
    if not path:
        return None
    root, ext = os.path.splitext(path)
    ext = ext.lower()
    filt = (name_filter or "").lower()
    if "png" in filt:
        if ext != ".png":
            path = root + ".png"
    elif "jpg" in filt or "jpeg" in filt:
        if ext not in (".jpg", ".jpeg"):
            path = root + ".jpg"
    elif ext not in _ALLOWED_EXTS:
        return None
    return path


class Save2DCanvasDialog(_gui_common.PilotFeature):
    """
    File-menu action that saves the focused 2D canvas via ``saveImage``.

    The save dialog carries an "Include labels" switch with a normal/advanced
    selector and an independent "Include coordinates" switch, so the exported
    image can bake in the annotation overlay independent of what the canvas
    currently shows on screen. The controls need custom widgets in the dialog,
    so it runs non-native.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        # Build the dialog in run(), not here: parenting a QFileDialog to the
        # main window before launch forces native window creation that aborts
        # the Windows Debug CRT (exit 0xC0000409). Deferring keeps the parent
        # (dialog stays window-modal) with no Qt window built before launch.
        self._diag = None

    def _ensure_dialog(self):
        """Build the save dialog and its label controls on first use."""
        if self._diag is not None:
            return
        self._diag = QtWidgets.QFileDialog(self._mainWindow)
        self._diag.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        self._diag.setAcceptMode(QtWidgets.QFileDialog.AcceptSave)
        self._diag.setFileMode(QtWidgets.QFileDialog.AnyFile)
        self._diag.setNameFilters([_PNG_FILTER, _JPG_FILTER])
        self._diag.setDefaultSuffix("png")
        self._diag.setWindowTitle("Save 2D canvas")
        self._diag.filterSelected.connect(self._on_filter_selected)
        self._build_label_controls()

    def _build_label_controls(self):
        """Add the label switch, normal/advanced selector, and the independent
        coordinate switch to the dialog."""
        self._labels_check = QtWidgets.QCheckBox("Include labels")
        self._normal_radio = QtWidgets.QRadioButton("normal")
        self._advanced_radio = QtWidgets.QRadioButton("advanced")
        self._normal_radio.setChecked(True)
        group = QtWidgets.QButtonGroup(self._diag)
        group.addButton(self._normal_radio)
        group.addButton(self._advanced_radio)
        self._label_group = group
        self._coords_check = QtWidgets.QCheckBox("Include coordinates")
        row = QtWidgets.QHBoxLayout()
        row.addWidget(self._labels_check)
        row.addWidget(self._normal_radio)
        row.addWidget(self._advanced_radio)
        row.addWidget(self._coords_check)
        row.addStretch(1)
        grid = self._diag.layout()
        grid.addLayout(row, grid.rowCount(), 0, 1, grid.columnCount())
        self._labels_check.toggled.connect(self._sync_label_radios)
        self._sync_label_radios()

    def _sync_label_radios(self, _checked=False):
        """Enable the mode radios only while labels are included."""
        on = self._labels_check.isChecked()
        self._normal_radio.setEnabled(on)
        self._advanced_radio.setEnabled(on)

    def populate_menu(self):
        self.add_action(
            "File",
            text="Save 2D canvas",
            tip="Save the focused 2D canvas as a PNG or JPEG image",
            func=self.run,
            id="file.save_2d_canvas",
            weight=30,
        )

    def run(self):
        widget = self._mgr.currentR2DWidget()
        if widget is None:
            self._pycon.writeToHistory(
                "Save 2D canvas: no focused 2D canvas\n")
            return False
        self._ensure_dialog()
        self._init_label_controls(widget)
        self._on_filter_selected(self._diag.selectedNameFilter())
        self._diag.open(self, QtCore.SLOT("on_finished()"))
        return True

    def _init_label_controls(self, widget):
        """Preset the label controls from the canvas's current overlay.

        The export defaults to matching what the canvas shows on screen.
        """
        on, advanced, coords = _gui_common.label_switch_and_mode(
            widget.overlay)
        self._labels_check.setChecked(on)
        self._coords_check.setChecked(coords)
        radio = self._advanced_radio if advanced else self._normal_radio
        radio.setChecked(True)
        self._sync_label_radios()

    def _on_filter_selected(self, name_filter):
        filt = (name_filter or "").lower()
        if "jpg" in filt or "jpeg" in filt:
            self._diag.setDefaultSuffix("jpg")
        else:
            self._diag.setDefaultSuffix("png")

    @QtCore.Slot()
    def on_finished(self):
        selected = self._diag.selectedFiles()
        if not selected:
            return
        path = resolve_save_path(selected[0], self._diag.selectedNameFilter())
        if path is None:
            self._pycon.writeToHistory(
                "Save 2D canvas: only png or jpg is allowed\n")
            return
        self._save_current(path)

    def _export_overlay(self, widget):
        """Build the overlay to bake into the export from the dialog controls.

        Starts from the canvas's current overlay so a highlight or bounding
        box set elsewhere survives the export.
        """
        on = self._labels_check.isChecked()
        advanced = on and self._advanced_radio.isChecked()
        coords = self._coords_check.isChecked()
        return _gui_common.apply_label_mode(
            widget.overlay, on, advanced, coords)

    def _save_current(self, path):
        widget = self._mgr.currentR2DWidget()
        if widget is None:
            self._pycon.writeToHistory(
                "Save 2D canvas: no focused 2D canvas\n")
            return False
        # _export_overlay reads the dialog's label controls; build it first.
        self._ensure_dialog()
        ok = widget.saveImage(path, self._export_overlay(widget))
        if ok:
            self._pycon.writeToHistory(f"Save 2D canvas: wrote {path}\n")
        else:
            self._pycon.writeToHistory(
                f"Save 2D canvas: failed to write {path}\n")
        return ok


class Canvas(_gui_common.PilotFeature):
    """
    Canvas feature providing menu items for drawing curves and polygons.
    """

    def __init__(self, *args, **kw):
        self._painter = kw.pop('painter')
        super(Canvas, self).__init__(*args, **kw)
        self._world = core.WorldFp64()
        self._widget = None
        self._widget_2d = None
        self._blank_worlds = []

    def populate_menu(self):
        # Group the geometry samples under their own submenu, leaving the
        # working items at the Canvas top level.
        self._mgr.menu_model.menu("Canvas/Samples", weight=20)
        self.add_action(
            "Canvas/Samples",
            text="Sample: Create ICCAD-2013",
            tip="Create ICCAD-2013 polygon examples",
            func=self.mesh_iccad_2013,
            id="canvas.sample.iccad2013",
            weight=10,
        )
        self.add_action(
            "Canvas/Samples",
            text="Sample: Bezier S-curve",
            tip="Draw a sample S-shaped cubic Bezier curve with control "
                "points",
            func=self._bezier_s_curve,
            id="canvas.sample.bezier_s",
            weight=20,
        )
        self.add_action(
            "Canvas/Samples",
            text="Sample: Bezier Arch",
            tip="Draw a sample arch-shaped cubic Bezier curve with control "
                "points",
            func=self._bezier_arch,
            id="canvas.sample.bezier_arch",
            weight=30,
        )
        self.add_action(
            "Canvas/Samples",
            text="Sample: Bezier Loop",
            tip="Draw a sample loop-like cubic Bezier curve with control "
                "points",
            func=self._bezier_loop,
            id="canvas.sample.bezier_loop",
            weight=40,
        )
        self.add_action(
            "Canvas/Samples",
            text="Sample: Ellipse",
            tip="Draw a sample ellipse (a=2, b=1)",
            func=self._ellipse,
            id="canvas.sample.ellipse",
            weight=50,
        )
        self.add_action(
            "Canvas/Samples",
            text="Sample: Parabola",
            tip="Draw a sample parabola (y = 0.5*x^2)",
            func=self._parabola,
            id="canvas.sample.parabola",
            weight=60,
        )
        self.add_action(
            "Canvas/Samples",
            text="Sample: Hyperbola",
            tip="Draw a sample hyperbola (both branches)",
            func=self._hyperbola,
            id="canvas.sample.hyperbola",
            weight=70,
        )

        self._mgr.menu_model.place_separator("Canvas", weight=30)
        self.add_action(
            "Canvas",
            text="Create blank 2D canvas",
            tip="Open an empty 2D canvas with the Painter toolbox for "
                "drawing shapes",
            func=self._create_blank_2d_canvas,
            id="canvas.blank_2d",
            weight=80,
        )
        self.add_action(
            "Canvas",
            text="View: Open canvas in 2D",
            tip="Show the current canvas world in a strictly-2D QPainter "
                "widget; the same world also drives the 3D view",
            func=self._open_2d,
            id="canvas.open_2d",
            weight=90,
        )

    def _create_blank_2d_canvas(self):
        """
        Open an empty 2D canvas and show the Painter toolbox. Each blank
        canvas gets its own world, so shapes drawn here stay independent of
        the sample geometry and of other blank canvases. The new canvas
        takes focus, so the toolbox drives it right away.
        """
        world = core.WorldFp64()
        widget = self._mgr.add2DWidget()
        widget.updateWorld(world)
        widget.resetView()
        self._blank_worlds.append(world)
        self._painter.present()
        return widget

    @staticmethod
    def _draw_layer(world, layer):
        point_type = core.Point3dFp64

        for polygon in layer.get_polys():
            segment_pad = core.SegmentPadFp64(ndim=2)

            for coord in polygon:
                segment_pad.append(core.Segment3dFp64(
                    point_type(coord[0][0], coord[0][1]),
                    point_type(coord[1][0], coord[1][1])
                ))

            world.add_segments(pad=segment_pad)

    def mesh_iccad_2013(self):
        layer = plane_layer.PlaneLayer()
        layer.add_figure("RECT N M1 70 800 180 40")
        layer.add_figure(
            "PGON N M1 70 720 410 720 410 920 70 920 "
            "70 880 370 880 370 760 70 760"
        )
        layer.add_figure("RECT N M1 70 1060 180 40")
        layer.add_figure(
            "PGON N M1 70 980 410 980 410 1180 70 1180 "
            "70 1140 370 1140 370 1020 70 1020"
        )

        self._draw_layer(self._world, layer)
        self._update_widget()

    def _update_widget(self):
        # The canvas world is planar geometry, so it renders in the 2D canvas
        # (the 3D domain viewer is for meshes and fields).
        if self._widget is None:
            self._widget = self._mgr.add2DWidget()
        self._widget.updateWorld(self._world)
        self._widget.resetView()
        # Keep a separately-opened 2D view in sync with the same world.
        if self._widget_2d is not None:
            self._widget_2d.updateWorld(self._world)

    def _open_2d(self):
        """
        Show the current canvas world in a strictly-2D QPainter widget. The
        world is the same object the 3D view renders; the 2D widget simply
        drops the z coordinate. Subsequent samples refresh both views via
        ``_update_widget``.
        """
        if self._widget_2d is None:
            self._widget_2d = self._mgr.add2DWidget()
        self._widget_2d.updateWorld(self._world)
        self._widget_2d.resetView()

    def _bezier_s_curve(self):
        bezier_sample = curve.BezierSample.s_curve()
        sampler = curve.BezierSampler(self._world, bezier_sample)
        sampler.draw(nsample=50, fac=1.0, off_x=0.0, off_y=0.0)
        self._update_widget()

    def _bezier_arch(self):
        bezier_sample = curve.BezierSample.arch()
        sampler = curve.BezierSampler(self._world, bezier_sample)
        sampler.draw(nsample=50, fac=1.0, off_x=0.0, off_y=0.0)
        self._update_widget()

    def _bezier_loop(self):
        bezier_sample = curve.BezierSample.loop()
        sampler = curve.BezierSampler(self._world, bezier_sample)
        sampler.draw(nsample=50, fac=1.0, off_x=0.0, off_y=0.0)
        self._update_widget()

    def _ellipse(self):
        ellipse = curve.Ellipse(a=2.0, b=1.0)
        sampler = curve.CurveSampler(self._world, ellipse)
        sampler.populate_points(npoint=100)
        sampler.draw_cbc()
        self._update_widget()

    def _parabola(self):
        parabola = curve.Parabola(a=0.5, t_min=-3.0, t_max=6.0)
        sampler = curve.CurveSampler(self._world, parabola)
        sampler.populate_points(npoint=100)
        sampler.draw_cbc()
        self._update_widget()

    def _hyperbola(self):
        hyperbola = curve.Hyperbola(a=1.0, b=1.0, t_min=-2.0, t_max=2.0)

        right_sampler = curve.CurveSampler(self._world, hyperbola)
        right_sampler.populate_points(npoint=100)
        right_sampler.draw_cbc()

        left_sampler = curve.CurveSampler(self._world, hyperbola)
        left_sampler.populate_points(npoint=100)
        left_sampler.points.x.ndarray[:] *= -1.0
        left_sampler.draw_cbc()

        self._update_widget()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
