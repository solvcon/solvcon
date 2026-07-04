# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Canvas GUI feature for pilot.
"""

from .. import core
from ..plot import curve, plane_layer

from . import _gui_common

__all__ = [
    'Canvas',
]


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
