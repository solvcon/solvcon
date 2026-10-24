# Copyright (c) 2024, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

import solvcon as sc
from solvcon import testing
from solvcon.pilot.canvas import _naca_airfoil


class Naca4TC(unittest.TestCase):
    def test_npoint(self):
        def _check_size(naca4):
            points = naca4.calc_points(5)
            self.assertEqual(points.ndim, 2)
            self.assertEqual(len(points), 11)
            self.assertEqual((11, 2), points.pack_array().shape)
            points = naca4.calc_points(11)
            self.assertEqual(points.ndim, 2)
            self.assertEqual(len(points), 23)
            self.assertEqual((23, 2), points.pack_array().shape)

        _check_size(_naca_airfoil.Naca4(
            number='0012', open_trailing_edge=False, cosine_spacing=False))
        _check_size(_naca_airfoil.Naca4(
            number='0012', open_trailing_edge=True, cosine_spacing=False))
        _check_size(_naca_airfoil.Naca4(
            number='0012', open_trailing_edge=True, cosine_spacing=False))
        _check_size(_naca_airfoil.Naca4(
            number='0012', open_trailing_edge=True, cosine_spacing=True))


class Naca4SamplerTC(unittest.TestCase):
    def test_construction(self):
        w = sc.WorldFp64()
        naca4 = _naca_airfoil.Naca4(number='0012', open_trailing_edge=False,
                                    cosine_spacing=False)
        _naca_airfoil.Naca4Sampler(w, naca4)


@testing.requires_pilot
class CanvasSampleTC(unittest.TestCase):
    """The airfoil sample of the Canvas/Samples menu."""

    def setUp(self):
        from solvcon import pilot
        from solvcon.pilot.canvas import _canvas_gui
        from solvcon.pilot import painter as _painter
        self.mgr = pilot.RManager.instance.setUp()
        self.canvas = _canvas_gui.Canvas(
            mgr=self.mgr, painter=_painter.Painter(mgr=self.mgr))
        self.before = len(self.mgr.mdiArea.subWindowList())

    def tearDown(self):
        # Close the canvas window while the canvas lives: its close filter
        # dies with the canvas and would break a later close of the window.
        for w in self.mgr.mdiArea.subWindowList()[self.before:]:
            w.close()

    def test_airfoil_is_drawn_in_a_2d_canvas(self):
        canvas = self.canvas
        canvas._naca4_airfoil()
        self.assertEqual(self.before + 1,
                         len(self.mgr.mdiArea.subWindowList()))
        self.assertIsNotNone(canvas._widget)
        self.assertGreater(canvas._world.nbezier, 0)
        # A second sample joins the same world in the same canvas.
        nbezier = canvas._world.nbezier
        canvas._naca4_airfoil()
        self.assertEqual(self.before + 1,
                         len(self.mgr.mdiArea.subWindowList()))
        self.assertEqual(canvas._world.nbezier, 2 * nbezier)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
