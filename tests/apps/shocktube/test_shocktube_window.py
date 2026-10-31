# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The legend, the band, the axes, the zoom, and the x-t head of the
shock-tube window."""

import unittest

import numpy as np

from solvcon import testing

try:
    from solvcon import pilot
    if pilot.enable:
        from solvcon.apps import shocktube
        from solvcon.apps.shocktube import _window
except ImportError:
    pilot = None


@testing.requires_pilot
class WindowTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        pilot.RManager.instance.setUp()

    def setUp(self):
        exact = shocktube.ExactRiemann((1.0, 0.0, 1.0), (0.125, 0.0, 0.1),
                                       gamma=1.4)
        x = np.linspace(-9.8, 9.8, 99, dtype='float64')
        sol = exact.sample(x, 1.0)
        values = {'density': sol['rho'], 'velocity': sol['u'],
                  'pressure': sol['p']}
        self.window = _window.ShockTubeWindow()
        self.window.resize(1000, 700)
        self.plot = self.window.plot
        self.plot.set_data((-10.0, 10.0), (x, values),
                           (x, {name: value + 0.01
                                for name, value in values.items()}))
        self.plot.set_reference(values)
        self.plot.set_fronts(exact.positions(1.0), exact.x0)

    def test_the_legend_names_its_curves(self):
        legend = self.window.legend
        legend.set_entries([('line', "exact"), ('marker', "CESE")])
        self.assertGreater(legend.sizeHint().width(), 0)
        self.assertFalse(legend.grab().isNull())

    def test_the_band_names_the_fronts(self):
        rect = self.plot.panels()[0][1]
        texts = [text for text, _, _ in self.plot.band_labels(rect)]
        self.assertIn('shock', texts)
        self.plot.set_show_fronts(False)
        self.assertEqual(0, self.plot.band_rows())

    def test_the_options_draw_a_log_axis_and_the_residual(self):
        plot = self.plot
        plot.set_option('log', True)
        self.assertTrue(plot.log_axis('density'))
        self.assertFalse(plot.log_axis('velocity'))
        plot.set_option('residual', True)
        self.assertEqual(3, len(plot.strips()))
        np.testing.assert_allclose(0.01, plot.residual('density'))
        self.assertFalse(plot.grab().isNull())

    def test_a_zoom_narrows_the_view_until_reset(self):
        self.plot.zoom(0.5, 0.0)
        self.assertEqual((-5.0, 5.0), self.plot.view)
        self.assertEqual(self.plot.view, self.window.xt.view)
        self.window.fit_button.click()
        self.assertIsNone(self.plot.view)
        self.assertTrue(self.plot.readout('density', 1.0))

    def test_the_xt_head_names_and_keys_its_colors(self):
        xt = self.window.xt
        xt.set_label("density, CESE")
        x = np.linspace(-1.0, 1.0, 11, dtype='float64')
        xt.set_history((-1.0, 1.0), x, [0.0],
                       [np.where(x < 0, 1.0, 0.125)])
        self.assertEqual(("0.125", "1"), xt._ends())
        self.assertFalse(xt.grab().isNull())

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
