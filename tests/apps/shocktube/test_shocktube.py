# Copyright (c) 2018, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import unittest

import numpy as np

from solvcon.onedim import euler
from solvcon.apps import shocktube


class ShockTubeTC(unittest.TestCase):

    def setUp(self):
        self.st = shocktube.ShockTube()
        self.st.build_constant(
            gamma=1.4,
            pressure1=1.0,
            density1=1.0,
            pressure5=0.1,
            density5=0.125,
        )

    def _check_field_value_at_t0(self):
        np.testing.assert_equal(0, self.st.velocity_field)
        slct_left = self.st.coord_field < 0
        slct_right = np.logical_not(slct_left)
        np.testing.assert_equal(self.st.pressure1,
                                self.st.pressure_field[slct_left])
        np.testing.assert_equal(self.st.pressure5,
                                self.st.pressure_field[slct_right])
        np.testing.assert_equal(self.st.density1,
                                self.st.density_field[slct_left])
        np.testing.assert_equal(self.st.density5,
                                self.st.density_field[slct_right])

    def test_field_without_numerical(self):
        self.st.build_field(t=0.0, coord=np.linspace(-1, 1, num=11))
        self.assertIs(self.st.svr, None)
        self.assertEqual(len(self.st.coord_field), 11)
        self._check_field_value_at_t0()

    def test_field_with_numerical(self):
        self.st.build_numerical(xmin=-1, xmax=1, ncoord=21,
                                time_increment=0.05, keep_edge=True)
        self.st.build_field(t=0.0)
        self.assertIsInstance(self.st.svr, euler.Euler1DSolver)
        self.assertEqual(len(self.st.coord_field), 11)
        self._check_field_value_at_t0()

    def test_pressure45(self):
        np.testing.assert_allclose(self.st.calc_pressure45(),
                                   3.0313017805064697, rtol=1e-7)

    def test_constant(self):
        np.testing.assert_allclose(
            [self.st.density1, self.st.density3, self.st.density4,
             self.st.density5],
            [1.0, 0.42631942817849494, 0.2655737117053072, 0.125],
            rtol=1e-7)
        self.assertEqual(self.st.pressure3, self.st.pressure4)
        np.testing.assert_allclose(
            [self.st.pressure1, self.st.pressure3, self.st.pressure4,
             self.st.pressure5],
            [1.0, 0.303130178050647, 0.303130178050647, 0.1],
            rtol=1e-7)
        self.assertEqual(self.st.velocity3, self.st.velocity4)
        np.testing.assert_allclose(
            [self.st.velocity1, self.st.velocity3, self.st.velocity4,
             self.st.velocity5],
            [0.0, 0.9274526200489503, 0.9274526200489503, 0.0],
            rtol=1e-7)

    def test_locations(self):
        np.testing.assert_allclose(
            self.st.calc_locations(t=0.1),
            [-0.11832159566199232, -0.007027281256118345, 0.09274526200489504,
             0.17521557320301784],
            rtol=1e-7)

    def test_speedofsound(self):
        st = self.st
        self.assertEqual(
            np.sqrt(st.gamma),
            st.calc_speedofsound(pressure=1.0, density=1.0))
        self.assertEqual(np.sqrt(st.gamma), st.speedofsound1)
        self.assertNotEqual(st.speedofsound5, st.speedofsound1)
        self.assertEqual(np.sqrt(st.gamma * st.pressure5 / st.density5),
                         st.speedofsound5)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
