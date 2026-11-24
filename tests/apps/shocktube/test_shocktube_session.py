# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""A shock-tube run marched a chunk at a time and judged against the exact
solution of its Riemann problem.
"""

import math
import unittest

import numpy as np

from solvcon.apps.shocktube import _riemann, _session, _shocktube


class MarchTC(unittest.TestCase):

    def test_the_chunks_land_on_the_step_cap(self):
        sess = _session.ShockTubeSession(steps_per_chunk=4, max_steps=10)
        self.assertEqual([4, 8, 10],
                         [sess.advance().step for _ in range(3)])
        self.assertTrue(sess.finished)
        self.assertIsNone(sess.advance())
        self.assertEqual(0.5, sess.time)
        # A record and a density row of step 0, then one a chunk.
        self.assertEqual([0, 4, 8, 10],
                         [record.step for record in sess.records])
        self.assertEqual([0.0, 0.2, 0.4, 0.5],
                         [time for time, _ in sess.density_rows])

    def test_a_fine_grid_keeps_fewer_rows(self):
        sess = _session.ShockTubeSession(ncoord=8193, time_increment=0.001,
                                         max_steps=3)
        sess.run()
        self.assertEqual(4, len(sess.density_rows))
        self.assertEqual(sess.ROW_BUDGET // len(sess.x),
                         sess.density_rows.maxlen)

    def test_the_march_is_the_legacy_solver_step_for_step(self):
        sess = _session.ShockTubeSession(steps_per_chunk=5, max_steps=50)
        sess.run()
        tube = _shocktube.ShockTube()
        tube.build_constant(gamma=1.4, pressure1=1.0, density1=1.0,
                            pressure5=0.1, density5=0.125)
        tube.build_numerical(xmin=-10, xmax=10, ncoord=201,
                             time_increment=0.05)
        tube.svr.march_alpha2(steps=50)
        np.testing.assert_array_equal(
            np.array(tube.svr.density, dtype='float64')[tube.svr.xindices],
            sess.values()['density'])

    def test_the_cap_moves_both_ways(self):
        sess = _session.ShockTubeSession(max_steps=3)
        sess.run()
        sess.set_max_steps(5)
        self.assertFalse(sess.finished)
        self.assertEqual('cap', sess.run())
        self.assertEqual(5, sess.step)
        sess.set_max_steps(2)
        self.assertEqual('cap', sess.stop_reason)


class JudgeTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sess = _session.ShockTubeSession()
        cls.sess.run()

    def test_the_mass_holds_until_a_wave_leaves_the_tube(self):
        self.assertLess(abs(self.sess.mass_drift()), 1e-12)
        self.assertGreater(self.sess.records[-1].cfl, 0.5)

    def test_the_star_state_matches_the_exact_one(self):
        zones = {zone.label: zone for zone in self.sess.star_zones()}
        self.assertEqual(['p*', 'u*', 'rho* left', 'rho* right'],
                         list(zones))
        for zone in zones.values():
            with self.subTest(zone=zone.label):
                self.assertGreater(zone.count, 0)
                self.assertLess(abs(zone.error), 0.02)
        self.assertAlmostEqual(0.30313, zones['p*'].exact, places=5)

    def test_the_fronts_stand_near_the_exact_ones(self):
        fits = {fit.label: fit for fit in self.sess.front_fits()}
        self.assertEqual(['contact x', 'shock x'], list(fits))
        shock = fits['shock x']
        self.assertTrue(shock.formed)
        # Within a cell of the exact shock, at 1.7522 t.
        self.assertAlmostEqual(1.7522 * 2.5, shock.exact, places=3)
        self.assertLess(abs(shock.computed - shock.exact), 0.2)

    def test_the_run_strays_little_from_the_cell_means(self):
        self.assertLess(self.sess.l1_error('density'), 0.02)

    def test_nothing_is_judged_before_the_waves_part(self):
        sess = _session.ShockTubeSession(max_steps=1)
        sess.run()
        self.assertEqual([0, 0, 0, 0],
                         [zone.count for zone in sess.star_zones()])
        self.assertTrue(all(math.isnan(zone.error)
                            for zone in sess.star_zones()))
        self.assertEqual([False, False],
                         [fit.formed for fit in sess.front_fits()])

    def test_two_shocks_are_told_apart(self):
        sess = _session.ShockTubeSession(rho_left=1.0, p_left=1.0,
                                         rho_right=1.0, p_right=1.0)
        exact = _riemann.ExactRiemann((1.0, 2.0, 1.0), (1.0, -2.0, 1.0))
        sess.analysis.exact = exact
        x = sess.x
        values = {name: np.zeros_like(x, dtype='float64')
                  for name in ('pressure', 'density')}
        values['pressure'] = exact.sample(x, 2.0)['p']
        fits = sess.analysis.front_fits(values, 2.0)
        self.assertEqual(['left shock x', 'right shock x'],
                         [fit.label for fit in fits])
        for fit in fits:
            self.assertLess(abs(fit.computed - fit.exact), 0.2)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
