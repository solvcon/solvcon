# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The exact Riemann solver.

The star states are pinned to the values other codes publish for the same
tests, Sod's problem to the closed form of the legacy solver, and the sampled
solution to what every exact solution obeys: it starts from the initial data
and it mirrors with the problem.
"""

import math
import unittest

import numpy as np

from solvcon.apps import shocktube
from solvcon.apps.shocktube import _shocktube

SOD = ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1))
#: Einfeldt's 1-2-3 problem: two streams parting, short of vacuum.
EINFELDT = ((1.0, -2.0, 0.4), (1.0, 2.0, 0.4))
#: Two streams parting fast enough to leave vacuum between them.
VACUUM = ((1.0, -10.0, 1.0), (1.0, 10.0, 1.0))


def _published_delta(text):
    """Return the tolerance for a published value printed as ``text``.

    1e-4 relative is the tolerance other exact-solver test suites hold the
    published star states to (for example
    https://github.com/mrodrig6/approximate_riemann_solver); a value printed
    with fewer digits than that is held to one unit in its last digit,
    because the tables round and truncate inconsistently.
    """
    decimals = len(text.split('.')[1]) if '.' in text else 0
    return max(1.e-4 * abs(float(text)), 10.0 ** -decimals)


class PublishedStarStateTC(unittest.TestCase):

    # Table 1 of Toro and co-authors, https://arxiv.org/abs/2412.03522, rows
    # 1 to 5 and 7, as (left, right, p*, u*).  Row 6 prints a garbled p*
    # and u*, so it is left out.
    TABLE_A2412 = [
        ((1.0, 0.0, 1.0), (1.0, 0.0, 0.1), '0.5219', '0.5248'),
        ((1.0, 0.0, 1.0), (0.125, 0.0, 0.1), '0.3031', '0.9274'),
        ((1.0, 0.0, 1.0), (0.001, 0.0, 0.8), '0.8060', '0.1794'),
        ((1.0, 0.0, 0.01), (1.0, 0.0, 1000.0), '460.8938', '-19.5975'),
        ((6.0, 8.0, 460.0), (6.0, -6.0, 46.0), '790.2928', '3.8194'),
        ((1.0, -2.0, 0.4), (1.0, 2.0, 0.4), '0.0019', '0.0000'),
    ]

    def assert_published(self, value, text):
        self.assertAlmostEqual(float(text), value,
                               delta=_published_delta(text))

    def test_sod_matches_the_published_star_state(self):
        # Toro (2009) as quoted by https://github.com/cfdgasman/sod-shock-tube
        # and the validation page of
        # https://github.com/mrodrig6/approximate_riemann_solver.
        exact = shocktube.ExactRiemann(*SOD)
        self.assert_published(exact.p_star, '0.30313')
        self.assert_published(exact.u_star, '0.92745')

    def test_the_published_table_matches(self):
        for left, right, p_star, u_star in self.TABLE_A2412:
            with self.subTest(left=left, right=right):
                exact = shocktube.ExactRiemann(left, right)
                self.assert_published(exact.p_star, p_star)
                self.assert_published(exact.u_star, u_star)

    def test_both_waves_leave_the_star_velocity(self):
        # At p* the gas behind the left wave and behind the right one moves
        # at the same velocity; the mismatch over its slope bounds the error
        # of p*.
        for left, right, _, _ in self.TABLE_A2412:
            exact = shocktube.ExactRiemann(left, right)
            with self.subTest(left=left, right=right):
                star, sides = exact.p_star, ((exact.left, exact.c_left),
                                             (exact.right, exact.c_right))
                mismatch = (sum(exact._velocity_loss(star, *side)
                                for side in sides)
                            + exact.right.u - exact.left.u)
                slope = sum(exact._velocity_loss_slope(star, *side)
                            for side in sides)
                self.assertLess(abs(mismatch) / slope, 1.e-11 * star)

    def test_sod_wave_positions_match_the_published_ones(self):
        # At t = 0.2 with the diaphragm at 0.5: the fan from 0.263 to 0.486,
        # the contact at 0.685, and the shock at 0.850 (the FLASH user's
        # guide and https://github.com/ibackus/sod-shocktube).
        fronts = shocktube.ExactRiemann(*SOD, x0=0.5).positions(0.2)
        self.assertEqual(['head', 'tail', 'contact', 'shock'],
                         [label for label, _ in fronts])
        for (_, pos), published in zip(fronts, (0.263, 0.486, 0.685, 0.850)):
            self.assertAlmostEqual(published, pos, delta=1.e-3)

    def test_sod_agrees_with_the_closed_form(self):
        # The legacy ShockTube solves only this pattern, in closed form, so
        # the two have to agree to round-off where both apply.
        legacy = _shocktube.ShockTube()
        legacy.build_constant(gamma=1.4, pressure1=1.0, density1=1.0,
                              pressure5=0.1, density5=0.125)
        exact = shocktube.ExactRiemann(*SOD)
        self.assertAlmostEqual(legacy.pressure3, exact.p_star, places=13)
        self.assertAlmostEqual(legacy.velocity3, exact.u_star, places=13)
        self.assertAlmostEqual(legacy.density3, exact.rho_star_left,
                               places=13)
        self.assertAlmostEqual(legacy.density4, exact.rho_star_right,
                               places=13)
        coord = np.linspace(-1.0, 1.0, 401, dtype='float64')
        legacy.build_field(t=0.3, coord=coord)
        sol = exact.sample(coord, 0.3)
        np.testing.assert_allclose(legacy.density_field, sol['rho'],
                                   rtol=0, atol=1.e-13)
        np.testing.assert_allclose(legacy.velocity_field, sol['u'],
                                   rtol=0, atol=1.e-13)
        np.testing.assert_allclose(legacy.pressure_field, sol['p'],
                                   rtol=0, atol=1.e-13)


class WavePatternTC(unittest.TestCase):

    def test_the_patterns_name_the_waves_with_a_jump(self):
        # A fan or a contact without a jump is no wave, and V marks the
        # vacuum between two fans.
        for states, pattern in ((SOD, 'R-C-S'), (EINFELDT, 'R-R'),
                                (VACUUM, 'R-V-R')):
            with self.subTest(pattern=pattern):
                self.assertEqual(pattern,
                                 shocktube.ExactRiemann(*states).pattern)
        contact = shocktube.ExactRiemann((1.0, 0.1, 1.0), (0.5, 0.1, 1.0),
                                         x0=0.5)
        self.assertEqual('C', contact.pattern)
        self.assertEqual([('contact', 0.5 + 0.1 * 2.0)],
                         contact.positions(2.0))

    def test_fronts_ascend_and_start_at_the_diaphragm(self):
        exact = shocktube.ExactRiemann(*SOD, x0=0.5)
        at_end = [pos for _, pos in exact.positions(0.2)]
        self.assertEqual(sorted(at_end), at_end)
        self.assertEqual({0.5}, {pos for _, pos in exact.positions(0.0)})

    def test_the_fans_join_the_states_around_them(self):
        exact = shocktube.ExactRiemann(*EINFELDT)
        eps = 1.e-10
        for wave, outer in ((exact.waves[0], exact.left),
                            (exact.waves[2], exact.right)):
            head, tail = wave.speeds
            star = (exact.rho_star_left if 'left' == wave.side
                    else exact.rho_star_right)
            inward = 1.0 if 'left' == wave.side else -1.0
            speeds = np.array([head + inward * eps, tail - inward * eps],
                              dtype='float64')
            sol = exact.sample(exact.x0 + speeds, 1.0)
            self.assertAlmostEqual(outer.rho, sol['rho'][0], places=8)
            self.assertAlmostEqual(outer.p, sol['p'][0], places=8)
            self.assertAlmostEqual(star, sol['rho'][1], places=8)
            self.assertAlmostEqual(exact.p_star, sol['p'][1], places=8)


class SamplingTC(unittest.TestCase):

    def test_time_zero_gives_the_initial_data(self):
        exact = shocktube.ExactRiemann(*SOD, x0=0.5)
        coord = np.array([0.0, 0.4999, 0.5, 0.7], dtype='float64')
        for time in (0.0, -1.0):
            sol = exact.sample(coord, time)
            np.testing.assert_array_equal([1.0, 1.0, 0.125, 0.125],
                                          sol['rho'])
            np.testing.assert_array_equal([0.0, 0.0, 0.0, 0.0], sol['u'])
            np.testing.assert_array_equal([1.0, 1.0, 0.1, 0.1], sol['p'])

    def test_the_mirrored_problem_gives_the_mirrored_solution(self):
        # Swapping the states and reversing the velocities mirrors the
        # solution about the diaphragm.  An irrational spacing keeps every
        # sample off the fronts, where the two sides are ambiguous.
        coord = np.linspace(-0.6, 0.6, 337, dtype='float64') * math.sqrt(2.0)
        for left, right in (SOD, EINFELDT, VACUUM):
            exact = shocktube.ExactRiemann(left, right)
            (rho_l, u_l, p_l), (rho_r, u_r, p_r) = left, right
            mirror = shocktube.ExactRiemann((rho_r, -u_r, p_r),
                                            (rho_l, -u_l, p_l))
            with self.subTest(left=left, right=right):
                self.assertEqual(exact.p_star, mirror.p_star)
                self.assertEqual(exact.rho_star_left, mirror.rho_star_right)
                if not exact.vacuum:
                    self.assertAlmostEqual(-exact.u_star, mirror.u_star,
                                           places=12)
                sol = exact.sample(coord, 0.05)
                sym = mirror.sample(-coord, 0.05)
                np.testing.assert_allclose(sol['rho'], sym['rho'],
                                           rtol=1.e-12, atol=1.e-14)
                np.testing.assert_allclose(sol['u'], -sym['u'],
                                           rtol=1.e-12, atol=1.e-12)
                np.testing.assert_allclose(sol['p'], sym['p'],
                                           rtol=1.e-12, atol=1.e-14)

    def test_sample_keeps_the_shape_of_x(self):
        exact = shocktube.ExactRiemann(*SOD)
        sol = exact.sample(np.zeros((2, 3), dtype='float64'), 0.1)
        self.assertEqual((2, 3), sol['rho'].shape)
        self.assertEqual((), exact.sample(0.3, 0.1)['p'].shape)


class VacuumTC(unittest.TestCase):

    def test_streams_apart_generate_vacuum(self):
        exact = shocktube.ExactRiemann(*VACUUM)
        self.assertTrue(exact.vacuum)
        self.assertEqual((0.0, 0.0, 0.0), (exact.p_star, exact.rho_star_left,
                                           exact.rho_star_right))
        self.assertTrue(math.isnan(exact.u_star))
        self.assertEqual(['rarefaction', 'vacuum', 'rarefaction'],
                         [wave.kind for wave in exact.waves])
        # The vacuum fronts move at u_K -+ 2 c_K / (gamma - 1).
        front = 2.0 / 0.4 * math.sqrt(1.4)
        np.testing.assert_allclose([-10.0 + front, 10.0 - front],
                                   exact.waves[1].speeds, rtol=1.e-14)

    def test_the_fans_drain_into_vacuum(self):
        exact = shocktube.ExactRiemann(*VACUUM)
        time = 0.1
        low, high = exact.waves[1].speeds
        coord = time * np.array([low - 1.e-3, low, 0.0, high, high + 1.e-3],
                                dtype='float64')
        sol = exact.sample(coord, time)
        np.testing.assert_array_equal([0.0, 0.0, 0.0], sol['rho'][1:4])
        np.testing.assert_array_equal([0.0, 0.0, 0.0], sol['p'][1:4])
        self.assertTrue(np.all(sol['rho'][[0, 4]] > 0.0))
        self.assertTrue(np.all(sol['rho'][[0, 4]] < 1.e-3))

    def test_the_123_problem_stays_short_of_vacuum(self):
        # The threshold relative speed is 2 (c_L + c_R) / (gamma - 1) =
        # 7.48 and the streams part at 4, which leaves p* = 0.0019.
        exact = shocktube.ExactRiemann(*EINFELDT)
        self.assertFalse(exact.vacuum)
        self.assertGreater(exact.p_star, 0.0)
        just = shocktube.ExactRiemann((1.0, -3.75, 0.4), (1.0, 3.75, 0.4))
        self.assertTrue(just.vacuum)

    def test_streams_just_short_of_vacuum_converge(self):
        # Streams of rho = p = 1 part into vacuum from |u| = 5.9160798 on;
        # just short of it the star pressure falls to about 1e-46.
        for micro in range(5915000, 5916080):
            speed = micro * 1.e-6
            exact = shocktube.ExactRiemann((1.0, -speed, 1.0),
                                           (1.0, speed, 1.0))
            self.assertFalse(exact.vacuum, speed)
            self.assertGreater(exact.p_star, 0.0, speed)
        self.assertLess(exact.p_star, 1.e-40)
        self.assertTrue(shocktube.ExactRiemann((1.0, -5.91608, 1.0),
                                               (1.0, 5.91608, 1.0)).vacuum)

    def test_a_star_pressure_below_the_floats_is_no_vacuum(self):
        # Near gamma = 1 a fan's sound speed barely falls with its pressure:
        # these streams leave a star pressure of 1e-404, below the smallest
        # float, yet a star sound speed of a hundredth of the outer one, so
        # the fan tails part at -+0.01 c and not at the vacuum fronts.
        sound = math.sqrt(1.01)
        speed = 0.99 * 2.0 / 0.01 * sound
        exact = shocktube.ExactRiemann((1.0, -speed, 1.0), (1.0, speed, 1.0),
                                       gamma=1.01)
        self.assertFalse(exact.vacuum)
        self.assertEqual(0.0, exact.p_star)
        self.assertEqual('R-R', exact.pattern)
        np.testing.assert_allclose(
            [-0.01 * sound, 0.01 * sound],
            [exact.waves[0].speeds[1], exact.waves[2].speeds[1]], rtol=1.e-9)


class DeriveTC(unittest.TestCase):

    def test_derived_quantities(self):
        rho = np.array([1.0, 0.5], dtype='float64')
        vel = np.array([-2.0, 0.0], dtype='float64')
        pre = np.array([0.4, 1.0], dtype='float64')
        gamma = 1.4
        np.testing.assert_allclose(pre / (0.4 * rho),
                                   shocktube.derive('e', rho, vel, pre,
                                                    gamma))
        np.testing.assert_allclose(pre / rho ** gamma,
                                   shocktube.derive('s', rho, vel, pre,
                                                    gamma))
        np.testing.assert_allclose(
            [2.0 / math.sqrt(0.56), 0.0],
            shocktube.derive('mach', rho, vel, pre, gamma))
        out = shocktube.derive('rho', rho, vel, pre, gamma)
        self.assertIsNot(rho, out)
        np.testing.assert_array_equal(rho, out)

    def test_an_unknown_quantity_is_rejected(self):
        for name in shocktube.QUANTITIES:
            shocktube.derive(name, 1.0, 1.0, 1.0, 1.4)
        with self.assertRaisesRegex(ValueError, "unknown quantity"):
            shocktube.derive('temperature', 1.0, 1.0, 1.0, 1.4)

    def test_vacuum_derives_nan_without_complaint(self):
        with np.errstate(all='raise'):
            for name in ('e', 's', 'mach'):
                self.assertTrue(np.isnan(shocktube.derive(name, 0.0, 1.0,
                                                          0.0, 1.4)))


class InputTC(unittest.TestCase):

    def test_unphysical_input_is_rejected(self):
        good = (1.0, 0.0, 1.0)
        for left, right, gamma, x0 in ((good, (0.0, 0.0, 1.0), 1.4, 0.0),
                                       (good, (1.0, 0.0, -1.0), 1.4, 0.0),
                                       ((1.0, math.nan, 1.0), good, 1.4, 0.0),
                                       (good, good, 1.0, 0.0),
                                       (good, good, 1.4, math.inf)):
            with self.subTest(left=left, right=right, gamma=gamma, x0=x0):
                with self.assertRaises(ValueError):
                    shocktube.ExactRiemann(left, right, gamma=gamma, x0=x0)

    def test_states_become_gas_states(self):
        exact = shocktube.ExactRiemann([1, 0, 1], (0.125, 0, 0.1), x0=0.5)
        self.assertEqual(shocktube.GasState(1.0, 0.0, 1.0), exact.left)
        self.assertIsInstance(exact.right.rho, float)
        self.assertEqual(0.5, exact.x0)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
