# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The exact Riemann solver.

The star states are pinned to the values other codes publish for the same
tests, Sod's problem to the closed form of the legacy solver, and the sampled
solution to what every exact solution obeys: it starts from the initial data,
it mirrors with the problem, and it conserves mass, momentum, and energy up
to the fluxes through the ends.
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
#: Standard tests on [0, 1] as ``(left, right, x0, t_end)``, with the
#: diaphragm and the end time their sources use (Sod as in the FLASH user's
#: guide, Toro's tests 1 to 4 and 7, and Lax and the shock-contact-shock
#: test of LANL ExactPack); the vacuum gets an end time before its fans
#: leave [0, 1].
PROBLEMS = {
    'sod': SOD + (0.5, 0.2),
    'modified_sod': ((1.0, 0.75, 1.0), (0.125, 0.0, 0.1), 0.3, 0.2),
    'lax': ((0.445, 0.698, 3.528), (0.5, 0.0, 0.571), 0.5, 0.15),
    'einfeldt': EINFELDT + (0.5, 0.15),
    'left_blast': ((1.0, 0.0, 1000.0), (1.0, 0.0, 0.01), 0.5, 0.012),
    'collision': ((5.99924, 19.5975, 460.894),
                  (5.99242, -6.19633, 46.0950), 0.4, 0.035),
    'shock_contact_shock': ((1.0, 0.5, 1.0), (1.25, -0.5, 1.0), 0.5, 0.3),
    'moving_contact': ((1.4, 0.1, 1.0), (1.0, 0.1, 1.0), 0.5, 2.0),
    'vacuum': VACUUM + (0.5, 0.04),
}


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


def _problem(key):
    """Return ``(exact, left, right, t_end)`` of :data:`PROBLEMS` ``key``."""
    left, right, x0, t_end = PROBLEMS[key]
    return shocktube.ExactRiemann(left, right, x0=x0), left, right, t_end


def _flux(state, gamma):
    rho, vel, pre = state
    energy = pre / (gamma - 1.0) + 0.5 * rho * vel * vel
    return np.array([rho * vel, rho * vel * vel + pre, vel * (energy + pre)],
                    dtype='float64')


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


class IntegralTC(unittest.TestCase):
    """Hold the integrals to what the Euler equations demand: every signal
    stays within the speed bound, and the totals over a window change only
    by the fluxes through its ends."""

    def test_max_speed_bounds_every_signal(self):
        coord = np.linspace(0.0, 1.0, 501, dtype='float64')
        for key in PROBLEMS:
            exact, _, _, t_end = _problem(key)
            sol = exact.sample(coord, t_end)
            with np.errstate(divide='ignore', invalid='ignore'):
                signal = np.abs(sol['u']) + np.sqrt(exact.gamma * sol['p']
                                                    / sol['rho'])
            signal = signal[np.isfinite(signal)]
            with self.subTest(key=key):
                self.assertLessEqual(signal.max(),
                                     exact.max_speed() * (1 + 1.e-12))

    def test_the_star_sound_speed_behind_a_shock_bounds_the_signals(self):
        # In Sod's problem the gas behind the shock is hotter than the gas
        # behind the fan at the same p*, and its |u| + c, 2.19, is the
        # largest signal: above every wave speed and above the 1.93 the
        # isentrope from the left state gives.
        exact, _, _, _ = _problem('sod')
        behind = [abs(exact.u_star) + math.sqrt(exact.gamma * exact.p_star
                                                / rho)
                  for rho in (exact.rho_star_left, exact.rho_star_right)]
        self.assertAlmostEqual(1.9252, behind[0], places=4)
        self.assertAlmostEqual(2.1916, behind[1], places=4)
        self.assertEqual(behind[1], exact.max_speed())

    def test_the_solution_conserves_over_a_closed_window(self):
        # Over a window the waves have not left, the totals change only by
        # the fluxes of the two initial states through its ends.
        for key in PROBLEMS:
            exact, left, right, time = _problem(key)
            reach = 1.01 * exact.max_speed() * time
            lower, upper = exact.x0 - reach, exact.x0 + reach
            start = np.array(exact.integrate(lower, upper, 0.0),
                             dtype='float64')
            end = np.array(exact.integrate(lower, upper, time),
                           dtype='float64')
            change = time * (_flux(left, exact.gamma)
                             - _flux(right, exact.gamma))
            with self.subTest(key=key):
                np.testing.assert_allclose(end - start, change, rtol=0,
                                           atol=1.e-11 * np.abs(start).max())

    def test_fans_near_gamma_one_conserve(self):
        # The fans of the streams in
        # VacuumTC.test_a_star_pressure_below_the_floats_is_no_vacuum
        # integrate with powers of the sound speed of order 200.
        sound = math.sqrt(1.01)
        speed = 0.99 * 2.0 / 0.01 * sound
        left, right = (1.0, -speed, 1.0), (1.0, speed, 1.0)
        exact = shocktube.ExactRiemann(left, right, gamma=1.01)
        reach = 1.01 * exact.max_speed()
        start = np.array(exact.integrate(-reach, reach, 0.0), dtype='float64')
        end = np.array(exact.integrate(-reach, reach, 1.0), dtype='float64')
        change = _flux(left, 1.01) - _flux(right, 1.01)
        np.testing.assert_allclose(end - start, change, rtol=0,
                                   atol=1.e-11 * np.abs(change).max())

    def test_the_integral_is_the_integral_of_the_samples(self):
        # integrate() cuts at the fronts and integrates each piece in closed
        # form; a fine trapezoid over the samples has to land on the same
        # numbers.
        exact, _, _, t_end = _problem('lax')
        coord = np.linspace(0.0, 1.0, 200001, dtype='float64')
        sol = exact.sample(coord, t_end)
        energy = (sol['p'] / (exact.gamma - 1.0)
                  + 0.5 * sol['rho'] * sol['u'] ** 2)
        trapezoid = [(0.5 * (values[1:] + values[:-1])
                      * np.diff(coord)).sum()
                     for values in (sol['rho'], sol['rho'] * sol['u'],
                                    energy)]
        np.testing.assert_allclose(trapezoid,
                                   exact.integrate(0.0, 1.0, t_end),
                                   rtol=1.e-5)


class AveragesTC(unittest.TestCase):
    """Hold the cell averages, the reference of a conservative scheme, to
    what such a cell holds: the exact state where the solution is uniform
    over the cell, the mix of the conserved variables where a front cuts
    it, and in total the integral of the solution."""

    def test_a_uniform_cell_holds_its_state_bit_for_bit(self):
        exact, _, _, t_end = _problem('sod')
        faces = np.linspace(0.0, 1.0, 51, dtype='float64')
        avg = exact.averages(faces, t_end)
        lower, upper = faces[:-1], faces[1:]
        sol = exact.sample(0.5 * (lower + upper), t_end)
        head, tail, contact, shock = (pos for _, pos
                                      in exact.positions(t_end))
        uniform = (upper < head) | (lower > tail)
        for pos in (contact, shock):
            uniform &= (upper < pos) | (lower > pos)
        self.assertEqual(36, uniform.sum())
        for name in ('rho', 'u', 'p'):
            np.testing.assert_array_equal(sol[name][uniform],
                                          avg[name][uniform])

    def test_a_cut_cell_holds_the_mix_of_the_conserved_variables(self):
        # The diaphragm of the modified Sod problem cuts the cell [0.28,
        # 0.32] in half; the cell holds the mean momentum and energy, so
        # its velocity is the momentum over the mean density, not the mean
        # velocity.
        exact, (rho_l, u_l, p_l), (rho_r, u_r, p_r), _ = _problem(
            'modified_sod')
        avg = exact.averages([0.28, 0.32], 0.0)
        gm1 = exact.gamma - 1.0
        rho = 0.5 * (rho_l + rho_r)
        mom = 0.5 * (rho_l * u_l + rho_r * u_r)
        energy = 0.5 * ((p_l + p_r) / gm1 + 0.5 * rho_l * u_l ** 2
                        + 0.5 * rho_r * u_r ** 2)
        np.testing.assert_allclose(
            [rho, mom / rho, gm1 * (energy - 0.5 * mom ** 2 / rho)],
            [avg['rho'][0], avg['u'][0], avg['p'][0]], rtol=1.e-14)
        self.assertNotAlmostEqual(0.5 * (u_l + u_r), avg['u'][0], places=3)

    def test_the_cells_add_up_to_the_integral(self):
        faces = np.linspace(0.0, 1.0, 38, dtype='float64') ** 1.5
        for key in ('lax', 'einfeldt', 'vacuum'):
            exact, _, _, t_end = _problem(key)
            avg = exact.averages(faces, t_end)
            # A cell of pure vacuum carries no momentum and no energy,
            # though it has no velocity either.
            rho = avg['rho']
            mom = np.where(rho > 0.0, rho * avg['u'], 0.0)
            kinetic = np.where(rho > 0.0, 0.5 * mom * avg['u'], 0.0)
            cells = np.stack([rho, mom, avg['p'] / (exact.gamma - 1.0)
                              + kinetic], axis=1)
            with self.subTest(key=key):
                np.testing.assert_allclose(
                    exact.integrate(0.0, 1.0, t_end),
                    (cells * np.diff(faces)[:, None]).sum(axis=0),
                    rtol=1.e-13, atol=1.e-15)

    def test_inside_a_fan_the_average_nears_the_centre_at_second_order(self):
        exact, _, _, t_end = _problem('sod')
        head, tail = (exact.x0 + speed * t_end
                      for speed in exact.waves[0].speeds)
        middle = 0.5 * (head + tail)
        centre = float(exact.sample(middle, t_end)['rho'])
        gaps = [float(exact.averages([middle - width, middle + width],
                                     t_end)['rho'][0]) - centre
                for width in (0.02, 0.01)]
        self.assertAlmostEqual(4.0, gaps[0] / gaps[1], delta=0.01)

    def test_a_cell_of_pure_vacuum_has_no_velocity(self):
        # No mass carries a velocity in the vacuum, so the cell mean has
        # none, while sample() draws the fans' velocity on; a cell the
        # vacuum front cuts holds gas and a velocity.
        exact, _, _, t_end = _problem('vacuum')
        front = exact.x0 + exact.waves[1].speeds[1] * t_end
        avg = exact.averages([0.49, 0.51, front - 0.01, front + 0.01], t_end)
        self.assertEqual((0.0, 0.0), (avg['rho'][0], avg['p'][0]))
        self.assertTrue(math.isnan(avg['u'][0]))
        self.assertTrue(math.isfinite(float(exact.sample(0.5, t_end)['u'])))
        self.assertGreater(avg['rho'][2], 0.0)
        self.assertTrue(math.isfinite(avg['u'][2]))

    def test_faces_out_of_order_are_rejected(self):
        # Reversed bounds or a repeated face would otherwise give a
        # plausible but wrong total, or a zero-width cell of infinite
        # density.
        exact, _, _, _ = _problem('sod')
        for faces in ([0.6, 0.4], [0.1, 0.1, 0.2], [0.5], 0.5,
                      [0.1, float('nan')]):
            with self.subTest(faces=faces):
                with self.assertRaisesRegex(ValueError, "strictly ascending"):
                    exact.averages(faces, 0.2)
        with self.assertRaisesRegex(ValueError, "strictly ascending"):
            exact.integrate(0.6, 0.4, 0.2)


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
