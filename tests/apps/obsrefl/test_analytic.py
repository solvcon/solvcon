# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The analysis of the oblique-shock reflection.

Most checks seed the analytic answer into the solver field instead of
marching to it.  The exact solution is the one field whose zones, shock
angle, and profile are all known in advance, so a measurement that misreads
it is wrong whatever the solver does, and the seeded checks stay fast enough
for the lane that runs without a window.  What is left to the marched run is
the plumbing: a field that has not developed its shocks yet must report a
missing measurement rather than raise.

The module imports the analysis without Qt, which is the point of keeping it
out of the panel.
"""

import math
import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from solvcon.multidim.euler import EulerField
from solvcon.pilot.apps.obsrefl import ObliqueShock, Reflection
from solvcon.pilot.apps.obsrefl import _analytic


def seed_analytic(analysis):
    """Fill the solver field with the analytic three-zone answer.

    The zone masks taken without a margin partition the body cells, so every
    cell gets the conserved row of the zone its centroid falls in.  The view
    aliases the solver memory, which is what makes the seeding stick.
    """
    masks = analysis.zone_masks(margin=0.0)
    zones = analysis.zone_conserved()
    cons = analysis.field.conserved
    cons[:] = zones[1]
    cons[masks[0]] = zones[0]
    cons[masks[2]] = zones[2]
    return masks


def build(cell_type='quad', nx=64, ny=16, **kw):
    """Return the driver and the analysis of an unmarched run."""
    shock = ObliqueShock()
    shock.build_constant()
    shock.build_numerical(cell_type=cell_type, nx=nx, ny=ny, **kw)
    return shock, Reflection(shock)


class AnalyticFieldTC(unittest.TestCase):
    """The analysis of the exact answer on a structured mesh."""

    NX = 64
    NY = 16

    def setUp(self):
        self.shock, self.analysis = build(nx=self.NX, ny=self.NY)
        self.masks = seed_analytic(self.analysis)

    def test_the_shock_path_is_carried_as_segments(self):
        # The analytic path is the geometry every measurement is cut from,
        # so it is kept as the segments the path names rather than derived
        # again at each use.  The two arms meet at the reflection point.
        path = self.shock.shock_path()
        self.assertEqual(len(path) - 1, len(self.analysis.arms))
        incident, reflected = self.analysis.incident, self.analysis.reflected
        assert_almost_equal([incident.x0, incident.y0], path[0])
        assert_almost_equal([incident.x1, incident.y1], path[1])
        assert_almost_equal([reflected.x0, reflected.y0], path[1])
        assert_almost_equal([reflected.x1, reflected.y1], path[2])
        # Packed for a plot or a viewer overlay: one row per arm, holding
        # the two endpoints of a plane segment.
        assert_almost_equal(self.analysis.arms.pack_array().ndarray,
                            [[incident.x0, incident.y0,
                              incident.x1, incident.y1],
                             [reflected.x0, reflected.y0,
                              reflected.x1, reflected.y1]])

    def test_zone_masks_partition_the_domain(self):
        # Without a margin every cell belongs to exactly one zone, which is
        # what lets the seeding cover the field.
        counts = [int(mask.sum()) for mask in self.masks]
        self.assertEqual(self.shock.mesh.ncell, sum(counts))
        self.assertTrue(all(count > 0 for count in counts))
        for it in range(len(self.masks)):
            for jt in range(it + 1, len(self.masks)):
                self.assertFalse((self.masks[it] & self.masks[jt]).any())

    def test_the_margin_only_drops_cells(self):
        # A margin narrows each zone to a subset of the cells it holds
        # without one, which is what standing back from the shocks means.
        for kept, whole in zip(self.analysis.zone_masks(), self.masks):
            self.assertTrue((whole | kept == whole).all())
            self.assertLess(int(kept.sum()), int(whole.sum()))

    def test_zone_info_recovers_every_analytic_state(self):
        # The seeded field is the analytic answer, so each zone average has
        # to reproduce its state to the last bit for every derived field.
        for name in self.analysis.zone_fields():
            for record in self.analysis.zone_info(name):
                self.assertGreater(record.count, 0)
                assert_almost_equal(record.computed, record.analytic,
                                    decimal=12)

    def test_a_field_with_no_analytic_value_reports_no_zones(self):
        # A CFL number has no analytic value to be held against.
        self.assertEqual([], self.analysis.zone_info('cfl'))
        with self.assertRaises(ValueError):
            self.analysis.zone_info('nonesuch')

    def test_the_color_range_reaches_the_analytic_values(self):
        # A field short of its analytic range still colors against that
        # range; one the reflection cannot answer for keeps its own.
        rho = self.analysis.zone_field('density')
        self.assertEqual((float(rho.min()), float(rho.max())),
                         self.analysis.color_range('density', 1e9, -1e9))
        self.assertEqual((0.2, 0.7),
                         self.analysis.color_range('cfl', 0.2, 0.7))

    def test_the_analytic_side_answers_every_flow_field(self):
        # The computed and the analytic side are compared field by field, so
        # the analysis has to derive every flow quantity the field reader
        # does, and the primitives have to come back as the driver states
        # them.  The CFL number is the one field it answers nothing for.
        self.assertEqual(set(EulerField.FIELDS) - {'cfl'},
                         set(self.analysis.zone_fields()))
        states = self.shock.zone_states()
        assert_almost_equal(self.analysis.zone_field('density'),
                            [state[0] for state in states])
        assert_almost_equal(self.analysis.zone_field('pressure'),
                            [state[3] for state in states])

    def test_incident_angle_fit(self):
        # One crossing per mesh column, each pinned to within half a cell of
        # the shock, fits the angle to a fraction of a degree here.
        fit = self.analysis.fit_incident_angle()
        assert_almost_equal(fit.analytic,
                            math.degrees(self.shock.shock_angle), decimal=12)
        self.assertLess(abs(fit.degree - fit.analytic), 0.5)

    def test_reflection_point(self):
        # The fitted incident line meets the wall where the analytic path
        # turns, to within a cell.
        wall = self.analysis.reflection_point()
        self.assertLess(abs(wall.x - wall.analytic),
                        self.shock.mesher.cell_extent[0])
        assert_almost_equal(wall.analytic, self.shock.shock_path()[1][0])

    def test_profile_matches_the_analytic_step(self):
        # A line through a row of centroids samples that row alone, and the
        # seeded row is the analytic step it is drawn against.
        height = (self.shock.mesher.y1 - self.shock.mesher.y0) * 8.5 / self.NY
        profile = self.analysis.profile(height)
        self.assertEqual(self.NX, len(profile.x))
        self.assertTrue((np.diff(profile.x) > 0.0).all())
        assert_almost_equal(profile.computed, profile.analytic, decimal=12)
        # Density only ever rises downstream: free stream, zone 2, zone 3.
        self.assertTrue((np.diff(profile.analytic) >= 0.0).all())
        self.assertEqual(3, len(set(profile.analytic.tolist())))


class UnstructuredFlavorTC(unittest.TestCase):
    """The crossing walk holds up where the cells are not in rows.

    Only the unstructured flavor is checked: it is the one whose centroids
    sit at scattered heights, so a column bin has no row to walk and the
    walk has to work on the cells it finds.
    """

    def test_incident_angle_fit(self):
        shock, analysis = build(cell_type='unstructured', nx=32, ny=8)
        seed_analytic(analysis)
        fit = analysis.fit_incident_angle()
        self.assertLess(abs(fit.degree - fit.analytic), 1.5)


class UndevelopedFieldTC(unittest.TestCase):
    """A field the shocks have not reached yet reports what it cannot
    measure instead of raising, because the panel takes these readings while
    the run is still marching."""

    def setUp(self):
        self.shock, self.analysis = build(nx=24, ny=8)

    def test_the_free_stream_carries_no_crossing(self):
        # init_solution fills the domain with the free stream, whose density
        # never reaches the mid value the crossing walk looks for.
        self.assertEqual(0, len(self.analysis.crossings()))
        fit = self.analysis.fit_incident_angle()
        self.assertEqual(0, fit.npoint)
        self.assertTrue(math.isnan(fit.degree))
        self.assertTrue(math.isnan(fit.error))
        self.assertTrue(math.isnan(self.analysis.reflection_point().x))

    def test_an_empty_zone_reports_nothing(self):
        # A margin wide enough to swallow a zone leaves nothing to average.
        self.analysis.margin = 10.0
        record = self.analysis.zone_info()[0]
        self.assertEqual(0, record.count)
        self.assertTrue(math.isnan(record.computed))


class ShortDomainTC(unittest.TestCase):
    """A domain too short for the reflection has no zone 3."""

    def setUp(self):
        self.shock, self.analysis = build(nx=16, ny=8, ur=(1.0, 1.0))

    def test_the_incident_shock_leaves_through_the_outflow(self):
        # The path is one arm, so the pad holds one segment and there is no
        # reflected shock to hand out.
        self.assertEqual(2, len(self.shock.shock_path()))
        self.assertFalse(self.analysis.has_reflection)
        self.assertEqual(1, len(self.analysis.arms))
        self.assertIsNone(self.analysis.reflected)
        self.assertEqual(self.shock.mesher.x1, self.analysis.incident.x1)

    def test_zone_three_is_empty(self):
        masks = self.analysis.zone_masks(margin=0.0)
        self.assertEqual(0, int(masks[2].sum()))
        self.assertEqual(self.shock.mesh.ncell,
                         int(masks[0].sum()) + int(masks[1].sum()))

    def test_the_reflection_point_has_no_analytic_answer(self):
        seed_analytic(self.analysis)
        wall = self.analysis.reflection_point()
        self.assertTrue(math.isnan(wall.analytic))
        self.assertTrue(math.isnan(wall.error))
        # The incident shock is still there to fit.
        fit = self.analysis.fit_incident_angle()
        self.assertLess(abs(fit.degree - fit.analytic), 1.5)


class ReflectionBuildTC(unittest.TestCase):

    def test_reflection_requires_a_solver(self):
        shock = ObliqueShock()
        shock.build_constant()
        with self.assertRaises(ValueError):
            Reflection(shock)


class ZoneInfoTC(unittest.TestCase):
    """The zone number is checked on its way in.

    A report is read by a panel and written into a saved run report, where a
    number the analytic solution has no zone for would be taken for a real
    reading.
    """

    @staticmethod
    def _report(zone):
        return _analytic.ZoneInfo(zone, 1, 1.0, 1.0, 0.0)

    def test_only_the_analytic_zones_are_accepted(self):
        for zone in _analytic.ZoneInfo.ZONES:
            self.assertEqual(zone, self._report(zone).zone)
        for zone in (0, 4, '2'):
            with self.assertRaises(ValueError):
                self._report(zone)

    def test_the_check_outlives_the_construction(self):
        report = self._report(1)
        report.zone = 3
        self.assertEqual(3, report.zone)
        with self.assertRaises(ValueError):
            report.zone = 4


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
