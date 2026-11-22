# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""A reflection run read out into the records its widgets draw.

The run marches on a mesh coarse enough to take milliseconds.  The card
compares three zones and toned errors, and a fit that has not formed says
so.  The colors span the pinned range of the analysis.  Both profile series
share their abscissae.  The clock of the run strip runs on the time
increment.
"""

import math
import unittest

import numpy as np

from solvcon.apps.obsrefl import ReflectionSession
from solvcon.apps.obsrefl import _presenter
from solvcon.apps.obsrefl import _viewmodel


def _session(chunks=0, **kw):
    kw.setdefault('nx', 8)
    kw.setdefault('ny', 3)
    sess = ReflectionSession(**kw)
    for _ in range(chunks):
        sess.advance()
    return sess


class ResultsTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.marched = _session(chunks=4)

    def test_the_card_compares_three_zones_with_toned_errors(self):
        res = _presenter.results(self.marched, 'density')
        self.assertEqual('density', res.field)
        self.assertEqual(['zone 1', 'zone 2', 'zone 3'],
                         [row.label for row in res.zones])
        for row, info in zip(res.zones, self.marched.zone_info('density')):
            with self.subTest(zone=row.label):
                self.assertEqual(_viewmodel.format_error(info.error),
                                 row.error.text)
                self.assertEqual(_viewmodel.tone_of(info.error),
                                 row.error.tone)
                self.assertEqual(_viewmodel.format_number(info.analytic),
                                 row.exact.text)
                self.assertIn(f"{info.count} cells", row.computed.tip)
        # The free stream holds while zone 3 has yet to form.
        self.assertEqual('good', res.zones[0].error.tone)
        self.assertEqual('bad', res.zones[2].error.tone)
        self.assertIsNone(res.note)

    def test_the_fits_read_not_formed_at_step_0(self):
        res = _presenter.results(_session(), 'density')
        self.assertEqual(_presenter.FIT_LABELS,
                         tuple(row.label for row in res.fits))
        for row in res.fits:
            with self.subTest(fit=row.label):
                self.assertEqual(("not formed", 'muted'),
                                 row.computed[:2])
                self.assertIn("0 density crossings", row.computed.tip)
                self.assertEqual('muted', row.error.tone)
                self.assertNotEqual('muted', row.exact.tone)
        self.assertEqual((('mass drift', "-"), ('CFL', "-")), res.rows)

    def test_a_formed_fit_reads_against_the_exact_angle(self):
        sess = _session(chunks=40, nx=32, ny=8)
        fit = sess.fit_incident_angle()
        row = _presenter.results(sess, 'density').fits[0]
        self.assertTrue(math.isfinite(fit.degree))
        self.assertEqual(_viewmodel.format_number(fit.degree),
                         row.computed.text)
        self.assertEqual(_viewmodel.format_number(fit.analytic),
                         row.exact.text)
        self.assertEqual(_viewmodel.tone_of(fit.error), row.error.tone)
        self.assertIn(f"{fit.npoint} crossings", row.computed.tip)

    def test_the_run_rows_read_the_mass_drift_and_the_cfl(self):
        records = self.marched.history.records
        drift = (records[-1].mass - records[0].mass) / records[0].mass
        rows = dict(_presenter.results(self.marched, 'density').rows)
        self.assertEqual(_viewmodel.format_signed(drift), rows['mass drift'])
        self.assertTrue(rows['mass drift'].startswith("+"))
        low, high = records[-1].cfl_min, records[-1].cfl_max
        self.assertEqual(f"{low:.2f}-{high:.2f}", rows['CFL'])

    def test_a_field_without_an_exact_value_has_no_zones(self):
        res = _presenter.results(self.marched, 'cfl')
        self.assertEqual('CFL number', res.field)
        self.assertEqual((), res.zones)
        self.assertEqual(2, len(res.fits))
        self.assertEqual('muted', res.note[1])

    def test_an_exact_zero_leaves_the_error_undefined(self):
        # The free stream runs level, so zone 1 has no y velocity to be
        # a percent of.
        row = _presenter.results(self.marched, 'vely').zones[0]
        self.assertEqual(("undefined", 'muted'), row.error[:2])
        self.assertIn("exact value is 0", row.error.tip)

    def test_a_zone_without_cells_reads_so(self):
        # A domain too short for the reflection has no zone 3.
        sess = _session(chunks=1, mach=20.0, angle=0.5)
        self.assertFalse(sess.analysis.has_reflection)
        res = _presenter.results(sess, 'density')
        self.assertEqual(("no cells", 'muted'), res.zones[2].computed[:2])
        self.assertEqual('muted', res.zones[2].error.tone)
        self.assertEqual(("none", 'muted'), res.fits[1].exact[:2])
        self.assertIn("too short", res.note[0])

    def test_a_shock_through_the_top_inlet_voids_the_comparison(self):
        # The default domain holds a 10 deg reflection but not a 15 deg one.
        for angle, note in ((10.0, None), (15.0, 'bad')):
            with self.subTest(angle=angle):
                res = _presenter.results(_session(angle=angle), 'density')
                self.assertEqual(note, res.note and res.note[1])
        self.assertIn("top inlet", res.note[0])


class FieldFrameTC(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sess = _session(chunks=3)

    def test_the_colors_span_the_pinned_range(self):
        for name in ('density', 'pressure', 'cfl'):
            with self.subTest(name=name):
                frame = _presenter.field_frame(
                    self.sess, _viewmodel.Display(field=name))
                values = self.sess.field.field(name)
                np.testing.assert_array_equal(values, frame.values)
                self.assertEqual(self.sess.analysis.color_range(
                    name, float(values.min()), float(values.max())),
                    (frame.lo, frame.hi))
                self.assertEqual(_viewmodel.FIELD_LABELS[name],
                                 frame.label)

    def test_the_gauge_marks_only_what_the_display_asks_for(self):
        bare = _presenter.field_frame(self.sess, _viewmodel.Display())
        self.assertIsNone(bare.gauge_cells)
        self.assertIsNone(bare.gauge_height)
        frame = _presenter.field_frame(self.sess, _viewmodel.Display(
            mark_cells=True, mark_line=True, cut=0.25))
        height = self.sess.analysis.cut_height(0.25)
        self.assertEqual(height, frame.gauge_height)
        np.testing.assert_array_equal(
            np.nonzero(self.sess.analysis.profile_cells(height))[0],
            frame.gauge_cells)


class ProfileFrameTC(unittest.TestCase):

    def test_both_series_share_the_abscissae_of_the_cut(self):
        sess = _session(chunks=3)
        frame = _presenter.profile_frame(sess, _viewmodel.Display(cut=0.5))
        cut = sess.profile(sess.analysis.cut_height(0.5))
        np.testing.assert_array_equal(cut.x, frame.x)
        np.testing.assert_array_equal(cut.computed, frame.computed)
        np.testing.assert_array_equal(cut.analytic, frame.exact)
        self.assertTrue(np.all(np.diff(frame.x) >= 0))
        other = _presenter.profile_frame(sess, _viewmodel.Display(cut=0.3))
        self.assertEqual(frame.key, other.key)
        cfl = _presenter.profile_frame(sess, _viewmodel.Display(field='cfl'))
        self.assertNotEqual(frame.key, cfl.key)
        self.assertTrue(np.all(np.isfinite(cfl.computed)))
        self.assertTrue(np.all(np.isnan(cfl.exact)))


class StripStateTC(unittest.TestCase):

    def test_the_clock_runs_on_the_time_increment(self):
        sess = _session(chunks=2, time_increment=1e-3, max_steps=50)
        strip = _presenter.strip_state(sess, 'paused', dirty=True,
                                       can_reset=False)
        self.assertEqual(_viewmodel.StripState(
            'paused', 10, 50, 10 * 1e-3, 50 * 1e-3, '', True, False), strip)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
