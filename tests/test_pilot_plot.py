# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the native xy-plot core: PlotColor, the matplotlib C0-C9 cycle,
and RPlotSeries.

The core is pure C++ with no Qt widget, so everything here is exercised
through the pybind11 surface registered into ``solvcon.pilot``.
"""

import re
import unittest

import numpy as np

import solvcon

try:
    from solvcon import pilot
except ImportError:
    pilot = None


def _array(values):
    """Wrap a copy of a sequence of numbers as a float64 SimpleArray."""
    return solvcon.SimpleArrayFloat64(array=np.array(values, dtype='float64'))


def _series(x_values, y_values):
    """Build an RPlotSeries holding the given samples."""
    ser = pilot.RPlotSeries()
    ser.set_data(_array(x_values), _array(y_values))
    return ser


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class PilotPlotTC(unittest.TestCase):
    """The native plot vocabulary: the color cycle and the xy series."""

    def test_cycle_is_matplotlib_c0_to_c9(self):
        cycle = pilot.plot_color_cycle()
        self.assertEqual(10, len(cycle))
        self.assertEqual((31, 119, 180, 255),
                         (cycle[0].r, cycle[0].g, cycle[0].b, cycle[0].a))
        self.assertEqual(cycle[0], pilot.plot_cycle_color(10))
        self.assertEqual(cycle[3], pilot.plot_cycle_color(23))

    def test_color_is_an_immutable_value(self):
        color = pilot.PlotColor(1, 2, 3)
        with self.assertRaises(AttributeError):
            color.a = 9
        self.assertEqual(pilot.PlotColor(1, 2, 3, 255), color)
        self.assertNotEqual(pilot.PlotColor(1, 2, 3, 128), color)
        self.assertFalse(color == None)  # noqa: E711
        self.assertTrue(color != None)  # noqa: E711
        self.assertNotIn(color, [1, 'a'])
        self.assertEqual('custom', {color: 'custom'}[pilot.PlotColor(1, 2, 3)])

    def test_set_data_stores_the_samples(self):
        ser = _series([0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0])
        self.assertEqual(4, ser.size)
        self.assertEqual(4, len(ser))
        for index in range(4):
            self.assertEqual(float(index), ser.x(index))
            self.assertEqual(10.0 + index, ser.y(index))

    def test_set_data_copies_the_operand_buffer(self):
        samples = np.arange(4, dtype='float64')
        ser = pilot.RPlotSeries()
        ser.set_data(solvcon.SimpleArrayFloat64(array=samples),
                     solvcon.SimpleArrayFloat64(array=samples))
        limits = ser.data_limits()
        samples[0] = -100.0
        self.assertEqual(0.0, ser.x(0))
        self.assertEqual(limits, ser.data_limits())

    def test_clear_data_empties_and_invalidates(self):
        ser = _series([0.0, 1.0], [2.0, 3.0])
        self.assertIsNotNone(ser.data_limits())
        ser.clear_data()
        self.assertEqual(0, ser.size)
        self.assertIsNone(ser.data_limits())

    def test_index_out_of_range_raises_index_error(self):
        ser = _series([0.0, 1.0, 2.0], [3.0, 4.0, 5.0])
        for name in ('x', 'y'):
            for index in (3, -1):
                with self.subTest(accessor=name, index=index):
                    message = ('index %d is out of bounds with size 3'
                               % index)
                    with self.assertRaisesRegex(IndexError,
                                                re.escape(message)):
                        getattr(ser, name)(index)

    def test_set_data_rejects_what_it_cannot_draw(self):
        ser = pilot.RPlotSeries()
        long_x = _array(np.arange(10, dtype='float64'))
        short_y = _array(np.arange(9, dtype='float64'))
        square = _array(np.zeros((2, 3), dtype='float64'))
        # Wrap the strided view itself: SimpleArrayFloat64 keeps the NumPy
        # stride, which is the input that must not reach the accessors.
        strided = solvcon.SimpleArrayFloat64(
            array=np.arange(10, dtype='float64')[::2])
        # A collector clones the whole buffer, so the ghost part would become
        # samples that the nbody-based length check never saw.
        ghosted = _array(np.arange(10, dtype='float64'))
        ghosted.nghost = 3
        cases = [
            ('length', long_x, short_y,
             'x and y must have the same length, but they are 10 and 9'),
            ('ndim', square, square, 'must be 1-dimensional, but ndim is 2'),
            ('stride', strided, strided,
             'must be contiguous with unit stride, but stride is 2'),
            ('ghost', ghosted, ghosted,
             'must be ghost-free, but nghost is 3'),
        ]
        for reason, x_arr, y_arr, message in cases:
            with self.subTest(reason=reason):
                with self.assertRaisesRegex(ValueError, re.escape(message)):
                    ser.set_data(x_arr, y_arr)

    def test_stride_is_checked_only_where_a_step_exists(self):
        for values in ([], np.zeros(0, dtype='float64'),
                       np.arange(10, dtype='float64')[5:5]):
            ser = _series(values, values)
            self.assertEqual(0, ser.size)
            self.assertIsNone(ser.data_limits())

        one = solvcon.SimpleArrayFloat64(
            array=np.arange(4, dtype='float64')[::2][:1])
        ser = pilot.RPlotSeries()
        ser.set_data(one, one)
        self.assertEqual(1, ser.size)
        self.assertEqual(0.0, ser.x(0))

    def test_rejected_set_data_leaves_the_series_untouched(self):
        ser = _series([0.0, 1.0, 2.0], [3.0, 4.0, 5.0])
        size = ser.size
        limits = ser.data_limits()
        with self.assertRaises(ValueError):
            ser.set_data(_array(np.arange(10, dtype='float64')),
                         _array(np.arange(9, dtype='float64')))
        self.assertEqual(size, ser.size)
        self.assertEqual(limits, ser.data_limits())

    def test_data_limits_are_the_raw_extent(self):
        ser = _series([3.0, -1.0, 2.0], [7.0, 9.0, -4.0])
        self.assertEqual((-1.0, 3.0, -4.0, 9.0), ser.data_limits())
        self.assertEqual((5.0, 5.0, 7.0, 7.0),
                         _series([5.0], [7.0]).data_limits())

    def test_non_finite_sample_is_dropped_whole(self):
        for bad in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(bad=bad):
                ser = _series([0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, bad])
                self.assertEqual((0.0, 2.0, 10.0, 12.0), ser.data_limits())
                ser = _series([bad, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0])
                self.assertEqual((1.0, 3.0, 11.0, 13.0), ser.data_limits())
                self.assertIsNone(_series([bad] * 4, [bad] * 4).data_limits())

    def test_limits_cache_is_stable_and_invalidates(self):
        ser = _series([0.0, 1.0], [2.0, 3.0])
        first = ser.data_limits()
        self.assertEqual((0.0, 1.0, 2.0, 3.0), first)
        self.assertEqual(first, ser.data_limits())
        ser.set_data(_array([0.0, 4.0]), _array([2.0, 8.0]))
        self.assertEqual((0.0, 4.0, 2.0, 8.0), ser.data_limits())

    def test_style_changes_do_not_disturb_the_limits(self):
        ser = _series([0.0, 1.0], [2.0, 3.0])
        limits = ser.data_limits()
        self.assertFalse(ser.color_is_set)
        self.assertEqual('', ser.label)
        ser.label = 'pressure'
        ser.color = pilot.PlotColor(1, 2, 3, 4)
        ser.line_width = 2.5
        self.assertEqual('pressure', ser.label)
        self.assertEqual(pilot.PlotColor(1, 2, 3, 4), ser.color)
        self.assertEqual(2.5, ser.line_width)
        self.assertTrue(ser.color_is_set)
        self.assertEqual(limits, ser.data_limits())

    def test_bad_line_width_is_rejected(self):
        ser = pilot.RPlotSeries()
        for width in (0.0, -1.0, float('nan')):
            with self.assertRaises(ValueError):
                ser.line_width = width
        self.assertEqual(1.5, ser.line_width)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class PilotPlotModelTC(unittest.TestCase):
    """The series list of one plot and the view derived from it."""

    def test_add_series_walks_the_color_cycle(self):
        model = pilot.RPlotModel()
        self.assertEqual(pilot.plot_cycle_color(0), model.add_series().color)
        self.assertEqual(pilot.plot_cycle_color(1), model.add_series().color)
        colored = pilot.RPlotSeries()
        colored.color = pilot.PlotColor(9, 9, 9)
        self.assertEqual(pilot.PlotColor(9, 9, 9),
                         model.add_series(colored).color)
        self.assertEqual(pilot.plot_cycle_color(2), model.add_series().color)

    def test_added_series_is_shared_not_copied(self):
        model = pilot.RPlotModel()
        ser = pilot.RPlotSeries()
        model.add_series(ser)
        ser.label = 'pressure'
        ser.set_data(_array([0.0, 1.0]), _array([2.0, 3.0]))
        self.assertEqual('pressure', model.series(0).label)
        self.assertEqual((0.0, 1.0, 2.0, 3.0), model.series(0).data_limits())
        self.assertEqual(1, model.size)
        self.assertEqual(1, len(model))

    def test_add_series_rejects_none(self):
        model = pilot.RPlotModel()
        with self.assertRaisesRegex(ValueError, 'must not be None'):
            model.add_series(None)
        self.assertEqual(0, model.size)

    def test_series_index_out_of_range_raises_index_error(self):
        model = pilot.RPlotModel()
        model.add_series()
        for index in (1, -1):
            message = 'index %d is out of bounds with size 1' % index
            with self.assertRaisesRegex(IndexError, re.escape(message)):
                model.series(index)

    def test_data_limits_union_all_series(self):
        model = pilot.RPlotModel()
        self.assertIsNone(model.data_limits())
        model.add_series().set_data(_array([0.0, 1.0]), _array([5.0, 6.0]))
        model.add_series()
        self.assertEqual((0.0, 1.0, 5.0, 6.0), model.data_limits())
        model.add_series().set_data(_array([-3.0, 0.5]), _array([7.0, 8.0]))
        self.assertEqual((-3.0, 1.0, 5.0, 8.0), model.data_limits())

    def test_autoscale_margins_the_data(self):
        model = pilot.RPlotModel()
        self.assertEqual((0.0, 1.0, 0.0, 1.0), model.view_limits())
        model.autoscale()
        self.assertEqual((0.0, 1.0, 0.0, 1.0), model.view_limits())
        model.add_series().set_data(_array([0.0, 10.0]),
                                    _array([0.0, 100.0]))
        self.assertEqual(0.05, model.margin)
        model.autoscale()
        for expected, actual in zip((-0.5, 10.5, -5.0, 105.0),
                                    model.view_limits()):
            self.assertAlmostEqual(expected, actual, places=12)

    def test_autoscale_guards_a_singular_span(self):
        model = pilot.RPlotModel()
        model.add_series().set_data(_array([3.0]), _array([0.0]))
        model.autoscale()
        # x: opened to 3 +- 0.15, then the 5% margin of the 0.3 span.
        # y: opened to +-0.5 around zero, then the margin of the 1.0 span.
        for expected, actual in zip((2.835, 3.165, -0.55, 0.55),
                                    model.view_limits()):
            self.assertAlmostEqual(expected, actual, places=12)

    def test_margin_and_view_limits_are_validated(self):
        model = pilot.RPlotModel()
        for margin in (-0.1, float('nan')):
            with self.assertRaises(ValueError):
                model.margin = margin
        model.margin = 0.0
        model.add_series().set_data(_array([0.0, 10.0]), _array([1.0, 2.0]))
        model.autoscale()
        self.assertEqual((0.0, 10.0, 1.0, 2.0), model.view_limits())
        for bad in ((1.0, 1.0, 0.0, 1.0), (0.0, 1.0, 2.0, 1.0),
                    (float('nan'), 1.0, 0.0, 1.0)):
            with self.assertRaises(ValueError):
                model.set_view_limits(*bad)
        model.set_view_limits(-2.0, 2.0, -4.0, 4.0)
        self.assertEqual((-2.0, 2.0, -4.0, 4.0), model.view_limits())

    def test_view_fits_and_centers_the_limits(self):
        model = pilot.RPlotModel()
        model.set_view_limits(0.0, 10.0, 0.0, 10.0)
        transform = model.view(200.0, 100.0)
        self.assertEqual(10.0, transform.zoom)
        self.assertEqual((100.0, 50.0), transform.screen_from_world(5.0, 5.0))
        self.assertEqual((50.0, 100.0), transform.screen_from_world(0.0, 0.0))
        for width, height in ((0.0, 100.0), (200.0, -1.0),
                              (float('nan'), 100.0)):
            with self.assertRaises(ValueError):
                model.view(width, height)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
