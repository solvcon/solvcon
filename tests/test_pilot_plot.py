# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

"""
Tests for the native xy-plot core: limits, colors, models, series, and tick
locators.

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


def _limits(xmin, xmax, ymin, ymax):
    """Build a PlotLimits2d with the given bounds."""
    return pilot.PlotLimits2d(xmin, xmax, ymin, ymax)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class PilotPlotTC(unittest.TestCase):
    """The native plot limits, color cycle, and xy series."""

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

    def test_limits_is_a_mutable_value(self):
        limits = _limits(0.0, 1.0, 2.0, 3.0)
        self.assertEqual((0.0, 1.0, 2.0, 3.0), tuple(limits))
        self.assertEqual(_limits(0.0, 1.0, 2.0, 3.0), limits)
        self.assertNotEqual(_limits(0.0, 2.0, 2.0, 3.0), limits)
        self.assertFalse(limits == None)  # noqa: E711
        self.assertTrue(limits != None)  # noqa: E711
        with self.assertRaises(TypeError):
            hash(limits)
        limits.xmin = -1.0
        limits.ymax = 4.0
        self.assertIsNone(limits.merge(_limits(-2.0, 3.0, 1.0, 5.0)))
        self.assertEqual(_limits(-2.0, 3.0, 1.0, 5.0), limits)
        self.assertEqual('PlotLimits2d(xmin=-2, xmax=3, ymin=1, ymax=5)',
                         repr(limits))

    def test_returned_limits_do_not_mutate_the_plot(self):
        ser = _series([0.0, 1.0], [2.0, 3.0])
        limits = ser.data_limits()
        limits.xmin = -1.0
        self.assertEqual(_limits(0.0, 1.0, 2.0, 3.0), ser.data_limits())

        model = pilot.RPlotModel()
        limits = model.view_limits()
        limits.xmin = -1.0
        self.assertEqual(_limits(0.0, 1.0, 0.0, 1.0), model.view_limits())

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
        self.assertEqual(_limits(-1.0, 3.0, -4.0, 9.0), ser.data_limits())
        self.assertEqual(_limits(5.0, 5.0, 7.0, 7.0),
                         _series([5.0], [7.0]).data_limits())

    def test_non_finite_sample_is_dropped_whole(self):
        for bad in (float('nan'), float('inf'), float('-inf')):
            with self.subTest(bad=bad):
                ser = _series([0.0, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, bad])
                self.assertEqual(_limits(0.0, 2.0, 10.0, 12.0),
                                 ser.data_limits())
                ser = _series([bad, 1.0, 2.0, 3.0], [10.0, 11.0, 12.0, 13.0])
                self.assertEqual(_limits(1.0, 3.0, 11.0, 13.0),
                                 ser.data_limits())
                self.assertIsNone(_series([bad] * 4, [bad] * 4).data_limits())

    def test_limits_cache_is_stable_and_invalidates(self):
        ser = _series([0.0, 1.0], [2.0, 3.0])
        first = ser.data_limits()
        self.assertEqual(_limits(0.0, 1.0, 2.0, 3.0), first)
        self.assertEqual(first, ser.data_limits())
        ser.set_data(_array([0.0, 4.0]), _array([2.0, 8.0]))
        self.assertEqual(_limits(0.0, 4.0, 2.0, 8.0), ser.data_limits())

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
class PilotPlotTickerTC(unittest.TestCase):
    """The native locator used to position axis ticks and labels."""

    def test_linear_ticks_are_round_and_cover_the_span(self):
        ticker = pilot.RPlotTicker(5)
        self.assertEqual([0.0, 2.0, 4.0, 6.0, 8.0, 10.0],
                         ticker.locate(0.0, 10.0))
        # Every tick has to land inside the span it was asked for, or the
        # axis is labelled outside its own frame.
        for lo, hi in ((0.3, 0.7), (-5.0, 5.0), (1e4, 1.2e4)):
            for tick in ticker.locate(lo, hi):
                self.assertGreaterEqual(tick, lo)
                self.assertLessEqual(tick, hi)

    def test_invalid_linear_ranges_have_no_ticks(self):
        ticker = pilot.RPlotTicker(5)
        # A flat curve leaves a zero span; ticking it would divide by it.
        self.assertEqual([], ticker.locate(1.0, 1.0))
        self.assertEqual([], ticker.locate(1.0, float('nan')))
        self.assertEqual([], ticker.locate(1.0, float('inf')))

    def test_ticks_are_counted_and_not_accumulated(self):
        ticker = pilot.RPlotTicker(5)
        # Where the span is small beside the offset, adding the step rounds
        # back to where it started and a walk along the axis never ends.
        # This runs inside paintEvent, so it takes the GUI thread with it.
        ticks = ticker.locate(1e16, 1e16 + 4.0)
        self.assertGreater(len(ticks), 0)
        self.assertLessEqual(len(ticks), 12)

    def test_decade_ticks_mark_the_axis_or_its_ends(self):
        ticker = pilot.RPlotTicker(5)
        self.assertEqual([-4.0, -3.0, -2.0],
                         ticker.locate_decades(-4.2, -1.8))
        # A range inside one decade still gets its ends marked, so the axis
        # is never left blank.
        self.assertEqual([-2.4, -2.1],
                         ticker.locate_decades(-2.4, -2.1))
        self.assertEqual([], ticker.locate_decades(float('nan'), 1.0))

    def test_a_log_tick_off_a_whole_decade_reads_its_own_value(self):
        ticker = pilot.RPlotTicker(5)
        # A range inside one decade is ticked at its own ends, which are
        # not powers of ten.  Labelling those as powers of ten puts the
        # axis off by a factor the reader has no way to see.
        self.assertEqual(['1e-4', '1e-3'], ticker.decade_labels([-4.0, -3.0]))
        self.assertEqual(['1.91', '5.235'],
                         ticker.decade_labels(
                             ticker.locate_decades(0.2811, 0.7189)))

    def test_labels_are_short_and_unambiguous(self):
        ticker = pilot.RPlotTicker(5)
        self.assertEqual(['0', '2', '4', '6', '8', '10'],
                         ticker.labels(ticker.locate(0.0, 10.0)))
        self.assertEqual(['0.3', '0.4', '0.5', '0.6', '0.7'],
                         ticker.labels(ticker.locate(0.3, 0.7)))
        self.assertEqual(['10000', '10500', '11000', '11500', '12000'],
                         ticker.labels(ticker.locate(1e4, 1.2e4)))
        # A lone tick has no neighbour to tell apart, so it stays short.
        self.assertEqual(['1234'], ticker.labels([1234.5]))
        self.assertEqual(['1e+05'], ticker.labels([123456.0]))
        self.assertEqual(['1e-04'], ticker.labels([1e-4]))
        self.assertEqual([], ticker.labels([]))

    def test_neighbouring_labels_never_read_alike(self):
        # Rounded on its own, a label can read the same as its neighbour,
        # and an axis that repeats a number tells the reader nothing.
        ticker = pilot.RPlotTicker(4)
        self.assertEqual(['1.0e+05', '1.5e+05', '2.0e+05'],
                         ticker.labels(ticker.locate(1e5, 2e5)))
        self.assertEqual(['1.0134e+05', '1.0136e+05', '1.0138e+05',
                          '1.0140e+05'],
                         ticker.labels(ticker.locate(101325.0, 101400.0)))
        ticker = pilot.RPlotTicker(5)
        self.assertEqual(['1234.0', '1234.1', '1234.2', '1234.3', '1234.4',
                          '1234.5'],
                         ticker.labels(ticker.locate(1234.0, 1234.5)))
        self.assertEqual(['1.9103', '1.9104'],
                         ticker.decade_labels([0.2811, 0.28112]))

    def test_an_axis_keeps_one_notation(self):
        # Plain and exponent forms side by side make neighbouring labels
        # hard to compare, so the largest tick picks the form for all.
        ticker = pilot.RPlotTicker(5)
        self.assertEqual(['0', '5.0e+04', '1.0e+05', '1.5e+05'],
                         ticker.labels(ticker.locate(0.0, 1.5e5)))
        self.assertEqual(['1.0e-04', '1.2e-04', '1.4e-04', '1.6e-04',
                          '1.8e-04', '2.0e-04'],
                         ticker.labels(ticker.locate(1e-4, 2e-4)))

    def test_target_count_is_validated(self):
        # No count is compiled in: plot defaults are to come from a
        # runtime container, so every caller names its own.
        with self.assertRaises(TypeError):
            pilot.RPlotTicker()
        ticker = pilot.RPlotTicker(2)
        self.assertEqual(2, ticker.target_count)
        self.assertEqual([0.0, 5.0, 10.0], ticker.locate(0.0, 10.0))
        ticker.target_count = 3
        self.assertEqual(3, ticker.target_count)
        for value in (0, -1):
            with self.subTest(value=value):
                message = ('target count must be at least 1, but it is %d'
                           % value)
                with self.assertRaisesRegex(ValueError, re.escape(message)):
                    ticker.target_count = value
        self.assertEqual(3, ticker.target_count)
        with self.assertRaises(ValueError):
            pilot.RPlotTicker(0)


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
        self.assertEqual(_limits(0.0, 1.0, 2.0, 3.0),
                         model.series(0).data_limits())
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
        self.assertEqual(_limits(0.0, 1.0, 5.0, 6.0), model.data_limits())
        model.add_series().set_data(_array([-3.0, 0.5]), _array([7.0, 8.0]))
        self.assertEqual(_limits(-3.0, 1.0, 5.0, 8.0), model.data_limits())

    def test_autoscale_margins_the_data(self):
        model = pilot.RPlotModel()
        self.assertEqual(_limits(0.0, 1.0, 0.0, 1.0), model.view_limits())
        model.autoscale()
        self.assertEqual(_limits(0.0, 1.0, 0.0, 1.0), model.view_limits())
        model.add_series().set_data(_array([0.0, 10.0]),
                                    _array([0.0, 100.0]))
        self.assertEqual(0.05, model.margin)
        model.autoscale()
        for expected, actual in zip((-0.5, 10.5, -5.0, 105.0),
                                    tuple(model.view_limits())):
            self.assertAlmostEqual(expected, actual, places=12)

    def test_autoscale_guards_a_singular_span(self):
        model = pilot.RPlotModel()
        model.add_series().set_data(_array([3.0]), _array([0.0]))
        model.autoscale()
        # x: opened to 3 +- 0.15, then the 5% margin of the 0.3 span.
        # y: opened to +-0.5 around zero, then the margin of the 1.0 span.
        for expected, actual in zip((2.835, 3.165, -0.55, 0.55),
                                    tuple(model.view_limits())):
            self.assertAlmostEqual(expected, actual, places=12)

    def test_margin_and_view_limits_are_validated(self):
        model = pilot.RPlotModel()
        for margin in (-0.1, float('nan')):
            with self.assertRaises(ValueError):
                model.margin = margin
        model.margin = 0.0
        model.add_series().set_data(_array([0.0, 10.0]), _array([1.0, 2.0]))
        model.autoscale()
        self.assertEqual(_limits(0.0, 10.0, 1.0, 2.0), model.view_limits())
        for bad in ((1.0, 1.0, 0.0, 1.0), (0.0, 1.0, 2.0, 1.0),
                    (float('nan'), 1.0, 0.0, 1.0)):
            with self.assertRaises(ValueError):
                model.set_view_limits(_limits(*bad))
        model.set_view_limits(_limits(-2.0, 2.0, -4.0, 4.0))
        self.assertEqual(_limits(-2.0, 2.0, -4.0, 4.0), model.view_limits())

    def test_view_fits_and_centers_the_limits(self):
        model = pilot.RPlotModel()
        model.set_view_limits(_limits(0.0, 10.0, 0.0, 10.0))
        transform = model.view(200.0, 100.0)
        self.assertEqual(10.0, transform.zoom)
        self.assertEqual((100.0, 50.0), transform.screen_from_world(5.0, 5.0))
        self.assertEqual((50.0, 100.0), transform.screen_from_world(0.0, 0.0))
        for width, height in ((0.0, 100.0), (200.0, -1.0),
                              (float('nan'), 100.0)):
            with self.assertRaises(ValueError):
                model.view(width, height)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
