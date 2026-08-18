# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The color-bar legend of the oblique-shock reflection viewer.

The bar is a widget, not a window, so it is checked here rather than in the
window lane: a `ColorBar` and a coarse session are all it takes to give it a
range and read back what it drew.  Where it stands in the sub-window is the
viewer's business and is checked there.
"""

import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot.apps.obsrefl import ReflectionSession
    from solvcon.pilot.apps.obsrefl._colorbar import ColorBar
except ImportError:
    pilot = None


class _Recorder(object):
    """Stand in for the painter and note what it was asked to draw."""

    def __init__(self):
        self.lines = 0
        self.texts = []

    def drawLine(self, *_args):
        self.lines += 1

    def drawText(self, _rect, _align, text):
        self.texts.append(text)


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class ColorBarTC(unittest.TestCase):
    """What the legend spans, what it ticks, and what it paints."""

    @classmethod
    def setUpClass(cls):
        # The manager owns the QApplication the widget needs to exist.
        pilot.RManager.instance.setUp()

    def _scaled(self, name='density', vertical=False):
        """A bar filled from one marched chunk of a coarse run, with the
        run and the drawn field range it was filled from."""
        sess = ReflectionSession(nx=8, ny=3, steps_per_chunk=3)
        sess.advance()
        field = sess.field.field(name)
        vmin, vmax = float(field.min()), float(field.max())
        bar = ColorBar(vertical=vertical)
        bar.show_scale(*sess.analysis.color_range(name, vmin, vmax))
        return bar, sess, vmin, vmax

    def test_the_legend_spans_the_range_the_viewer_pins(self):
        # The legend is what says which color means what, so it has to be
        # drawn over the range the viewer colors against, not the frame's
        # own: a field short of the answer would otherwise read as if it
        # had arrived.
        bar, sess, vmin, vmax = self._scaled()
        self.assertEqual(sess.analysis.color_range('density', vmin, vmax),
                         (bar.lo, bar.hi))
        self.assertLessEqual(bar.lo, vmin)
        self.assertGreaterEqual(bar.hi, vmax)

    def test_a_field_with_no_analytic_value_spans_the_frame(self):
        # The CFL number belongs to the discretization, not to the flow, so
        # there is no analytic value to widen its range to and the ramp is
        # the frame's own.
        bar, _, vmin, vmax = self._scaled('cfl')
        self.assertEqual((vmin, vmax), (bar.lo, bar.hi))

    def test_an_unscaled_legend_draws_nothing_and_survives(self):
        bar = ColorBar()
        self.assertIsNone(bar.lo)
        # A ramp with no range still has to paint, since the bar stands
        # before the first run fills it.
        self.assertFalse(bar.grab().isNull())
        # A degenerate range is no range at all; mapping it would divide by
        # its zero span.
        bar.show_scale(1.0, 1.0)
        self.assertIsNone(bar.lo)

    def test_a_flat_legend_runs_low_at_the_left(self):
        # The ramp is drawn rather than laid out from child widgets, so
        # only rendering it says whether it works at all.
        bar, _, _, _ = self._scaled()
        bar.resize(200, bar.thickness())
        pixmap = bar.grab()
        image = pixmap.toImage()
        self.assertFalse(image.isNull())
        # A grab comes back in device pixels, which on a scaled display are
        # not the widget's own; the ramp has to be sampled where it landed.
        scale = pixmap.devicePixelRatio()
        rect = bar._bar_rect()
        row = round(rect.center().y() * scale)
        low = image.pixelColor(round((rect.left() + 2) * scale), row)
        high = image.pixelColor(round((rect.right() - 2) * scale), row)
        # The map runs blue to red, so the two ends cannot come out alike.
        self.assertGreater(high.red(), low.red())
        self.assertGreater(low.blue(), high.blue())

    def test_a_vertical_legend_runs_low_at_the_bottom(self):
        # A bar against a side edge runs up the view, the way an axis does,
        # so the low end has to be at the bottom rather than the left.
        bar, _, _, _ = self._scaled(vertical=True)
        bar.resize(bar.thickness(), 200)
        rect = bar._bar_rect()
        self.assertGreater(bar._at(bar.lo, rect), bar._at(bar.hi, rect))
        pixmap = bar.grab()
        image = pixmap.toImage()
        scale = pixmap.devicePixelRatio()
        column = round(rect.center().x() * scale)
        low = image.pixelColor(column, round((rect.bottom() - 2) * scale))
        high = image.pixelColor(column, round((rect.top() + 2) * scale))
        self.assertGreater(low.blue(), high.blue())
        self.assertGreater(high.red(), low.red())

    def test_a_vertical_legend_keeps_its_labels_inside(self):
        # The end labels sit in a column beside the ramp; a column running
        # off the widget would lose the reading it carries.
        bar, _, _, _ = self._scaled(vertical=True)
        bar.resize(bar.thickness(), 200)
        rect = bar._bar_rect()
        self.assertGreaterEqual(rect.left(), bar.LABEL_WIDTH)
        self.assertLessEqual(rect.right(), bar.width())

    def test_the_ramp_carries_the_two_ends_and_nothing_else(self):
        # Marking the analytic zone values here would put a second reading
        # on a scale, where the eye takes it for the scale itself.  The bar
        # is the two ends of the range and the ramp between them.
        bar, _, _, _ = self._scaled()
        bar.resize(240, bar.thickness())
        drawn = _Recorder()
        bar._draw_ends(drawn, bar._bar_rect())
        self.assertEqual(2, len(drawn.texts))
        self.assertEqual(0, drawn.lines)
        # And there is no zone marking left to call.
        self.assertFalse(hasattr(bar, '_draw_zones'))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
