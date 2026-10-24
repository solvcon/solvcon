# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The field coloring and the color legend of ``solvcon.pilot.visual``."""

import unittest

import numpy as np
from numpy.testing import assert_allclose

from solvcon import mesh as meshmod
from solvcon import testing

try:
    from solvcon import pilot
    from solvcon.pilot.visual import ColorBar, _colorizer
except ImportError:
    pilot = None

CELL_TYPES = ('quad', 'triangle', 'unstructured')


def _make_mesh(cell_type, nx=4, ny=2):
    return meshmod.RectDomainMesher(nx=nx, ny=ny).make_mesh(
        cell_type=cell_type)


def _cells(mh, name):
    """The non-ghost rows of a per-cell mesh array."""
    arr = getattr(mh, name)
    return arr.ndarray[arr.nghost:]


@testing.requires_pilot
class ColormapTC(unittest.TestCase):

    def test_the_ramp_runs_blue_to_red(self):
        low, high = _colorizer.colormap([0.0, 1.0])
        self.assertGreater(low[2], low[0])
        self.assertGreater(high[0], high[2])

    def test_values_outside_the_unit_range_take_the_ends(self):
        assert_allclose(_colorizer.colormap(-3.0), _colorizer.colormap(0.0))
        assert_allclose(_colorizer.colormap(7.0), _colorizer.colormap(1.0))

    def test_the_channels_stay_inside_the_unit_range(self):
        rgb = _colorizer.colormap(np.linspace(0.0, 1.0, 101, dtype='float64'))
        self.assertEqual((101, 3), rgb.shape)
        self.assertGreaterEqual(rgb.min(), 0.0)
        self.assertLessEqual(rgb.max(), 1.0)


@testing.requires_pilot
class CellTriangulationTC(unittest.TestCase):

    def _fan(self, cell_type):
        mh = _make_mesh(cell_type)
        fan, counts = _colorizer.cell_triangulation(mh)
        return mh, fan.pack_array().ndarray.reshape(-1, 3, 3), counts

    def test_each_cell_fans_into_its_own_triangles(self):
        for cell_type in CELL_TYPES:
            with self.subTest(cell_type=cell_type):
                mh, tris, counts = self._fan(cell_type)
                nnd = _cells(mh, 'clnds')[:, 0]
                self.assertEqual(mh.ncell, len(counts))
                self.assertEqual(list(3 * (nnd - 2)), list(counts))
                self.assertEqual(counts.sum(), 3 * len(tris))

    def test_the_fan_covers_the_cells_and_nothing_more(self):
        # A fan that skipped or overlapped a triangle would still have the
        # right count; only the area says it tiles the cells exactly.
        for cell_type in CELL_TYPES:
            with self.subTest(cell_type=cell_type):
                mh, tris, _ = self._fan(cell_type)
                edge1 = tris[:, 1, :2] - tris[:, 0, :2]
                edge2 = tris[:, 2, :2] - tris[:, 0, :2]
                area = 0.5 * np.abs(edge1[:, 0] * edge2[:, 1]
                                    - edge1[:, 1] * edge2[:, 0])
                assert_allclose(_cells(mh, 'clvol').sum(), area.sum(),
                                rtol=1.e-6)

    def test_the_fan_lies_on_the_field_plane(self):
        _, tris, _ = self._fan('unstructured')
        self.assertEqual(0.0, np.abs(tris[:, :, 2]).max())


@testing.requires_pilot
class FieldColorsTC(unittest.TestCase):

    def test_each_cell_repeats_its_color_over_its_vertices(self):
        field = np.array([0.0, 0.5, 1.0], dtype='float64')
        counts = np.array([3, 6, 3], dtype='int32')
        colors = _colorizer.field_colors(field, counts, 0.0, 1.0)
        self.assertEqual((12, 3), colors.shape)
        self.assertEqual(np.float32, colors.dtype)
        assert_allclose(colors[:3], np.tile(_colorizer.colormap(0.0), (3, 1)))
        assert_allclose(colors[3:9], np.tile(_colorizer.colormap(0.5), (6, 1)))
        assert_allclose(colors[9:], np.tile(_colorizer.colormap(1.0), (3, 1)))

    def test_the_range_scales_the_field(self):
        field = np.array([10.0, 20.0], dtype='float64')
        colors = _colorizer.field_colors(field, np.ones(2, dtype='int32'),
                                         10.0, 20.0)
        assert_allclose(colors, _colorizer.colormap([0.0, 1.0]))

    def test_a_flat_range_paints_the_low_end(self):
        # A uniform field gives a zero span, which must not divide.
        colors = _colorizer.field_colors(np.full(4, 2.0, dtype='float64'),
                                         np.full(4, 3, dtype='int32'),
                                         2.0, 2.0)
        self.assertTrue(np.isfinite(colors).all())
        assert_allclose(colors, np.tile(_colorizer.colormap(0.0), (12, 1)))


@testing.requires_pilot
class ThickSegmentsTC(unittest.TestCase):

    def test_a_segment_becomes_a_quad_run_past_its_ends(self):
        quad = _colorizer.thick_segments([(0.0, 0.0)], [(1.0, 0.0)], 0.1)
        self.assertEqual((6, 2), quad.shape)
        assert_allclose([-0.1, -0.1], quad.min(axis=0))
        assert_allclose([1.1, 0.1], quad.max(axis=0))

    def test_the_quad_follows_the_segment_direction(self):
        quad = _colorizer.thick_segments([(0.0, 0.0)], [(0.0, 2.0)], 0.25)
        assert_allclose([-0.25, -0.25], quad.min(axis=0))
        assert_allclose([0.25, 2.25], quad.max(axis=0))

    def test_a_segment_of_no_length_is_dropped(self):
        quads = _colorizer.thick_segments([(0.0, 0.0), (1.0, 1.0)],
                                          [(1.0, 0.0), (1.0, 1.0)], 0.1)
        self.assertEqual((6, 2), quads.shape)
        empty = _colorizer.thick_segments([(1.0, 1.0)], [(1.0, 1.0)], 0.1)
        self.assertEqual((0, 2), empty.shape)


@testing.requires_pilot
class CellRingsTC(unittest.TestCase):

    def test_each_marked_cell_gives_its_whole_rim(self):
        for cell_type in CELL_TYPES:
            with self.subTest(cell_type=cell_type):
                mh = _make_mesh(cell_type)
                mark = np.zeros(mh.ncell, dtype='bool')
                mark[::2] = True
                starts, ends = _colorizer.cell_rings(mh, mark)
                nnd = _cells(mh, 'clnds')[mark, 0]
                self.assertEqual(nnd.sum(), len(starts))
                self.assertEqual(len(starts), len(ends))

    def test_a_rim_closes_on_itself(self):
        # Every node a rim leaves from is one it also arrives at, which
        # holds only when the last edge comes back to the first node.
        mh = _make_mesh('unstructured')
        for icl in range(mh.ncell):
            mark = np.zeros(mh.ncell, dtype='bool')
            mark[icl] = True
            starts, ends = _colorizer.cell_rings(mh, mark)
            self.assertEqual(sorted(map(tuple, starts)),
                             sorted(map(tuple, ends)))

    def test_no_marked_cell_gives_no_rim(self):
        mh = _make_mesh('quad')
        mark = np.zeros(mh.ncell, dtype='bool')
        starts, ends = _colorizer.cell_rings(mh, mark)
        self.assertEqual(0, len(starts))
        self.assertEqual(0, len(ends))


@testing.requires_pilot
class FieldColorizerTC(unittest.TestCase):

    def setUp(self):
        self.mesh = _make_mesh('unstructured', nx=8, ny=3)
        self.colorizer = _colorizer.FieldColorizer(self.mesh)
        self.field = np.linspace(0.0, 1.0, self.mesh.ncell, dtype='float64')

    def _box(self):
        verts = self.colorizer.verts.ndarray
        return verts.min(axis=0).tolist(), verts.max(axis=0).tolist()

    def _count(self, color):
        colors = self.colorizer.colors(self.field, 0.0, 1.0).ndarray
        return int((colors == np.array(color, dtype='float32')
                    ).all(axis=1).sum())

    def test_the_soup_is_indexed_in_order(self):
        verts = self.colorizer.verts.ndarray
        indices = self.colorizer.indices.ndarray
        self.assertEqual(0, len(verts) % 3)
        self.assertEqual((len(verts) // 3, 3), indices.shape)
        self.assertEqual(list(range(len(verts))), list(indices.ravel()))

    def test_a_frame_colors_every_vertex(self):
        colors = self.colorizer.colors(self.field, 0.0, 1.0).ndarray
        self.assertEqual((len(self.colorizer.verts.ndarray), 3), colors.shape)

    def test_a_bare_field_carries_only_the_anchors(self):
        # Two zero-area triangles pin the box; nothing else is gauge.
        verts = self.colorizer.verts.ndarray
        lifted = verts[verts[:, 2] > 0.0]
        self.assertEqual(6, len(lifted))
        self.assertEqual(0, self._count(self.colorizer.OUTLINE_COLOR))
        self.assertEqual(0, self._count(self.colorizer.BAR_COLOR))

    def test_each_gauge_layer_switches_on_its_own(self):
        mark = np.zeros(self.mesh.ncell, dtype='bool')
        mark[:3] = True
        self.colorizer.set_gauge(mark, None)
        self.assertGreater(self._count(self.colorizer.OUTLINE_COLOR), 0)
        self.assertEqual(0, self._count(self.colorizer.BAR_COLOR))
        self.colorizer.set_gauge(None, 0.5)
        self.assertEqual(0, self._count(self.colorizer.OUTLINE_COLOR))
        self.assertEqual(6, self._count(self.colorizer.BAR_COLOR))

    def test_the_gauge_stands_above_the_field_and_the_bar_on_top(self):
        mark = np.ones(self.mesh.ncell, dtype='bool')
        self.colorizer.set_gauge(mark, 0.5)
        verts = self.colorizer.verts.ndarray
        colors = self.colorizer.colors(self.field, 0.0, 1.0).ndarray
        nfield = len(verts) - len(self.colorizer._gauge)
        self.assertEqual(0.0, np.abs(verts[:nfield, 2]).max())
        outline = (colors == np.array(self.colorizer.OUTLINE_COLOR,
                                      dtype='float32')).all(axis=1)
        bar = (colors == np.array(self.colorizer.BAR_COLOR,
                                  dtype='float32')).all(axis=1)
        assert_allclose(self.colorizer.OUTLINE_LIFT, verts[outline, 2])
        assert_allclose(self.colorizer.BAR_LIFT, verts[bar, 2])
        self.assertGreater(self.colorizer.BAR_LIFT,
                           self.colorizer.OUTLINE_LIFT)

    def test_the_gauge_holds_the_framing_box_still(self):
        # Installing a field refits the camera when its box changes.
        box = self._box()
        mark = np.ones(self.mesh.ncell, dtype='bool')
        for height in (0.0, 0.5, 1.0):
            with self.subTest(height=height):
                self.colorizer.set_gauge(mark, height)
                self.assertEqual(box, self._box())
        self.colorizer.set_gauge(None, None)
        self.assertEqual(box, self._box())

    def test_a_finer_mesh_frames_the_same_box(self):
        # The gauge is sized off the cell and the anchors off the domain,
        # so a remesh does not move the box either.
        fine = _colorizer.FieldColorizer(_make_mesh('quad', nx=24, ny=9))
        fine.set_gauge(np.ones(24 * 9, dtype='bool'), 1.0)
        self.colorizer.set_gauge(None, None)
        assert_allclose(self._box()[0][:2], fine.verts.ndarray.min(axis=0)[:2])
        assert_allclose(self._box()[1][:2], fine.verts.ndarray.max(axis=0)[:2])


class _Recorder(object):
    """Stand in for the painter and note what it was asked to draw."""

    def __init__(self):
        self.lines = 0
        self.texts = []

    def drawLine(self, *_args):
        self.lines += 1

    def drawText(self, _rect, _align, text):
        self.texts.append(text)


@testing.requires_pilot
class ColorBarTC(unittest.TestCase):
    """What the legend spans, what it labels, and what it paints."""

    @classmethod
    def setUpClass(cls):
        # The manager owns the QApplication the widget needs to exist.
        pilot.RManager.instance.setUp()

    @staticmethod
    def _scaled(lo=0.25, hi=1.75, vertical=False):
        bar = ColorBar(vertical=vertical)
        bar.show_scale(lo, hi)
        return bar

    def test_the_legend_spans_the_range_it_is_given(self):
        bar = self._scaled(-2, 3)
        self.assertEqual((-2.0, 3.0), (bar.lo, bar.hi))
        self.assertIsInstance(bar.lo, float)

    def test_an_unscaled_legend_draws_nothing_and_survives(self):
        bar = ColorBar()
        self.assertIsNone(bar.lo)
        # A ramp with no range still has to paint, since the bar stands
        # before the first run fills it.
        self.assertFalse(bar.grab().isNull())

    def test_a_missing_or_degenerate_range_blanks_the_legend(self):
        # A degenerate range is no range at all; mapping it would divide by
        # its zero span.  A reversed one would run the ramp backwards.
        for lo, hi in ((1.0, 1.0), (2.0, 1.0), (None, 1.0), (0.0, None)):
            with self.subTest(lo=lo, hi=hi):
                bar = self._scaled()
                bar.show_scale(lo, hi)
                self.assertIsNone(bar.lo)
                self.assertIsNone(bar.hi)
                self.assertFalse(bar.grab().isNull())

    def test_a_flat_legend_runs_low_at_the_left(self):
        # The ramp is drawn rather than laid out from child widgets, so
        # only rendering it says whether it works at all.
        bar = self._scaled()
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
        bar = self._scaled(vertical=True)
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
        bar = self._scaled(vertical=True)
        bar.resize(bar.thickness(), 200)
        rect = bar._bar_rect()
        self.assertGreaterEqual(rect.left(), bar.LABEL_WIDTH)
        self.assertLessEqual(rect.right(), bar.width())

    def test_the_ramp_carries_the_two_ends_and_nothing_else(self):
        # A reference value marked here would put a second reading on a
        # scale, where the eye takes it for the scale itself.  The bar is
        # the two ends of the range and the ramp between them.
        bar = self._scaled(0.0, 1234.5)
        bar.resize(240, bar.thickness())
        drawn = _Recorder()
        bar._draw_ends(drawn, bar._bar_rect())
        self.assertEqual(['0.00', '1.23e+03'], drawn.texts)
        self.assertEqual(0, drawn.lines)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
