# Copyright (c) 2023, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import json
import math
import unittest

import numpy as np

import solvcon
from solvcon import testing


class WorldTB(testing.TestBase):

    def test_bezier(self):
        Point = self.Point
        World = self.World

        w = World()

        # Empty
        self.assertEqual(w.nbezier, 0)
        with self.assertRaisesRegex(
                IndexError, "World: \\(bezier\\) i 0 >= size 0"):
            w.bezier(0)

        # Add Bezier curve
        b = w.add_bezier(p0=Point(0, 0, 0), p1=Point(1, 1, 0),
                         p2=Point(3, 1, 0), p3=Point(4, 0, 0))
        self.assertEqual(w.nbezier, 1)
        with self.assertRaisesRegex(
                IndexError, "World: \\(bezier\\) i 1 >= size 1"):
            w.bezier(1)
        b2 = w.add_bezier(b=self.Bezier(p0=Point(0, 0, 1), p1=Point(1, 1, 2),
                                        p2=Point(3, 1, 3), p3=Point(4, 0, 4)))
        self.assertEqual(w.nbezier, 2)
        w.bezier(1)
        with self.assertRaisesRegex(
                IndexError, "World: \\(bezier\\) i 2 >= size 2"):
            w.bezier(2)

        # Check control points
        self.assertEqual(len(b), 4)
        self.assertEqual(list(b[0]), [0, 0, 0])
        self.assertEqual(list(b[1]), [1, 1, 0])
        self.assertEqual(list(b[2]), [3, 1, 0])
        self.assertEqual(list(b[3]), [4, 0, 0])
        self.assertEqual(len(b2), 4)
        self.assertEqual(list(b2[0]), [0, 0, 1])
        self.assertEqual(list(b2[1]), [1, 1, 2])
        self.assertEqual(list(b2[2]), [3, 1, 3])
        self.assertEqual(list(b2[3]), [4, 0, 4])

        # Check locus points
        segs = b.sample(nlocus=5)
        self.assertEqual(len(segs), 4)
        self.assert_allclose(
            list(segs[0]), [[0.0, 0.0, 0.0], [0.90625, 0.5625, 0.0]])
        self.assert_allclose(
            list(segs[1]), [[0.90625, 0.5625, 0.0], [2.0, 0.75, 0.0]])
        self.assert_allclose(
            list(segs[2]), [[2.0, 0.75, 0.0], [3.09375, 0.5625, 0.0]])
        self.assert_allclose(
            list(segs[3]), [[3.09375, 0.5625, 0.0], [4.0, 0.0, 0.0]])

    def test_beziers(self):
        cp = self.CurvePad(ndim=3)
        p0 = self.Point(0, 0, 0)
        p1 = self.Point(1, 1, 0)
        p2 = self.Point(3, 1, 0)
        p3 = self.Point(4, 0, 0)
        cp.append(p0=p0, p1=p1, p2=p2, p3=p3)

        w = self.World()
        self.assertEqual(w.nbezier, 0)
        with self.assertRaisesRegex(
                IndexError, "World: \\(bezier\\) i 0 >= size 0"):
            w.bezier(0)
        w.add_beziers(cp)
        self.assertEqual(w.nbezier, 1)
        b = w.bezier(0)

        self.assertEqual(list(b[0]), [0, 0, 0])
        self.assertEqual(list(b[1]), [1, 1, 0])
        self.assertEqual(list(b[2]), [3, 1, 0])
        self.assertEqual(list(b[3]), [4, 0, 0])

    def test_point(self):
        Point = self.Point
        World = self.World

        w = World()

        # Empty
        self.assertEqual(w.npoint, 0)
        with self.assertRaisesRegex(
                IndexError, "World: \\(point\\) i 0 >= size 0"):
            w.point(0)

        # Add a point by object
        p = w.add_point(Point(0, 1, 2))
        self.assertEqual(list(p), [0, 1, 2])
        self.assertEqual(list(w.point(0)), [0, 1, 2])
        self.assertIsNot(p, w.point(0))
        self.assertEqual(w.npoint, 1)
        with self.assertRaisesRegex(
                IndexError, "World: \\(point\\) i 1 >= size 1"):
            w.point(1)

        # Add a point by coordinate
        p = w.add_point(3.1415, 3.1416, 3.1417)
        self.assert_allclose(list(p), [3.1415, 3.1416, 3.1417])
        self.assert_allclose(list(w.point(1)), [3.1415, 3.1416, 3.1417])
        self.assertIsNot(p, w.point(1))
        self.assertEqual(w.npoint, 2)
        with self.assertRaisesRegex(
                IndexError, "World: \\(point\\) i 2 >= size 2"):
            w.point(2)

        # Add many points
        for it in range(10):
            w.add_point(3.1415 + it, 3.1416 + it, 3.1417 + it)
            self.assertEqual(w.npoint, 2 + it + 1)

        # Array/batch interface
        pndarr = w.points.pack_array().ndarray
        self.assertEqual(pndarr.shape, (12, 3))
        self.assertEqual(w.npoint, 12)

    def test_segment(self):
        Point = self.Point
        Segment = self.Segment
        World = self.World

        w = World()

        # Empty
        self.assertEqual(w.nsegment, 0)
        with self.assertRaisesRegex(
                IndexError, "World: \\(segment\\) i 0 >= size 0"):
            w.segment(0)

        # Add a segment by object
        s = w.add_segment(s=Segment(Point(0, 1, 2), Point(7.1, 8.2, 9.3)))
        self.assert_allclose(list(s), [[0, 1, 2], [7.1, 8.2, 9.3]])
        self.assert_allclose(list(w.segment(0)), [[0, 1, 2], [7.1, 8.2, 9.3]])
        self.assertIsNot(s, w.segment(0))
        self.assertEqual(w.nsegment, 1)
        with self.assertRaisesRegex(
                IndexError, "World: \\(segment\\) i 1 >= size 1"):
            w.segment(1)

        # Add a segment by coordinate
        s = w.add_segment(Point(3.1415, 3.1416, 3.1417), Point(7.1, 8.2, 9.3))
        self.assert_allclose(list(s),
                             [[3.1415, 3.1416, 3.1417], [7.1, 8.2, 9.3]])
        self.assert_allclose(list(w.segment(1)),
                             [[3.1415, 3.1416, 3.1417], [7.1, 8.2, 9.3]])
        self.assertIsNot(s, w.segment(1))
        self.assertEqual(w.nsegment, 2)
        with self.assertRaisesRegex(
                IndexError, "World: \\(segment\\) i 2 >= size 2"):
            w.segment(2)

        # Add many segments
        for it in range(11):
            w.add_segment(Point(3.1415 + it, 3.1416 + it, 3.1417 + it),
                          Point(7.1 + it, 8.2 + it, 9.3 + it))
            self.assertEqual(w.nsegment, 2 + it + 1)

        # Array/batch interface
        sndarr = w.segments.pack_array().ndarray
        self.assertEqual(sndarr.shape, (13, 6))
        self.assertEqual(w.nsegment, 13)

    def test_segments(self):
        x0arr = self.SimpleArray(array=np.array([1, 2, 3], dtype=self.dtype))
        y0arr = self.SimpleArray(array=np.array([4, 5, 6], dtype=self.dtype))
        z0arr = self.SimpleArray(array=np.array([7, 8, 9], dtype=self.dtype))
        x1arr = self.SimpleArray(array=np.array([-1, -2, -3],
                                                dtype=self.dtype))
        y1arr = self.SimpleArray(array=np.array([-4, -5, -6],
                                                dtype=self.dtype))
        z1arr = self.SimpleArray(array=np.array([-7, -8, -9],
                                                dtype=self.dtype))
        sp = self.SegmentPad(x0=x0arr, y0=y0arr, z0=z0arr,
                             x1=x1arr, y1=y1arr, z1=z1arr, clone=False)

        w = self.World()

        # Empty
        self.assertEqual(w.nsegment, 0)
        with self.assertRaisesRegex(
                IndexError, "World: \\(segment\\) i 0 >= size 0"):
            w.segment(0)
        # Add the SegmentPad
        w.add_segments(sp)
        self.assertEqual(w.nsegment, 3)

        s0 = w.segment(0)
        self.assert_allclose(list(s0[0]), [1, 4, 7])
        self.assert_allclose(list(s0[1]), [-1, -4, -7])
        s1 = w.segment(1)
        self.assert_allclose(list(s1[0]), [2, 5, 8])
        self.assert_allclose(list(s1[1]), [-2, -5, -8])
        s2 = w.segment(2)
        self.assert_allclose(list(s2[0]), [3, 6, 9])
        self.assert_allclose(list(s2[1]), [-3, -6, -9])


class WorldFp32TC(WorldTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float32'
        self.SimpleArray = solvcon.SimpleArrayFloat32
        self.Point = solvcon.Point3dFp32
        self.Segment = solvcon.Segment3dFp32
        self.Bezier = solvcon.Bezier3dFp32
        self.SegmentPad = solvcon.SegmentPadFp32
        self.CurvePad = solvcon.CurvePadFp32
        self.World = solvcon.WorldFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)

    def test_type(self):
        self.assertIs(solvcon.WorldFp32, self.World)


class WorldFp64TC(WorldTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float64'
        self.SimpleArray = solvcon.SimpleArrayFloat64
        self.Point = solvcon.Point3dFp64
        self.Segment = solvcon.Segment3dFp64
        self.Bezier = solvcon.Bezier3dFp64
        self.SegmentPad = solvcon.SegmentPadFp64
        self.CurvePad = solvcon.CurvePadFp64
        self.World = solvcon.WorldFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

    def test_type(self):
        self.assertIs(solvcon.WorldFp64, self.World)


class WorldShapeTC(unittest.TestCase):
    """Shape registry: add, translate, remove, clear."""

    def setUp(self):
        self.w = solvcon.WorldFp64()

    def test_add_triangle(self):
        sid = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.assertEqual(self.w.nshape, 1)
        self.assertEqual(self.w.nsegment, 3)
        self.assertEqual(self.w.shape_type_of(sid), "triangle")

    def test_add_multiple(self):
        self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.add_triangle(2, 2, 3, 2, 2, 3)
        self.assertEqual(self.w.nshape, 2)
        self.assertEqual(self.w.nsegment, 6)

    def test_ids_are_unique(self):
        s0 = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        s1 = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.assertNotEqual(s0, s1)

    def test_translate(self):
        sid = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.translate_shape(sid, 10, 20)
        seg = self.w.segment(0)
        self.assertAlmostEqual(seg.x0, 10.0)
        self.assertAlmostEqual(seg.y0, 20.0)
        self.assertAlmostEqual(seg.x1, 11.0)
        self.assertAlmostEqual(seg.y1, 20.0)

    def test_translate_isolates_shapes(self):
        s0 = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.add_triangle(0, 0, 2, 0, 0, 2)
        self.w.translate_shape(s0, 10, 10)
        self.assertAlmostEqual(self.w.segment(0).x0, 10.0)
        self.assertAlmostEqual(self.w.segment(3).x0, 0.0)

    def test_remove(self):
        sid = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.remove_shape(sid)
        self.assertEqual(self.w.nshape, 0)

    def test_remove_nonexistent_raises(self):
        with self.assertRaises(IndexError):
            self.w.remove_shape(999)

    def test_remove_dead_raises(self):
        sid = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.remove_shape(sid)
        with self.assertRaises(ValueError):
            self.w.remove_shape(sid)

    def test_translate_dead_raises(self):
        sid = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.remove_shape(sid)
        with self.assertRaises(ValueError):
            self.w.translate_shape(sid, 1, 1)

    def test_shape_type_of_dead_raises(self):
        sid = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.remove_shape(sid)
        with self.assertRaises(ValueError):
            self.w.shape_type_of(sid)

    def test_clear(self):
        self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.add_triangle(2, 2, 3, 2, 2, 3)
        self.w.add_segment(
            s=solvcon.Segment3dFp64(
                solvcon.Point3dFp64(0, 0),
                solvcon.Point3dFp64(1, 1),
            )
        )
        self.w.clear()
        self.assertEqual(self.w.nshape, 0)
        self.assertEqual(self.w.nsegment, 0)


class WorldPrimitivesTC(unittest.TestCase):
    """Polyline, polygon, and text primitives."""

    def setUp(self):
        self.w = solvcon.WorldFp64()

    def test_polyline_is_open(self):
        sid = self.w.add_polyline([[0, 0], [1, 0], [1, 1]])
        # An open chain of n vertices owns n-1 segments.
        self.assertEqual(self.w.shape_type_of(sid), "polyline")
        self.assertEqual(self.w.nsegment, 2)

    def test_polygon_is_closed(self):
        sid = self.w.add_polygon([[0, 0], [2, 0], [2, 2]])
        # A closed chain of n vertices owns n segments; the last returns home.
        self.assertEqual(self.w.shape_type_of(sid), "polygon")
        self.assertEqual(self.w.nsegment, 3)
        self.assertEqual(self.w.shape_bbox(sid), [0, 0, 2, 2])

    def test_polyline_needs_two_vertices(self):
        with self.assertRaises(ValueError):
            self.w.add_polyline([[0, 0]])

    def test_polygon_needs_three_vertices(self):
        with self.assertRaises(ValueError):
            self.w.add_polygon([[0, 0], [1, 1]])

    def test_text_anchor_and_height(self):
        sid = self.w.add_text("SOLVCON", -5, -5, 2.0)
        self.assertEqual(self.w.shape_type_of(sid), "text")
        # No segments or curves; the label is boxed from its anchor, up by the
        # height and down by a descender allowance (0.25 * height).
        self.assertEqual(self.w.nsegment, 0)
        bbox = self.w.shape_bbox(sid)
        self.assertAlmostEqual(bbox[0], -5.0)
        self.assertAlmostEqual(bbox[1], -5.5)  # anchor y - descender
        self.assertAlmostEqual(bbox[3], -3.0)  # anchor y + height

    def test_text_needs_positive_finite_height(self):
        # The direct binding guards like add_polyline/add_polygon; a zero or
        # non-finite height would render nothing and collapse the pick box.
        with self.assertRaises(ValueError):
            self.w.add_text("x", 0.0, 0.0, 0.0)
        with self.assertRaises(ValueError):
            self.w.add_text("x", 0.0, 0.0, float("nan"))

    def test_text_obb_stays_axis_aligned_after_rotate(self):
        # Glyphs never rotate, so a text label's oriented box must remain
        # axis-aligned; otherwise the selection box would tilt off the text.
        sid = self.w.add_text("hi", 0.0, 0.0, 2.0)
        self.w.rotate_shape(sid, math.pi / 3, 0.0, 0.0)
        obb = self.w.shape_obb(sid)  # TL, TR, BR, BL as x, y pairs
        self.assertAlmostEqual(obb[1], obb[3])  # TL.y == TR.y
        self.assertAlmostEqual(obb[5], obb[7])  # BR.y == BL.y
        self.assertAlmostEqual(obb[0], obb[6])  # TL.x == BL.x
        self.assertAlmostEqual(obb[2], obb[4])  # TR.x == BR.x


class WorldUndoRedoTC(unittest.TestCase):
    """Undo and redo of shape creation."""

    def setUp(self):
        self.w = solvcon.WorldFp64()

    def test_empty_world(self):
        self.assertFalse(self.w.can_undo)
        self.assertFalse(self.w.can_redo)
        # Undo/redo on an empty world are harmless no-ops.
        self.w.undo()
        self.w.redo()
        self.assertEqual(self.w.nshape, 0)

    def test_undo_removes_last_shape(self):
        self.w.add_circle(0, 0, 5)
        self.assertTrue(self.w.can_undo)
        self.assertFalse(self.w.can_redo)
        self.w.undo()
        self.assertEqual(self.w.nshape, 0)
        self.assertFalse(self.w.can_undo)
        self.assertTrue(self.w.can_redo)

    def test_redo_restores_shape(self):
        sid = self.w.add_circle(0, 0, 5)
        self.w.undo()
        self.w.redo()
        self.assertEqual(self.w.nshape, 1)
        self.assertEqual(self.w.shape_type_of(sid), "circle")
        self.assertTrue(self.w.can_undo)
        self.assertFalse(self.w.can_redo)

    def test_undo_redo_order(self):
        s0 = self.w.add_circle(0, 0, 1)
        s1 = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.undo()
        self.assertEqual(self.w.nshape, 1)
        self.assertEqual(self.w.shape_type_of(s0), "circle")
        with self.assertRaises(ValueError):
            self.w.shape_type_of(s1)
        self.w.undo()
        self.assertEqual(self.w.nshape, 0)

    def test_redo_restores_in_reverse(self):
        self.w.add_circle(0, 0, 1)
        self.w.add_square(0, 0, 2)
        self.w.undo()
        self.w.undo()
        self.w.redo()
        self.assertEqual(self.w.nshape, 1)
        self.w.redo()
        self.assertEqual(self.w.nshape, 2)

    def test_new_shape_clears_redo(self):
        self.w.add_circle(0, 0, 1)
        self.w.undo()
        self.assertTrue(self.w.can_redo)
        self.w.add_triangle(0, 0, 1, 0, 0, 1)
        # A fresh creation discards the redo history.
        self.assertFalse(self.w.can_redo)
        self.w.redo()
        self.assertEqual(self.w.nshape, 1)

    def test_undone_shape_not_visible(self):
        self.w.add_circle(0, 0, 1)
        self.w.undo()
        self.assertEqual(len(self.w.query_visible(-10, -10, 10, 10)), 0)
        self.w.redo()
        self.assertEqual(len(self.w.query_visible(-10, -10, 10, 10)), 1)

    def test_clear_resets_history(self):
        self.w.add_circle(0, 0, 1)
        self.w.undo()
        self.w.clear()
        self.assertFalse(self.w.can_undo)
        self.assertFalse(self.w.can_redo)


class WorldUndoRedoEditsTC(unittest.TestCase):
    """Undo and redo of edits beyond creation: a go-through that move, rotate,
    delete, and compound (drag) gestures undo and redo without raising."""

    def setUp(self):
        self.w = solvcon.WorldFp64()

    def test_edits_undo_redo_run_through(self):
        a = self.w.add_rectangle(-2, -1, 2, 1)
        b = self.w.add_circle(5, 5, 1)
        self.w.translate_shape(a, 3, 0)
        self.w.rotate_shape(a, math.pi / 2, 0, 0)
        self.w.remove_shape(b)
        self.assertEqual(self.w.nshape, 1)
        # A drag's many incremental moves collapse into one undo step: the
        # single undo below reverts both translates yet keeps the shape.
        self.w.begin_operation()
        self.w.translate_shape(a, 1, 0)
        self.w.translate_shape(a, 1, 0)
        self.w.end_operation()
        moved_x0 = self.w.segment(0).x0
        self.w.undo()
        self.assertNotAlmostEqual(self.w.segment(0).x0, moved_x0)
        self.assertEqual(self.w.nshape, 1)
        # Unwind every remaining change, then replay them all.
        while self.w.can_undo:
            self.w.undo()
        self.assertEqual(self.w.nshape, 0)
        while self.w.can_redo:
            self.w.redo()
        self.assertEqual(self.w.nshape, 1)
        self.assertEqual(self.w.shape_type_of(a), "rectangle")


class WorldViewportTC(unittest.TestCase):
    """R-tree spatial index and viewport query."""

    def setUp(self):
        self.w = solvcon.WorldFp64()

    def test_all_visible(self):
        self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.add_triangle(2, 2, 3, 2, 2, 3)
        ids = self.w.query_visible(-1, -1, 10, 10)
        self.assertEqual(len(ids), 2)

    def test_partial_visible(self):
        self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.add_triangle(100, 100, 101, 100, 100, 101)
        ids = self.w.query_visible(-1, -1, 2, 2)
        self.assertEqual(len(ids), 1)

    def test_none_visible(self):
        self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.assertEqual(len(self.w.query_visible(50, 50, 60, 60)), 0)

    def test_visible_after_translate(self):
        sid = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.translate_shape(sid, 200, 200)
        self.assertEqual(len(self.w.query_visible(-1, -1, 2, 2)), 0)
        self.assertEqual(
            len(self.w.query_visible(199, 199, 202, 202)), 1)

    def test_visible_after_remove(self):
        sid = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        self.w.remove_shape(sid)
        self.assertEqual(len(self.w.query_visible(-1, -1, 10, 10)), 0)


class WorldRotateTC(unittest.TestCase):
    """rotate_shape: go-through that rotation runs without raising."""

    def setUp(self):
        self.w = solvcon.WorldFp64()

    def test_rotate_runs(self):
        sid = self.w.add_rectangle(-2, -1, 2, 1)
        self.w.rotate_shape(sid, math.pi / 2, 0, 0)
        self.assertTrue(self.w.shape_is_live(sid))

    def test_rotate_dead_raises(self):
        sid = self.w.add_line(0, 0, 1, 0)
        self.w.remove_shape(sid)
        with self.assertRaises(ValueError):
            self.w.rotate_shape(sid, 1.0, 0, 0)


class WorldPickTC(unittest.TestCase):
    """pick_shape: go-through that hit-testing runs."""

    def setUp(self):
        self.w = solvcon.WorldFp64()

    def test_pick_runs(self):
        sid = self.w.add_rectangle(0, 0, 4, 4)
        self.assertEqual(self.w.pick_shape(2, 2, 0.1), sid)

    def test_pick_empty_world_returns_minus_one(self):
        self.assertEqual(self.w.pick_shape(0, 0, 1.0), -1)


class WorldShapeAccessorTC(unittest.TestCase):
    """shape_is_live, shape_bbox, shape_handle, shape_obb: go-through that
    each accessor runs and returns a sensibly-shaped result."""

    def setUp(self):
        self.w = solvcon.WorldFp64()
        self.sid = self.w.add_rectangle(-2, -1, 2, 1)

    def test_accessors_run(self):
        self.assertTrue(self.w.shape_is_live(self.sid))
        self.assertEqual(len(self.w.shape_bbox(self.sid)), 4)
        self.assertEqual(len(self.w.shape_handle(self.sid)), 2)
        self.assertEqual(len(self.w.shape_obb(self.sid)), 8)

    def test_dead_shape_not_live(self):
        self.w.remove_shape(self.sid)
        self.assertFalse(self.w.shape_is_live(self.sid))


class WorldLineTC(unittest.TestCase):
    """add_line: one segment per shape."""

    def setUp(self):
        self.w = solvcon.WorldFp64()

    def test_add_line(self):
        sid = self.w.add_line(0, 0, 3, 4)
        self.assertEqual(self.w.nshape, 1)
        self.assertEqual(self.w.nsegment, 1)
        self.assertEqual(self.w.shape_type_of(sid), "line")
        seg = self.w.segment(0)
        self.assertAlmostEqual(seg.x0, 0.0)
        self.assertAlmostEqual(seg.y0, 0.0)
        self.assertAlmostEqual(seg.x1, 3.0)
        self.assertAlmostEqual(seg.y1, 4.0)

    def test_translate_line(self):
        sid = self.w.add_line(0, 0, 1, 1)
        self.w.translate_shape(sid, 10, 20)
        seg = self.w.segment(0)
        self.assertAlmostEqual(seg.x0, 10.0)
        self.assertAlmostEqual(seg.y0, 20.0)
        self.assertAlmostEqual(seg.x1, 11.0)
        self.assertAlmostEqual(seg.y1, 21.0)

    def test_line_visible(self):
        self.w.add_line(0, 0, 1, 1)
        self.w.add_line(100, 100, 101, 101)
        self.assertEqual(len(self.w.query_visible(-1, -1, 2, 2)), 1)


class WorldRectangleTC(unittest.TestCase):
    """add_rectangle and specialized add_square."""

    def setUp(self):
        self.w = solvcon.WorldFp64()

    def test_add_rectangle(self):
        sid = self.w.add_rectangle(0, 0, 4, 2)
        self.assertEqual(self.w.nshape, 1)
        self.assertEqual(self.w.nsegment, 4)
        self.assertEqual(self.w.shape_type_of(sid), "rectangle")

    def test_rectangle_is_closed(self):
        self.w.add_rectangle(0, 0, 4, 2)
        # Segments form a closed loop: each endpoint shared with neighbour.
        segs = [self.w.segment(i) for i in range(4)]
        for cur, nxt in zip(segs, segs[1:] + segs[:1]):
            self.assertAlmostEqual(cur.x1, nxt.x0)
            self.assertAlmostEqual(cur.y1, nxt.y0)

    def test_add_square(self):
        sid = self.w.add_square(1, 1, 3)
        self.assertEqual(self.w.nshape, 1)
        self.assertEqual(self.w.nsegment, 4)
        self.assertEqual(self.w.shape_type_of(sid), "square")

    def test_translate_rectangle(self):
        sid = self.w.add_rectangle(0, 0, 4, 2)
        self.w.translate_shape(sid, 10, 20)
        seg = self.w.segment(0)
        self.assertAlmostEqual(seg.x0, 10.0)
        self.assertAlmostEqual(seg.y0, 20.0)
        self.assertAlmostEqual(seg.x1, 14.0)
        self.assertAlmostEqual(seg.y1, 20.0)

    def test_rectangle_visible(self):
        self.w.add_rectangle(0, 0, 1, 1)
        self.w.add_rectangle(100, 100, 101, 101)
        self.assertEqual(len(self.w.query_visible(-1, -1, 2, 2)), 1)


class WorldEllipseTC(unittest.TestCase):
    """add_ellipse and specialized add_circle."""

    def setUp(self):
        self.w = solvcon.WorldFp64()

    def test_add_ellipse(self):
        sid = self.w.add_ellipse(0, 0, 2, 1)
        self.assertEqual(self.w.nshape, 1)
        # Ellipse owns 4 cubic Beziers (one per quadrant) and no segments.
        self.assertEqual(self.w.nbezier, 4)
        self.assertEqual(self.w.nsegment, 0)
        self.assertEqual(self.w.shape_type_of(sid), "ellipse")

    def test_add_circle(self):
        sid = self.w.add_circle(0, 0, 5)
        self.assertEqual(self.w.nshape, 1)
        self.assertEqual(self.w.nbezier, 4)
        self.assertEqual(self.w.nsegment, 0)
        self.assertEqual(self.w.shape_type_of(sid), "circle")

    def test_ellipse_is_closed(self):
        # Each quadrant's p3 must match the next quadrant's p0.
        self.w.add_ellipse(0, 0, 3, 2)
        for i in range(4):
            cur = self.w.bezier(i)
            nxt = self.w.bezier((i + 1) % 4)
            self.assertAlmostEqual(cur[3][0], nxt[0][0])
            self.assertAlmostEqual(cur[3][1], nxt[0][1])

    def test_ellipse_anchor_points(self):
        # The 4 anchor points (p0 of each quadrant) sit at the ellipse's
        # compass points.
        cx, cy, rx, ry = 5.0, -3.0, 4.0, 2.0
        self.w.add_ellipse(cx, cy, rx, ry)
        anchors = [list(self.w.bezier(i)[0]) for i in range(4)]
        expected = [
            [cx + rx, cy, 0],
            [cx, cy + ry, 0],
            [cx - rx, cy, 0],
            [cx, cy - ry, 0],
        ]
        for got, want in zip(anchors, expected):
            for a, b in zip(got, want):
                self.assertAlmostEqual(a, b, places=12)

    def test_translate_circle(self):
        sid = self.w.add_circle(0, 0, 1)
        self.w.translate_shape(sid, 10, 0)
        # Translated center is (10, 0); every control point should shift.
        b0 = self.w.bezier(0)
        self.assertAlmostEqual(b0[0][0], 11.0)
        self.assertAlmostEqual(b0[0][1], 0.0)
        b2 = self.w.bezier(2)
        self.assertAlmostEqual(b2[0][0], 9.0)
        self.assertAlmostEqual(b2[0][1], 0.0)

    def test_circle_is_ellipse_with_equal_radii(self):
        w2 = solvcon.WorldFp64()
        self.w.add_circle(1, 2, 3)
        w2.add_ellipse(1, 2, 3, 3)
        for i in range(4):
            a = self.w.bezier(i)
            b = w2.bezier(i)
            for j in range(4):
                for k in range(3):
                    self.assertAlmostEqual(a[j][k], b[j][k], places=12)

    def test_circle_visible(self):
        self.w.add_circle(0, 0, 1)
        self.w.add_circle(100, 100, 1)
        self.assertEqual(len(self.w.query_visible(-2, -2, 2, 2)), 1)

    def test_ellipse_visible_after_translate(self):
        sid = self.w.add_ellipse(0, 0, 1, 1)
        self.w.translate_shape(sid, 200, 200)
        self.assertEqual(len(self.w.query_visible(-2, -2, 2, 2)), 0)
        self.assertEqual(len(self.w.query_visible(198, 198, 202, 202)), 1)

    def test_remove_ellipse(self):
        sid = self.w.add_ellipse(0, 0, 1, 1)
        self.w.remove_shape(sid)
        self.assertEqual(self.w.nshape, 0)
        self.assertEqual(len(self.w.query_visible(-2, -2, 2, 2)), 0)


class WorldBezierShapeTC(unittest.TestCase):
    """add_bezier_shape: one cubic Bezier per shape (curve counterpart of
    add_line). The bare add_bezier leaves the curve owned by no shape."""

    def setUp(self):
        self.w = solvcon.WorldFp64()
        self.Point = solvcon.Point3dFp64
        self.Bezier = solvcon.Bezier3dFp64

    def test_add_bezier_shape(self):
        p = self.Point
        sid = self.w.add_bezier_shape(p(0, 0, 0), p(1, 0, 0),
                                      p(2, 1, 0), p(3, 1, 0))
        self.assertEqual(self.w.nshape, 1)
        # The shape owns one cubic Bezier and no segments.
        self.assertEqual(self.w.nbezier, 1)
        self.assertEqual(self.w.nsegment, 0)
        self.assertEqual(self.w.shape_type_of(sid), "bezier")

    def test_add_bezier_shape_from_bezier(self):
        p = self.Point
        b = self.Bezier(p0=p(0, 0, 0), p1=p(1, 0, 0),
                        p2=p(2, 1, 0), p3=p(3, 1, 0))
        sid = self.w.add_bezier_shape(b=b)
        self.assertEqual(self.w.shape_type_of(sid), "bezier")
        self.assertEqual(self.w.nbezier, 1)

    def test_bare_bezier_is_not_a_shape(self):
        p = self.Point
        self.w.add_bezier(p(0, 0, 0), p(1, 0, 0), p(2, 1, 0), p(3, 1, 0))
        self.assertEqual(self.w.nshape, 0)
        self.assertEqual(self.w.nbezier, 1)

    def test_translate_bezier_shape(self):
        p = self.Point
        sid = self.w.add_bezier_shape(p(0, 0, 0), p(1, 0, 0),
                                      p(2, 1, 0), p(3, 1, 0))
        self.w.translate_shape(sid, 10, 20)
        b = self.w.bezier(0)
        self.assertAlmostEqual(b[0][0], 10.0)
        self.assertAlmostEqual(b[0][1], 20.0)
        self.assertAlmostEqual(b[3][0], 13.0)
        self.assertAlmostEqual(b[3][1], 21.0)

    def test_bezier_shape_visible(self):
        p = self.Point
        self.w.add_bezier_shape(p(0, 0, 0), p(1, 0, 0),
                                p(2, 1, 0), p(3, 1, 0))
        self.w.add_bezier_shape(p(100, 100, 0), p(101, 100, 0),
                                p(102, 101, 0), p(103, 101, 0))
        self.assertEqual(len(self.w.query_visible(-1, -1, 5, 5)), 1)

    def test_remove_bezier_shape(self):
        p = self.Point
        sid = self.w.add_bezier_shape(p(0, 0, 0), p(1, 0, 0),
                                      p(2, 1, 0), p(3, 1, 0))
        self.w.remove_shape(sid)
        self.assertEqual(self.w.nshape, 0)
        self.assertEqual(len(self.w.query_visible(-1, -1, 5, 5)), 0)


class WorldDescribeStateTC(unittest.TestCase):
    """describe_state(level="basic"): JSON serialization of visible state."""

    def setUp(self):
        self.w = solvcon.WorldFp64()

    def test_empty_world(self):
        # Locks the schema shape and the default level.
        self.assertEqual(
            self.w.describe_state(),
            '{"shapes":[],"segments":[],"curves":[],"points":[]}')

    def test_triangle_segments(self):
        sid = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        state = json.loads(self.w.describe_state(level="basic"))
        self.assertEqual(len(state["shapes"]), 1)
        shape = state["shapes"][0]
        self.assertEqual(shape["id"], sid)
        self.assertEqual(shape["type"], "triangle")
        self.assertEqual(shape["bbox"], [0, 0, 1, 1])
        self.assertEqual(shape["segments"],
                         [[0, 0, 1, 0], [1, 0, 0, 1], [0, 1, 0, 0]])
        self.assertEqual(shape["curves"], [])

    def test_circle_curves(self):
        self.w.add_circle(0, 0, 1)
        shape = json.loads(self.w.describe_state())["shapes"][0]
        self.assertEqual(shape["type"], "circle")
        self.assertEqual(shape["bbox"], [-1, -1, 1, 1])
        self.assertEqual(shape["segments"], [])
        self.assertEqual(len(shape["curves"]), 4)
        for curve in shape["curves"]:
            self.assertEqual(len(curve), 4)  # four control points
            for ctrl in curve:
                self.assertEqual(len(ctrl), 2)  # 2D: x, y only
        # First quadrant sweeps from (1, 0) to (0, 1).
        self.assertEqual(shape["curves"][0][0], [1, 0])
        self.assertEqual(shape["curves"][0][3], [0, 1])

    def test_bezier_shape_curves(self):
        p = solvcon.Point3dFp64
        sid = self.w.add_bezier_shape(p(0, 0, 0), p(1, 0, 0),
                                      p(2, 1, 0), p(3, 1, 0))
        state = json.loads(self.w.describe_state())
        shape = state["shapes"][0]
        self.assertEqual(shape["id"], sid)
        self.assertEqual(shape["type"], "bezier")
        self.assertEqual(shape["segments"], [])
        self.assertEqual(len(shape["curves"]), 1)
        self.assertEqual(shape["curves"][0],
                         [[0, 0], [1, 0], [2, 1], [3, 1]])
        # The shape owns its curve, so it is not also a bare top-level curve.
        self.assertEqual(state["curves"], [])

    def test_bare_segment_and_curve(self):
        # add_segment / add_bezier create geometry no shape owns, but it
        # still renders, so it appears under top-level segments/curves.
        p = solvcon.Point3dFp64
        self.w.add_segment(p(0, 0), p(1, 2))
        self.w.add_bezier(p(0, 0), p(1, 0), p(2, 1), p(3, 1))
        state = json.loads(self.w.describe_state())
        self.assertEqual(state["shapes"], [])
        self.assertEqual(state["segments"], [[0, 0, 1, 2]])
        self.assertEqual(len(state["curves"]), 1)
        self.assertEqual(state["curves"][0][0], [0, 0])
        self.assertEqual(state["curves"][0][3], [3, 1])

    def test_shape_curves_are_not_double_reported(self):
        # A circle's curves belong to the shape; they must not also appear as
        # bare top-level curves.
        self.w.add_circle(0, 0, 1)
        state = json.loads(self.w.describe_state())
        self.assertEqual(len(state["shapes"][0]["curves"]), 4)
        self.assertEqual(state["curves"], [])

    def test_coordinate_precision(self):
        # describe_state serializes through the shared JSON tool, whose
        # number format keeps six decimal places.
        self.w.add_point(1.0 / 3.0, 2.5, 0)
        pt = json.loads(self.w.describe_state())["points"][0]
        self.assertAlmostEqual(pt[0], 0.333333, places=6)
        self.assertEqual(pt[1], 2.5)

    def test_free_points(self):
        self.w.add_point(1, 2, 3)
        self.w.add_point(-4, 5, 6)
        state = json.loads(self.w.describe_state())
        # z is not rendered, so only x, y appear.
        self.assertEqual(state["points"], [[1, 2], [-4, 5]])

    def test_dead_shapes_excluded(self):
        keep = self.w.add_triangle(0, 0, 1, 0, 0, 1)
        drop = self.w.add_circle(10, 10, 1)
        self.w.remove_shape(drop)
        state = json.loads(self.w.describe_state())
        ids = [s["id"] for s in state["shapes"]]
        self.assertEqual(ids, [keep])
        # The dead circle does not render, so its curves must not leak into
        # the bare arrays either.
        self.assertEqual(state["curves"], [])
        self.assertEqual(state["segments"], [])

    def test_deterministic(self):
        def build(w):
            w.add_circle(0, 0, 1)
            w.add_rectangle(-2, -2, 2, 2)
            w.add_point(1, 1, 0)
        build(self.w)
        other = solvcon.WorldFp64()
        build(other)
        # Stable across repeated calls and equal across equal worlds.
        self.assertEqual(self.w.describe_state(), self.w.describe_state())
        self.assertEqual(self.w.describe_state(), other.describe_state())

    def test_unknown_level_raises(self):
        self.w.add_triangle(0, 0, 1, 0, 0, 1)
        with self.assertRaises(ValueError):
            self.w.describe_state(level="bogus")

    def test_fp32(self):
        w = solvcon.WorldFp32()
        w.add_line(0, 0, 1, 1)
        shape = json.loads(w.describe_state())["shapes"][0]
        self.assertEqual(shape["type"], "line")
        self.assertEqual(shape["segments"], [[0, 0, 1, 1]])


class WorldDescribeDiagnosticsTC(unittest.TestCase):
    """describe_state(level="diagnostics"): derived facts."""

    def setUp(self):
        self.w = solvcon.WorldFp64()

    def diag(self):
        state = json.loads(self.w.describe_state(level="diagnostics"))
        return state["diagnostics"]

    def assert_one_degeneracy(self, shape, kind, reason):
        degs = self.diag()["degeneracies"]
        self.assertEqual(
            degs, [{"shape": shape, "type": kind, "reason": reason}])

    def test_basic_omits_diagnostics(self):
        # Crossing geometry must not leak diagnostics into the basic level.
        self.w.add_line(0, 0, 2, 2)
        self.w.add_line(0, 2, 2, 0)
        self.assertNotIn("diagnostics", json.loads(self.w.describe_state()))

    def test_diagnostics_is_superset_of_basic(self):
        self.w.add_triangle(0, 0, 1, 0, 0, 1)
        basic = json.loads(self.w.describe_state(level="basic"))
        full = json.loads(self.w.describe_state(level="diagnostics"))
        for key in ("shapes", "segments", "curves", "points"):
            self.assertEqual(full[key], basic[key])
        self.assertIn("diagnostics", full)

    def test_clean_world_empty_diagnostics(self):
        self.w.add_triangle(0, 0, 4, 0, 0, 3)
        self.w.add_circle(10, 10, 1)
        diag = self.diag()
        self.assertEqual(diag["intersections"], [])
        self.assertEqual(diag["degeneracies"], [])

    def test_crossing_lines(self):
        a = self.w.add_line(0, 0, 2, 2)
        b = self.w.add_line(0, 2, 2, 0)
        hits = self.diag()["intersections"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["shapes"], [a, b])
        self.assertEqual(hits[0]["point"], [1, 1])

    def test_parallel_lines_no_crossing(self):
        self.w.add_line(0, 0, 2, 0)
        self.w.add_line(0, 1, 2, 1)
        self.assertEqual(self.diag()["intersections"], [])

    def test_shared_vertices_not_reported(self):
        # A triangle's edges meet only at its vertices (endpoints), which are
        # not proper crossings.
        self.w.add_triangle(0, 0, 4, 0, 0, 3)
        self.assertEqual(self.diag()["intersections"], [])

    def test_touching_endpoint_not_reported(self):
        # An endpoint resting on another segment's interior (a T-junction) is
        # endpoint contact, not a proper crossing.
        self.w.add_line(0, 0, 2, 0)
        self.w.add_line(1, 0, 1, 2)
        self.assertEqual(self.diag()["intersections"], [])

    def test_line_through_triangle(self):
        # The classic invisible bug: a line slices two triangle edges.
        tri = self.w.add_triangle(0, 0, 4, 0, 0, 4)
        line = self.w.add_line(-1, 1, 5, 1)
        hits = self.diag()["intersections"]
        self.assertEqual(len(hits), 2)
        for hit in hits:
            self.assertEqual(hit["shapes"], [tri, line])
        points = sorted(hit["point"] for hit in hits)
        self.assertEqual(points, [[0, 1], [3, 1]])

    def test_bare_segments_crossing(self):
        p = solvcon.Point3dFp64
        self.w.add_segment(p(0, 0), p(2, 2))
        self.w.add_segment(p(0, 2), p(2, 0))
        hits = self.diag()["intersections"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["shapes"], [-1, -1])
        self.assertEqual(hits[0]["point"], [1, 1])

    def test_removed_shape_excluded(self):
        self.w.add_line(0, 0, 2, 2)
        drop = self.w.add_line(0, 2, 2, 0)
        self.w.remove_shape(drop)
        self.assertEqual(self.diag()["intersections"], [])

    def test_undone_shape_excluded(self):
        # An undone shape is dead, so it drops out of the diagnostics, and
        # redo brings it and its crossing back. This matches how the basic
        # level treats dead shapes, now against undo/redo.
        self.w.add_line(0, 0, 2, 2)
        self.w.add_line(0, 2, 2, 0)
        self.assertEqual(len(self.diag()["intersections"]), 1)
        self.w.undo()
        self.assertEqual(self.diag()["intersections"], [])
        self.w.redo()
        self.assertEqual(len(self.diag()["intersections"]), 1)

    def test_collinear_triangle(self):
        sid = self.w.add_triangle(0, 0, 1, 1, 2, 2)
        self.assert_one_degeneracy(sid, "triangle", "collinear")

    def test_zero_radius_circle(self):
        sid = self.w.add_circle(0, 0, 0)
        self.assert_one_degeneracy(sid, "circle", "zero-radius")

    def test_zero_area_rectangle(self):
        sid = self.w.add_rectangle(0, 0, 2, 0)
        self.assert_one_degeneracy(sid, "rectangle", "zero-area")

    def test_zero_length_line(self):
        sid = self.w.add_line(1, 1, 1, 1)
        self.assert_one_degeneracy(sid, "line", "zero-length")

    def test_coincident_bezier(self):
        p = solvcon.Point3dFp64
        sid = self.w.add_bezier_shape(p(1, 1, 0), p(1, 1, 0),
                                      p(1, 1, 0), p(1, 1, 0))
        self.assert_one_degeneracy(sid, "bezier", "coincident-controls")

    def test_bare_zero_length_segment(self):
        p = solvcon.Point3dFp64
        self.w.add_segment(p(3, 3), p(3, 3))
        self.assert_one_degeneracy(-1, "segment", "zero-length")

    def test_healthy_shapes_no_degeneracies(self):
        self.w.add_triangle(0, 0, 4, 0, 0, 3)
        self.w.add_circle(10, 10, 2)
        self.w.add_rectangle(-5, -5, -1, -2)
        self.assertEqual(self.diag()["degeneracies"], [])

    def test_fractional_crossing_point(self):
        # A crossing whose point has fractional coordinates and whose two
        # parameters differ (t=1/6, s=1/2), guarding the 6-decimal serializer
        # and a t-vs-s mixup in the point computation.
        a = self.w.add_line(0, 0, 3, 3)
        b = self.w.add_line(0, 1, 1, 0)
        hits = self.diag()["intersections"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["shapes"], [a, b])
        self.assertEqual(hits[0]["point"], [0.5, 0.5])

    def test_shape_crossing_bare_segment(self):
        # A mixed pair: the slots carry the live shape id and the bare -1.
        p = solvcon.Point3dFp64
        sid = self.w.add_line(0, 0, 2, 2)
        self.w.add_segment(p(0, 2), p(2, 0))
        hits = self.diag()["intersections"]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["shapes"], [sid, -1])
        self.assertEqual(hits[0]["point"], [1, 1])

    def test_near_miss_not_crossing(self):
        # The vertical segment stops just short of the horizontal one, so the
        # near parameter falls outside (0, 1) and nothing is reported.
        self.w.add_line(0, 0, 2, 0)
        self.w.add_line(1, 0.0001, 1, 2)
        self.assertEqual(self.diag()["intersections"], [])

    def test_collinear_overlap_not_crossing(self):
        # Two segments on the same line overlap but share no single crossing
        # point, so the collinear short-circuit reports nothing.
        self.w.add_line(0, 0, 4, 0)
        self.w.add_line(2, 0, 6, 0)
        self.assertEqual(self.diag()["intersections"], [])

    def test_zero_one_axis_ellipse(self):
        sid = self.w.add_ellipse(0, 0, 0, 5)
        self.assert_one_degeneracy(sid, "ellipse", "zero-radius")

    def test_healthy_ellipse_no_degeneracy(self):
        self.w.add_ellipse(0, 0, 3, 1)
        self.assertEqual(self.diag()["degeneracies"], [])

    def test_zero_area_square(self):
        sid = self.w.add_square(0, 0, 0)
        self.assert_one_degeneracy(sid, "square", "zero-area")

    def test_multiple_degeneracies_ordered(self):
        # Documented order: live shapes by id, then bare segments, then bare
        # curves.
        p = solvcon.Point3dFp64
        tri = self.w.add_triangle(0, 0, 1, 1, 2, 2)
        cir = self.w.add_circle(5, 5, 0)
        self.w.add_segment(p(8, 8), p(8, 8))
        self.w.add_bezier(p(9, 9), p(9, 9), p(9, 9), p(9, 9))
        self.assertEqual(self.diag()["degeneracies"], [
            {"shape": tri, "type": "triangle", "reason": "collinear"},
            {"shape": cir, "type": "circle", "reason": "zero-radius"},
            {"shape": -1, "type": "segment", "reason": "zero-length"},
            {"shape": -1, "type": "bezier",
             "reason": "coincident-controls"},
        ])

    def test_deterministic(self):
        def build(w):
            w.add_line(0, 0, 2, 2)
            w.add_line(0, 2, 2, 0)
            w.add_circle(5, 5, 0)
        build(self.w)
        other = solvcon.WorldFp64()
        build(other)
        self.assertEqual(
            self.w.describe_state(level="diagnostics"),
            self.w.describe_state(level="diagnostics"))
        self.assertEqual(
            self.w.describe_state(level="diagnostics"),
            other.describe_state(level="diagnostics"))

    def test_fp32(self):
        w = solvcon.WorldFp32()
        w.add_line(0, 0, 2, 2)
        w.add_line(0, 2, 2, 0)
        diag = json.loads(
            w.describe_state(level="diagnostics"))["diagnostics"]
        self.assertEqual(len(diag["intersections"]), 1)
        self.assertEqual(diag["intersections"][0]["point"], [1, 1])

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
