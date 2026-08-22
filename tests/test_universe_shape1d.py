# Copyright (c) 2023, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

import numpy as np

import solvcon
from solvcon import testing


class Segment3dTB(testing.TestBase):

    def test_construct(self):
        Point = self.Point
        Segment = self.Segment

        s = Segment(p0=Point(x=0, y=0, z=0), p1=Point(x=1, y=1, z=1))
        self.assertEqual(len(s), 2)
        self.assertEqual(tuple(s.p0), (0.0, 0.0, 0.0))
        self.assertEqual(tuple(s.p1), (1.0, 1.0, 1.0))

        # Test six-scalar construcotr
        s = Segment(p0=Point(x=1, y=2, z=3), p1=Point(x=4, y=5, z=6))
        s_6scalar = Segment(x0=1, y0=2, z0=3, x1=4, y1=5, z1=6)
        self.assertEqual(s_6scalar, s)
        s_6scalar = Segment(1, 2, 3, 4, 5, 6)
        self.assertEqual(s_6scalar, s)

        # Test four-scalar construcotr
        s = Segment(p0=Point(x=1, y=2), p1=Point(x=4, y=5))
        s_4scalar = Segment(x0=1, y0=2, x1=4, y1=5)
        self.assertEqual(s_4scalar, s)
        s_4scalar = Segment(1, 2, 4, 5)
        self.assertEqual(s_4scalar, s)

        s.p0 = Point(x=3, y=7, z=0)
        s.p1 = Point(x=-1, y=-4, z=9)
        self.assertEqual(s.x0, 3)
        self.assertEqual(s.y0, 7)
        self.assertEqual(s.z0, 0)
        self.assertEqual(s.x1, -1)
        self.assertEqual(s.y1, -4)
        self.assertEqual(s.z1, 9)

        s = Segment(Point(x=3.1, y=7.4, z=0.6), Point(x=-1.2, y=-4.1, z=9.2))
        self.assert_allclose(tuple(s.p0), (3.1, 7.4, 0.6))
        self.assert_allclose(tuple(s.p1), (-1.2, -4.1, 9.2))

    def test_mirror(self):
        Point = self.Point
        Segment = self.Segment

        s1 = Segment(Point(1, 2, 3), Point(4, 5, 6))
        s1.mirror('x')
        self.assertEqual(list(s1.p0), [1, -2, -3])
        self.assertEqual(list(s1.p1), [4, -5, -6])

        s2 = Segment(Point(1, 2, 3), Point(4, 5, 6))
        s2.mirror('y')
        self.assertEqual(list(s2.p0), [-1, 2, -3])
        self.assertEqual(list(s2.p1), [-4, 5, -6])

        s3 = Segment(Point(1, 2, 3), Point(4, 5, 6))
        s3.mirror('z')
        self.assertEqual(list(s3.p0), [-1, -2, 3])
        self.assertEqual(list(s3.p1), [-4, -5, 6])

        s4 = Segment(Point(1, 2, 3), Point(4, 5, 6))
        s4.mirror('Y')
        self.assertEqual(list(s4.p0), [-1, 2, -3])
        self.assertEqual(list(s4.p1), [-4, 5, -6])

        with self.assertRaisesRegex(
                ValueError,
                "Segment3d::mirror: axis must be 'x', 'y', or 'z'"):
            Segment(Point(1, 2, 3), Point(4, 5, 6)).mirror('w')

    def test_calc_length(self):
        Point = self.Point
        Segment = self.Segment

        s = Segment(Point(1, 2, 3), Point(4, 6, 3))
        self.assertEqual(s.calc_length2(), 25.0)
        self.assertEqual(s.calc_length(), 5.0)
        self.assertEqual(
            Segment(Point(1, 2, 3), Point(1, 2, 3)).calc_length(), 0.0)

    def test_direction(self):
        Point = self.Point
        Segment = self.Segment

        s = Segment(Point(1, 2, 0), Point(4, 6, 0))
        self.assert_allclose(tuple(s.direction()), (0.6, 0.8, 0.0))
        # A vertical and a horizontal segment stay finite: the direction
        # is normalized instead of dividing a slope.
        vertical = Segment(Point(2, -1, 0), Point(2, 9, 0))
        self.assertEqual(tuple(vertical.direction()), (0.0, 1.0, 0.0))
        horizontal = Segment(Point(3, 5, 0), Point(-7, 5, 0))
        self.assertEqual(tuple(horizontal.direction()), (-1.0, 0.0, 0.0))
        # A zero-length segment regularizes to the zero vector.
        zero = Segment(Point(1, 2, 3), Point(1, 2, 3))
        self.assertEqual(tuple(zero.direction()), (0.0, 0.0, 0.0))

    def test_direction_is_unit_at_any_scale(self):
        Point = self.Point
        Segment = self.Segment

        # Scaling by the largest difference keeps the direction a unit
        # vector at any magnitude; the unscaled factor degraded from
        # around 1e-4 down in single precision and overflowed to nan for
        # a length beyond around 2e19.
        for length in (1e-8, 1e-6, 1e-4, 1.0, 1e4, 1e20):
            with self.subTest(length=length):
                s = Segment(Point(0, 0, 0), Point(length, 0, 0))
                self.assertEqual(tuple(s.direction()), (1.0, 0.0, 0.0))
        # A mixed huge segment must not overflow the squared sum.
        wide = Segment(Point(0, 0, 0), Point(1e20, 1e20, 0))
        self.assert_allclose(tuple(wide.direction()),
                             (2.0 ** -0.5, 2.0 ** -0.5, 0.0))

    def test_normal_by_axis(self):
        Point = self.Point
        Segment = self.Segment

        s = Segment(Point(0, 0, 0), Point(10, 0, 0))
        self.assertEqual(tuple(s.normal_by_axis()), (0.0, -1.0, 0.0))
        vertical = Segment(Point(0, 0, 0), Point(0, 10, 0))
        self.assertEqual(tuple(vertical.normal_by_axis()), (1.0, 0.0, 0.0))
        slanted = Segment(Point(0, 0, 5), Point(3, 4, 5))
        self.assert_allclose(tuple(slanted.normal_by_axis()),
                             (0.8, -0.6, 0.0))

        # The reference axis picks the plane the normal lies in: the
        # normal is the direction crossed with the reference.
        along_y = Segment(Point(0, 0, 0), Point(0, 10, 0))
        self.assertEqual(tuple(along_y.normal_by_axis('x')),
                         (0.0, 0.0, -1.0))
        along_z = Segment(Point(0, 0, 0), Point(0, 0, 10))
        self.assertEqual(tuple(along_z.normal_by_axis('y')),
                         (-1.0, 0.0, 0.0))
        risen = Segment(Point(0, 0, 0), Point(3, 4, 5))
        self.assert_allclose(tuple(risen.normal_by_axis('z')),
                             (0.8, -0.6, 0.0))

        with self.assertRaisesRegex(
                ValueError,
                "Segment3d::normal_by_axis: "
                "segment is parallel to the reference axis"):
            along_z.normal_by_axis()
        # A zero-length segment has the zero direction, so its cross
        # product with any reference vanishes the same way.
        with self.assertRaisesRegex(
                ValueError,
                "Segment3d::normal_by_axis: "
                "segment is parallel to the reference axis"):
            Segment(Point(1, 2, 3), Point(1, 2, 3)).normal_by_axis()
        with self.assertRaisesRegex(
                ValueError,
                "Segment3d::normal_by_axis: axis must be 'x', 'y', or 'z'"):
            s.normal_by_axis('Z')

    def test_normal_by_axis_right_hand_rule(self):
        Point = self.Point
        Segment = self.Segment

        # The normal n, the direction t, and the reference a form a
        # right-handed frame: n x t = a, t x a = n, and a x n = t. A
        # length-2 segment keeps the direction exact, so the products are.
        axes = {'x': (1.0, 0.0, 0.0), 'y': (0.0, 1.0, 0.0),
                'z': (0.0, 0.0, 1.0)}
        cases = [('x', Point(0, 2, 0)), ('y', Point(0, 0, 2)),
                 ('z', Point(2, 0, 0))]
        for axis, endpoint in cases:
            with self.subTest(axis=axis):
                segment = Segment(Point(0, 0, 0), endpoint)
                n = tuple(segment.normal_by_axis(axis))
                t = tuple(segment.direction())
                a = axes[axis]
                self.assertEqual(tuple(np.cross(n, t)), a)
                self.assertEqual(tuple(np.cross(t, a)), n)
                self.assertEqual(tuple(np.cross(a, n)), t)

        # A slanted segment goes through the same rule; its direction is
        # rounded, so the products are compared to tolerance.
        slanted = Segment(Point(0, 0, 0), Point(3, 4, 0))
        n = tuple(slanted.normal_by_axis('z'))
        t = tuple(slanted.direction())
        self.assert_allclose(np.cross(n, t), (0.0, 0.0, 1.0), atol=1e-6)
        self.assert_allclose(np.cross(t, (0.0, 0.0, 1.0)), n, atol=1e-6)
        self.assert_allclose(np.cross((0.0, 0.0, 1.0), n), t, atol=1e-6)

    def test_offset_by_axis(self):
        Point = self.Point
        Segment = self.Segment

        # Positive on the side the normal points to, the right of the
        # direction: below a left-to-right segment.
        s = Segment(Point(0, 0, 0), Point(10, 0, 0))
        self.assertEqual(s.offset_by_axis(Point(5, 3, 0)), -3.0)
        self.assertEqual(s.offset_by_axis(Point(5, -2, 0)), 2.0)
        vertical = Segment(Point(0, 0, 0), Point(0, 10, 0))
        self.assertEqual(vertical.offset_by_axis(Point(2, 5, 0)), 2.0)
        slanted = Segment(Point(0, 0, 0), Point(3, 4, 0))
        self.assert_allclose(slanted.offset_by_axis(Point(0, 5, 0)), -3.0)
        self.assert_allclose(slanted.offset_by_axis(Point(3, 4, 0)), 0.0,
                             atol=1e-6)
        # A non-default reference axis reaches the internal normal: with
        # the x reference the offset of a y-running segment measures -z.
        along_y = Segment(Point(0, 0, 0), Point(0, 10, 0))
        self.assertEqual(along_y.offset_by_axis(Point(5, 3, 2), 'x'), -2.0)

    def test_point_along_axis(self):
        Point = self.Point
        Segment = self.Segment

        s = Segment(Point(0, 0, 0), Point(2, 4, 6))
        self.assertEqual(tuple(s.point_along_axis(2, 'y')), (1.0, 2.0, 3.0))
        self.assertEqual(tuple(s.point_along_axis(1, 'x')), (1.0, 2.0, 3.0))
        self.assertEqual(tuple(s.point_along_axis(3)), (1.0, 2.0, 3.0))
        # The value may sit outside the segment: the line reaches it.
        self.assertEqual(tuple(s.point_along_axis(-4, 'y')),
                         (-2.0, -4.0, -6.0))
        horizontal = Segment(Point(0, 5, 0), Point(10, 5, 0))
        with self.assertRaisesRegex(
                ValueError,
                "Segment3d::point_along_axis: "
                "the direction has no component along the axis"):
            horizontal.point_along_axis(7, 'y')
        # A zero-length segment has no component along any axis.
        with self.assertRaisesRegex(
                ValueError,
                "Segment3d::point_along_axis: "
                "the direction has no component along the axis"):
            Segment(Point(1, 2, 3), Point(1, 2, 3)).point_along_axis(7, 'y')
        # The axis is named in lower case only.
        for axis in ('w', 'Z'):
            with self.assertRaisesRegex(
                    ValueError,
                    "Segment3d::point_along_axis: "
                    "axis must be 'x', 'y', or 'z'"):
                horizontal.point_along_axis(7, axis)


class Segment3dFp32TC(Segment3dTB, unittest.TestCase):

    def setUp(self):
        self.Point = solvcon.Point3dFp32
        self.Segment = solvcon.Segment3dFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)

    def test_repr_str(self):
        s = solvcon.Segment3dFp32(solvcon.Point3dFp32(504.8, -64.2, 0),
                                  solvcon.Point3dFp32(421.4, -250.5, 0))
        golden = ("Segment3dFp32(Point3dFp32(504.8, -64.2, 0), "
                  "Point3dFp32(421.4, -250.5, 0))")
        # __repr__ is the same as __str__ for Segment3d
        self.assertEqual(repr(s), golden)
        self.assertEqual(str(s), golden)
        # Evaluate the string and test the result
        e = eval(golden, vars(solvcon))
        self.assertEqual(s, e)


class Segment3dFp64TC(Segment3dTB, unittest.TestCase):

    def setUp(self):
        self.Point = solvcon.Point3dFp64
        self.Segment = solvcon.Segment3dFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

    def test_repr_str(self):
        s = solvcon.Segment3dFp64(solvcon.Point3dFp64(504.8, -64.2, 0),
                                  solvcon.Point3dFp64(421.4, -250.5, 0))
        golden = ("Segment3dFp64(Point3dFp64(504.8, -64.2, 0), "
                  "Point3dFp64(421.4, -250.5, 0))")
        # __repr__ is the same as __str__ for Segment3d
        self.assertEqual(repr(s), golden)
        self.assertEqual(str(s), golden)
        # Evaluate the string and test the result
        e = eval(golden, vars(solvcon))
        self.assertEqual(s, e)


class Bezier3dTB(testing.TestBase):

    def test_control_points(self):
        Point = self.Point
        Bezier = self.Bezier

        # Create a cubic Bezier curve
        bzr = Bezier(p0=Point(0, 0, 0), p1=Point(1, 1, 0), p2=Point(3, 1, 0),
                     p3=Point(4, 0, 0))
        self.assertEqual(len(bzr), 4)
        self.assertEqual(list(bzr[0]), [0, 0, 0])
        self.assertEqual(list(bzr[1]), [1, 1, 0])
        self.assertEqual(list(bzr[2]), [3, 1, 0])
        self.assertEqual(list(bzr[3]), [4, 0, 0])

        # Test equality and inequality comparison operators
        bzr_copy = Bezier(p0=Point(0, 0, 0), p1=Point(1, 1, 0),
                          p2=Point(3, 1, 0), p3=Point(4, 0, 0))
        self.assertTrue(bzr_copy == bzr)

        bzr1 = Bezier(p0=Point(0, 0, 0), p1=Point(1, 1, 0), p2=Point(3, 1, 0),
                      p3=Point(4, 4, 4))
        self.assertTrue(bzr1 != bzr)

        # Range error in C++
        with self.assertRaisesRegex(IndexError,
                                    "Bezier3d: \\(control\\) i 4 >= size 4"):
            bzr[4]

    def test_locus_points(self):
        Point = self.Point
        Bezier = self.Bezier

        b = Bezier(p0=Point(0, 0, 0), p1=Point(1, 1, 0), p2=Point(3, 1, 0),
                   p3=Point(4, 0, 0))
        self.assertEqual(len(b), 4)

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

        segs = b.sample(nlocus=9)
        self.assertEqual(len(segs), 8)
        self.assert_allclose(
            list(segs[0]), [[0.0, 0.0, 0.0], [0.41796875, 0.328125, 0.0]])
        self.assert_allclose(
            list(segs[1]),
            [[0.41796875, 0.328125, 0.0], [0.90625, 0.5625, 0.0]])
        self.assert_allclose(
            list(segs[2]),
            [[0.90625, 0.5625, 0.0], [1.44140625, 0.703125, 0.0]])
        self.assert_allclose(
            list(segs[3]), [[1.44140625, 0.703125, 0.0], [2.0, 0.75, 0.0]])
        self.assert_allclose(
            list(segs[4]), [[2.0, 0.75, 0.0], [2.55859375, 0.703125, 0.0]])
        self.assert_allclose(
            list(segs[5]),
            [[2.55859375, 0.703125, 0.0], [3.09375, 0.5625, 0.0]])
        self.assert_allclose(
            list(segs[6]),
            [[3.09375, 0.5625, 0.0], [3.58203125, 0.328125, 0.0]])
        self.assert_allclose(
            list(segs[7]), [[3.58203125, 0.328125, 0.0], [4.0, 0.0, 0.0]])

    def test_mirror(self):
        Point = self.Point
        Bezier = self.Bezier

        b1 = Bezier(Point(0, 0, 0), Point(1, 1, 0),
                    Point(3, 1, 0), Point(4, 0, 0))
        b1.mirror('x')
        self.assertEqual(list(b1[0]), [0, 0, 0])
        self.assertEqual(list(b1[1]), [1, -1, 0])
        self.assertEqual(list(b1[2]), [3, -1, 0])
        self.assertEqual(list(b1[3]), [4, 0, 0])

        b2 = Bezier(Point(0, 0, 0), Point(1, 1, 0),
                    Point(3, 1, 0), Point(4, 0, 0))
        b2.mirror('y')
        self.assertEqual(list(b2[0]), [0, 0, 0])
        self.assertEqual(list(b2[1]), [-1, 1, 0])
        self.assertEqual(list(b2[2]), [-3, 1, 0])
        self.assertEqual(list(b2[3]), [-4, 0, 0])

        b3 = Bezier(Point(1, 2, 3), Point(4, 5, 6),
                    Point(7, 8, 9), Point(10, 11, 12))
        b3.mirror('z')
        self.assertEqual(list(b3[0]), [-1, -2, 3])
        self.assertEqual(list(b3[1]), [-4, -5, 6])
        self.assertEqual(list(b3[2]), [-7, -8, 9])
        self.assertEqual(list(b3[3]), [-10, -11, 12])

        b4 = Bezier(Point(1, 2, 3), Point(4, 5, 6),
                    Point(7, 8, 9), Point(10, 11, 12))
        b4.mirror('Z')
        self.assertEqual(list(b4[0]), [-1, -2, 3])

        with self.assertRaisesRegex(
                ValueError,
                "Bezier3d::mirror: axis must be 'x', 'y', or 'z'"):
            Bezier(Point(0, 0, 0), Point(1, 1, 0),
                   Point(3, 1, 0), Point(4, 0, 0)).mirror('w')


class Bezier3dFp32TC(Bezier3dTB, unittest.TestCase):

    def setUp(self):
        self.Point = solvcon.Point3dFp32
        self.Bezier = solvcon.Bezier3dFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)

    def test_repr_str(self):
        b = solvcon.Bezier3dFp32(
            solvcon.Point3dFp32(607.7, -64.2, 0),
            solvcon.Point3dFp32(504.8, -64.2, 0),
            solvcon.Point3dFp32(421.4, -147.6, 0),
            solvcon.Point3dFp32(421.4, -250.5, 0))
        golden = ("Bezier3dFp32(Point3dFp32(607.7, -64.2, 0), "
                  "Point3dFp32(504.8, -64.2, 0), "
                  "Point3dFp32(421.4, -147.6, 0), "
                  "Point3dFp32(421.4, -250.5, 0))")
        # __repr__ is the same as __str__ for Bezier3d
        self.assertEqual(repr(b), golden)
        self.assertEqual(str(b), golden)
        # Evaluate the string and test the result
        e = eval(golden, vars(solvcon))
        self.assertEqual(b, e)


class Bezier3dFp64TC(Bezier3dTB, unittest.TestCase):

    def setUp(self):
        self.Point = solvcon.Point3dFp64
        self.Bezier = solvcon.Bezier3dFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

    def test_repr_str(self):
        b = solvcon.Bezier3dFp64(
            solvcon.Point3dFp64(607.7, -64.2, 0),
            solvcon.Point3dFp64(504.8, -64.2, 0),
            solvcon.Point3dFp64(421.4, -147.6, 0),
            solvcon.Point3dFp64(421.4, -250.5, 0))
        golden = ("Bezier3dFp64(Point3dFp64(607.7, -64.2, 0), "
                  "Point3dFp64(504.8, -64.2, 0), "
                  "Point3dFp64(421.4, -147.6, 0), "
                  "Point3dFp64(421.4, -250.5, 0))")
        # __repr__ is the same as __str__ for Bezier3d
        self.assertEqual(repr(b), golden)
        self.assertEqual(str(b), golden)
        # Evaluate the string and test the result
        e = eval(golden, vars(solvcon))
        self.assertEqual(b, e)


class SegmentPadTB(testing.TestBase):

    def test_ndim(self):
        sp2d = self.SegmentPad(ndim=2)
        self.assertEqual(sp2d.ndim, 2)
        sp3d = self.SegmentPad(ndim=3)
        self.assertEqual(sp3d.ndim, 3)

        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 0 < 2"):
            self.SegmentPad(ndim=0)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 0 < 2"):
            self.SegmentPad(ndim=0, nelem=2)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 1 < 2"):
            self.SegmentPad(ndim=1)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 1 < 2"):
            self.SegmentPad(ndim=1, nelem=3)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 4 > 3"):
            self.SegmentPad(ndim=4)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 4 > 3"):
            self.SegmentPad(ndim=4, nelem=5)

    def test_construct_2d(self):
        x0arr = self.SimpleArray(array=np.array([1, 2, 3], dtype=self.dtype))
        y0arr = self.SimpleArray(array=np.array([4, 5, 6], dtype=self.dtype))
        x1arr = self.SimpleArray(array=np.array([-1, -2, -3],
                                                dtype=self.dtype))
        y1arr = self.SimpleArray(array=np.array([-4, -5, -6],
                                                dtype=self.dtype))
        sp = self.SegmentPad(x0=x0arr, y0=y0arr, x1=x1arr, y1=y1arr,
                             clone=False)
        self.assertEqual(sp.ndim, 2)
        self.assert_allclose(sp.x0, [1, 2, 3])
        self.assert_allclose(sp.y0, [4, 5, 6])
        self.assert_allclose(sp.x1, [-1, -2, -3])
        self.assert_allclose(sp.y1, [-4, -5, -6])
        self.assertEqual(len(sp.z0), 0)
        self.assertEqual(len(sp.z1), 0)

        # Test zero-copy writing
        sp.x0[1] = 200.2
        sp.y0[0] = -700.3
        sp.x1[1] = -200.2
        sp.y1[0] = 700.3
        self.assert_allclose(list(sp[0]), [[1, -700.3, 0], [-1, 700.3, 0]])
        self.assert_allclose(list(sp[1]), [[200.2, 5, 0], [-200.2, -5, 0]])
        self.assert_allclose(list(sp[2]), [[3, 6, 0], [-3, -6, 0]])

        sp2 = self.SegmentPad(ndim=2, nelem=3)
        for i in range(len(sp)):
            sp2.set_at(i, sp.get_at(i).x0, sp.get_at(i).y0,
                       sp.get_at(i).x1, sp.get_at(i).y1)
        self.assert_allclose(sp2.x0, [1, 200.2, 3])
        self.assert_allclose(sp2.y0, [-700.3, 5, 6])
        self.assert_allclose(sp2.x1, [-1, -200.2, -3])
        self.assert_allclose(sp2.y1, [700.3, -5, -6])
        self.assertEqual(len(sp2.z0), 0)
        self.assertEqual(len(sp2.z1), 0)

        packed = sp2.pack_array().ndarray
        self.assertEqual(packed.shape, (3, 4))
        self.assert_allclose(list(packed[0]), (1, -700.3, -1, 700.3))
        self.assert_allclose(list(packed[1]), (200.2, 5, -200.2, -5))
        self.assert_allclose(list(packed[2]), (3, 6, -3, -6))

    def test_construct_3d(self):
        Point = self.Point

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
        self.assertEqual(sp.ndim, 3)
        self.assert_allclose(sp.x0, [1, 2, 3])
        self.assert_allclose(sp.y0, [4, 5, 6])
        self.assert_allclose(sp.z0, [7, 8, 9])
        self.assert_allclose(sp.x1, [-1, -2, -3])
        self.assert_allclose(sp.y1, [-4, -5, -6])
        self.assert_allclose(sp.z1, [-7, -8, -9])

        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            sp.x0_at(3)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            sp.y1_at(3)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            sp.z0_at(3)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            sp.p0_at(3)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            sp.get_at(3)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            sp.set_at(3, self.Segment(Point(0, 0, 0), Point(0, 0, 0)))

        # Test zero-copy writing
        sp.x0[1] = 200.2
        sp.y0[0] = -700.3
        sp.z0[2] = 213.9
        sp.x1[1] = -200.2
        sp.y1[0] = 700.3
        sp.z1[2] = -213.9
        self.assert_allclose(list(sp[0]), [[1, -700.3, 7], [-1, 700.3, -7]])
        self.assert_allclose(list(sp[1]), [[200.2, 5, 8], [-200.2, -5, -8]])
        self.assert_allclose(list(sp[2]), [[3, 6, 213.9], [-3, -6, -213.9]])

        sp2 = self.SegmentPad(ndim=3, nelem=3)
        for i in range(len(sp)):
            sp2.set_at(i, sp.get_at(i).x0, sp.get_at(i).y0, sp.get_at(i).z0,
                       sp.get_at(i).x1, sp.get_at(i).y1, sp.get_at(i).z1)
        self.assert_allclose(sp2.x0, [1, 200.2, 3])
        self.assert_allclose(sp2.y0, [-700.3, 5, 6])
        self.assert_allclose(sp2.z0, [7, 8, 213.9])
        self.assert_allclose(sp2.x1, [-1, -200.2, -3])
        self.assert_allclose(sp2.y1, [700.3, -5, -6])
        self.assert_allclose(sp2.z1, [-7, -8, -213.9])

        packed = sp2.pack_array().ndarray
        self.assertEqual(packed.shape, (3, 6))
        self.assert_allclose(list(packed[0]), (1, -700.3, 7, -1, 700.3, -7))
        self.assert_allclose(list(packed[1]), (200.2, 5, 8, -200.2, -5, -8))
        self.assert_allclose(list(packed[2]), (3, 6, 213.9, -3, -6, -213.9))

    def test_append_2d(self):
        Point = self.Point
        Segment = self.Segment

        sp = self.SegmentPad(ndim=2)
        self.assertEqual(sp.ndim, 2)
        self.assertEqual(len(sp), 0)
        sp.append(Segment(Point(1.1, 2.2, 0.0), Point(7.1, 8.2, 0.0)))
        self.assertEqual(len(sp), 1)
        self.assert_allclose(sp.x0_at(0), 1.1)
        self.assert_allclose(sp.y0_at(0), 2.2)
        self.assert_allclose(sp.x1_at(0), 7.1)
        self.assert_allclose(sp.y1_at(0), 8.2)
        sp.append(Point(1.1 * 3, 2.2 * 3), Point(7.1 * 3, 8.2 * 3))
        self.assertEqual(len(sp), 2)
        self.assert_allclose(sp.x0_at(1), 1.1 * 3)
        self.assert_allclose(sp.y0_at(1), 2.2 * 3)
        self.assert_allclose(sp.x1_at(1), 7.1 * 3)
        self.assert_allclose(sp.y1_at(1), 8.2 * 3)
        sp.append(1.1 * 3.1, 2.2 * 3.1, 7.1 * 3.1, 8.2 * 3.1)
        self.assertEqual(len(sp), 3)
        self.assert_allclose(sp.x0_at(2), 1.1 * 3.1)
        self.assert_allclose(sp.y0_at(2), 2.2 * 3.1)
        self.assert_allclose(sp.x1_at(2), 7.1 * 3.1)
        self.assert_allclose(sp.y1_at(2), 8.2 * 3.1)

        with self.assertRaisesRegex(
                IndexError, "PointPad::append: ndim must be 3 but is 2"):
            sp.append(3.2, 4.1, 5.7, 3.2, 4.1, 5.7)
        self.assertEqual(len(sp), 3)

        # Test batch interface
        self.assert_allclose(sp.x0, [1.1, 1.1 * 3, 1.1 * 3.1])
        self.assert_allclose(sp.y0, [2.2, 2.2 * 3, 2.2 * 3.1])
        self.assert_allclose(sp.x1, [7.1, 7.1 * 3, 7.1 * 3.1])
        self.assert_allclose(sp.y1, [8.2, 8.2 * 3, 8.2 * 3.1])
        sp.x0[0] = -10.9
        sp.x0.ndarray[2] = -13.2
        sp.x1[0] = 10.9
        sp.x1.ndarray[2] = 13.2
        self.assert_allclose(sp.x0_at(0), -10.9)
        self.assert_allclose(sp.x0_at(1), 1.1 * 3)
        self.assert_allclose(sp.x0_at(2), -13.2)
        self.assert_allclose(sp.x1_at(0), 10.9)
        self.assert_allclose(sp.x1_at(1), 7.1 * 3)
        self.assert_allclose(sp.x1_at(2), 13.2)
        sp.y0[1] = -0.93
        sp.y0.ndarray[2] = 29.1
        sp.y1[1] = 0.93
        sp.y1.ndarray[2] = -29.1
        self.assert_allclose(sp.y0_at(0), 2.2)
        self.assert_allclose(sp.y0_at(1), -0.93)
        self.assert_allclose(sp.y0_at(2), 29.1)
        self.assert_allclose(sp.y1_at(0), 8.2)
        self.assert_allclose(sp.y1_at(1), 0.93)
        self.assert_allclose(sp.y1_at(2), -29.1)
        self.assertEqual(len(sp.z0), 0)
        self.assertEqual(len(sp.z1), 0)

        nseg = len(sp)
        sp.extend_with(sp)
        for i in range(nseg):
            self.assertEqual(sp[i], sp[nseg + i])

    def test_append_3d(self):
        Point = self.Point
        Segment = self.Segment

        sp = self.SegmentPad(ndim=3)
        self.assertEqual(sp.ndim, 3)
        self.assertEqual(len(sp), 0)
        sp.append(s=Segment(Point(1.1, 2.2, 3.3), Point(7.1, 8.2, 9.3)))
        self.assertEqual(len(sp), 1)
        self.assert_allclose(sp.x0_at(0), 1.1)
        self.assert_allclose(sp.y0_at(0), 2.2)
        self.assert_allclose(sp.z0_at(0), 3.3)
        self.assert_allclose(sp.x1_at(0), 7.1)
        self.assert_allclose(sp.y1_at(0), 8.2)
        self.assert_allclose(sp.z1_at(0), 9.3)
        sp.append(p0=Point(1.1 * 5, 2.2 * 5, 3.3 * 5),
                  p1=Point(7.1 * 5, 8.2 * 5, 9.3 * 5))
        self.assertEqual(len(sp), 2)
        self.assert_allclose(sp.x0_at(1), 1.1 * 5)
        self.assert_allclose(sp.y0_at(1), 2.2 * 5)
        self.assert_allclose(sp.z0_at(1), 3.3 * 5)
        self.assert_allclose(sp.x1_at(1), 7.1 * 5)
        self.assert_allclose(sp.y1_at(1), 8.2 * 5)
        self.assert_allclose(sp.z1_at(1), 9.3 * 5)
        sp.append(x0=1.1 * 5.1, y0=2.2 * 5.1, z0=3.3 * 5.1,
                  x1=7.1 * 5.1, y1=8.2 * 5.1, z1=9.3 * 5.1)
        self.assertEqual(len(sp), 3)
        self.assert_allclose(sp.x0_at(2), 1.1 * 5.1)
        self.assert_allclose(sp.y0_at(2), 2.2 * 5.1)
        self.assert_allclose(sp.z0_at(2), 3.3 * 5.1)
        self.assert_allclose(sp.x1_at(2), 7.1 * 5.1)
        self.assert_allclose(sp.y1_at(2), 8.2 * 5.1)
        self.assert_allclose(sp.z1_at(2), 9.3 * 5.1)

        with self.assertRaisesRegex(
                IndexError, "PointPad::append: ndim must be 2 but is 3"):
            sp.append(3.2, 4.1, 5.2, 6.2)
        self.assertEqual(len(sp), 3)

        # Test batch interface
        self.assert_allclose(sp.x0, [1.1, 1.1 * 5, 1.1 * 5.1])
        self.assert_allclose(sp.y0, [2.2, 2.2 * 5, 2.2 * 5.1])
        self.assert_allclose(sp.z0, [3.3, 3.3 * 5, 3.3 * 5.1])
        self.assert_allclose(sp.x1, [7.1, 7.1 * 5, 7.1 * 5.1])
        self.assert_allclose(sp.y1, [8.2, 8.2 * 5, 8.2 * 5.1])
        self.assert_allclose(sp.z1, [9.3, 9.3 * 5, 9.3 * 5.1])
        sp.x0[0] = -10.9
        sp.x0.ndarray[2] = -13.2
        sp.x1[0] = 10.9
        sp.x1.ndarray[2] = 13.2
        self.assert_allclose(sp.x0_at(0), -10.9)
        self.assert_allclose(sp.x0_at(1), 1.1 * 5)
        self.assert_allclose(sp.x0_at(2), -13.2)
        self.assert_allclose(sp.x1_at(0), 10.9)
        self.assert_allclose(sp.x1_at(1), 7.1 * 5)
        self.assert_allclose(sp.x1_at(2), 13.2)
        sp.y0[1] = -0.93
        sp.y0.ndarray[2] = 29.1
        sp.y1[1] = 0.93
        sp.y1.ndarray[2] = -29.1
        self.assert_allclose(sp.y0_at(0), 2.2)
        self.assert_allclose(sp.y0_at(1), -0.93)
        self.assert_allclose(sp.y0_at(2), 29.1)
        self.assert_allclose(sp.y1_at(0), 8.2)
        self.assert_allclose(sp.y1_at(1), 0.93)
        self.assert_allclose(sp.y1_at(2), -29.1)
        sp.z0[0] = 2.31
        sp.z0.ndarray[1] = 8.23
        sp.z1[0] = -2.31
        sp.z1.ndarray[1] = -8.23
        self.assert_allclose(sp.z0_at(0), 2.31)
        self.assert_allclose(sp.z0_at(1), 8.23)
        self.assert_allclose(sp.z0_at(2), 3.3 * 5.1)
        self.assert_allclose(sp.z1_at(0), -2.31)
        self.assert_allclose(sp.z1_at(1), -8.23)
        self.assert_allclose(sp.z1_at(2), 9.3 * 5.1)

        nseg = len(sp)
        sp.extend_with(sp)
        for i in range(nseg):
            self.assertEqual(sp[i], sp[nseg + i])

        # Assert the equality between value array and PointPad
        self.assert_allclose(list(sp.x0), list(sp.p0.x))
        self.assert_allclose(list(sp.y0), list(sp.p0.y))
        self.assert_allclose(list(sp.z0), list(sp.p0.z))
        self.assert_allclose(list(sp.x1), list(sp.p1.x))
        self.assert_allclose(list(sp.y1), list(sp.p1.y))
        self.assert_allclose(list(sp.z1), list(sp.p1.z))

    def test_getitem_index(self):
        sp = self.SegmentPad(ndim=2)
        for it in range(4):
            sp.append(float(it), 0.0, float(it) + 1, 1.0)

        self.assert_allclose(list(sp[0]), [[0, 0, 0], [1, 1, 0]])
        self.assertEqual(sp[-1], sp[3])
        self.assertEqual(sp[-4], sp[0])

        with self.assertRaisesRegex(
                IndexError,
                "SegmentPad: index 4 is out of bounds with size 4"):
            sp[4]
        with self.assertRaisesRegex(
                IndexError,
                "SegmentPad: index -5 is out of bounds with size 4"):
            sp[-5]

        empty = self.SegmentPad(ndim=3)
        with self.assertRaisesRegex(
                IndexError,
                "SegmentPad: index 0 is out of bounds with size 0"):
            empty[0]
        with self.assertRaisesRegex(
                IndexError,
                "SegmentPad: index -1 is out of bounds with size 0"):
            empty[-1]

    def test_getitem_slice(self):
        sp = self.SegmentPad(ndim=3)
        for it in range(4):
            sp.append(float(it), 0.0, 0.0, float(it) + 1, 1.0, 1.0)

        full = sp[:]
        self.assertIsInstance(full, type(sp))
        self.assertEqual(full.ndim, 3)
        self.assertEqual(len(full), 4)
        for it in range(4):
            self.assertEqual(full[it], sp[it])

        strided = sp[1::2]
        self.assertEqual(strided.ndim, 3)
        self.assertEqual(len(strided), 2)
        self.assertEqual(strided[0], sp[1])
        self.assertEqual(strided[1], sp[3])

        flipped = sp[::-1]
        self.assert_allclose(list(flipped.x0), [3, 2, 1, 0])

        # A slice copies, so writing to it leaves the source alone.
        part = sp[1:3]
        part.set_at(0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0)
        self.assert_allclose(part.x0_at(0), -1.0)
        self.assert_allclose(sp.x0_at(1), 1.0)

        # A 2D pad keeps its dimensionality and its empty z arrays.
        sp2d = self.SegmentPad(ndim=2)
        sp2d.append(1.0, 2.0, 3.0, 4.0)
        sliced2d = sp2d[:]
        self.assertEqual(sliced2d.ndim, 2)
        self.assertEqual(len(sliced2d.z0), 0)
        self.assertEqual(sliced2d[0], sp2d[0])

        empty = self.SegmentPad(ndim=3)
        self.assertEqual(len(empty[:]), 0)
        self.assertEqual(empty[:].ndim, 3)
        self.assertEqual(len(sp[3:1]), 0)

    def test_mirror_2d(self):
        SegmentPad = self.SegmentPad

        sp = SegmentPad(ndim=2)
        sp.append(1.0, 2.0, 3.0, 4.0)
        sp.append(5.0, 6.0, 7.0, 8.0)

        sp.mirror('x')
        self.assert_allclose(sp.x0_at(0), 1.0)
        self.assert_allclose(sp.y0_at(0), -2.0)
        self.assert_allclose(sp.x1_at(0), 3.0)
        self.assert_allclose(sp.y1_at(0), -4.0)
        self.assert_allclose(sp.x0_at(1), 5.0)
        self.assert_allclose(sp.y0_at(1), -6.0)
        self.assert_allclose(sp.x1_at(1), 7.0)
        self.assert_allclose(sp.y1_at(1), -8.0)

        sp.mirror('y')
        self.assert_allclose(sp.x0_at(0), -1.0)
        self.assert_allclose(sp.y0_at(0), -2.0)
        self.assert_allclose(sp.x1_at(0), -3.0)
        self.assert_allclose(sp.y1_at(0), -4.0)
        self.assert_allclose(sp.x0_at(1), -5.0)
        self.assert_allclose(sp.y0_at(1), -6.0)
        self.assert_allclose(sp.x1_at(1), -7.0)
        self.assert_allclose(sp.y1_at(1), -8.0)

    def test_mirror_3d(self):
        SegmentPad = self.SegmentPad

        sp = SegmentPad(ndim=3)
        sp.append(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        sp.append(7.0, 8.0, 9.0, 10.0, 11.0, 12.0)

        sp.mirror('z')
        self.assert_allclose(sp.x0_at(0), -1.0)
        self.assert_allclose(sp.y0_at(0), -2.0)
        self.assert_allclose(sp.z0_at(0), 3.0)
        self.assert_allclose(sp.x1_at(0), -4.0)
        self.assert_allclose(sp.y1_at(0), -5.0)
        self.assert_allclose(sp.z1_at(0), 6.0)
        self.assert_allclose(sp.x0_at(1), -7.0)
        self.assert_allclose(sp.y0_at(1), -8.0)
        self.assert_allclose(sp.z0_at(1), 9.0)
        self.assert_allclose(sp.x1_at(1), -10.0)
        self.assert_allclose(sp.y1_at(1), -11.0)
        self.assert_allclose(sp.z1_at(1), 12.0)

        sp.mirror('X')
        self.assert_allclose(sp.x0_at(0), -1.0)
        self.assert_allclose(sp.y0_at(0), 2.0)
        self.assert_allclose(sp.z0_at(0), -3.0)
        self.assert_allclose(sp.x1_at(0), -4.0)
        self.assert_allclose(sp.y1_at(0), 5.0)
        self.assert_allclose(sp.z1_at(0), -6.0)

        with self.assertRaisesRegex(
                ValueError,
                "SegmentPad::mirror: axis must be 'x', 'y', or 'z'"):
            sp.mirror('w')

    def test_offset_by_axis(self):
        SegmentPad = self.SegmentPad
        SimpleArray = self.SimpleArray

        sp = SegmentPad(ndim=2)
        sp.append(0.0, 0.0, 10.0, 0.0)
        sp.append(0.0, 0.0, 0.0, 10.0)

        def wrap(values):
            return SimpleArray(array=np.array(values, dtype=self.dtype))

        xs = wrap([5.0, 5.0, 0.0])
        ys = wrap([3.0, -2.0, 0.0])
        # Against the horizontal arm: positive below, matching
        # Segment3d.offset_by_axis on the same points.
        np.testing.assert_array_equal(
            sp.offset_by_axis(0, xs, ys).ndarray, [-3.0, 2.0, 0.0])
        # Against the vertical arm the normal points to +x.
        np.testing.assert_array_equal(
            sp.offset_by_axis(1, xs, ys).ndarray, [5.0, 5.0, 0.0])
        # The pad queries sit at z = 0: with the x reference the normal
        # of the vertical arm points along -z, so every offset is zero.
        np.testing.assert_array_equal(
            sp.offset_by_axis(1, xs, ys, 'x').ndarray, [0.0, 0.0, 0.0])

        # A strided view is read through its stride, not densely.
        pts = np.array([[5.0, 3.0], [5.0, -2.0]], dtype=self.dtype)
        np.testing.assert_array_equal(
            sp.offset_by_axis(0, SimpleArray(array=pts[:, 0]),
                              SimpleArray(array=pts[:, 1])).ndarray,
            [-3.0, 2.0])

        with self.assertRaisesRegex(
                IndexError,
                "SegmentPad::offset_by_axis: "
                "index 2 is out of bounds with size 2"):
            sp.offset_by_axis(2, xs, ys)
        with self.assertRaisesRegex(
                ValueError, "must be 1-dimensional and equally long"):
            sp.offset_by_axis(0, xs, wrap([1.0, 2.0]))


class SegmentPadFp32TC(SegmentPadTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float32'
        self.SimpleArray = solvcon.SimpleArrayFloat32
        self.Point = solvcon.Point3dFp32
        self.PointPad = solvcon.PointPadFp32
        self.Segment = solvcon.Segment3dFp32
        self.SegmentPad = solvcon.SegmentPadFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)


class SegmentPadFp64TC(SegmentPadTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float64'
        self.SimpleArray = solvcon.SimpleArrayFloat64
        self.Point = solvcon.Point3dFp64
        self.PointPad = solvcon.PointPadFp64
        self.Segment = solvcon.Segment3dFp64
        self.SegmentPad = solvcon.SegmentPadFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)


class CurvePadTB(testing.TestBase):

    def test_ndim(self):
        cp2d = self.CurvePad(ndim=2)
        self.assertEqual(cp2d.ndim, 2)
        cp3d = self.CurvePad(ndim=3)
        self.assertEqual(cp3d.ndim, 3)

        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 0 < 2"):
            self.CurvePad(ndim=0)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 0 < 2"):
            self.CurvePad(ndim=0, nelem=2)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 1 < 2"):
            self.CurvePad(ndim=1)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 1 < 2"):
            self.CurvePad(ndim=1, nelem=3)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 4 > 3"):
            self.CurvePad(ndim=4)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 4 > 3"):
            self.CurvePad(ndim=4, nelem=5)

    def test_append_2d(self):
        cp = self.CurvePad(ndim=2)
        self.assertEqual(cp.ndim, 2)
        self.assertEqual(len(cp), 0)

        p0 = self.Point(0, 0, 0)
        p1 = self.Point(1, 1, 0)
        p2 = self.Point(3, 1, 0)
        p3 = self.Point(4, 0, 0)
        cp.append(p0=p0, p1=p1, p2=p2, p3=p3)
        self.assertEqual(len(cp), 1)

        self.assertEqual(cp.x0_at(0), 0)
        self.assertEqual(cp.y0_at(0), 0)
        self.assertEqual(cp.x1_at(0), 1)
        self.assertEqual(cp.y1_at(0), 1)
        self.assertEqual(cp.x2_at(0), 3)
        self.assertEqual(cp.y2_at(0), 1)
        self.assertEqual(cp.x3_at(0), 4)
        self.assertEqual(cp.y3_at(0), 0)

        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 0 is out of bounds with size 0"):
            cp.z0_at(0)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 0 is out of bounds with size 0"):
            cp.z1_at(0)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 0 is out of bounds with size 0"):
            cp.z2_at(0)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 0 is out of bounds with size 0"):
            cp.z3_at(0)

        b = cp[0]
        self.assertEqual(len(b), 4)
        self.assertEqual(list(b[0]), [0, 0, 0])
        self.assertEqual(list(b[1]), [1, 1, 0])
        self.assertEqual(list(b[2]), [3, 1, 0])
        self.assertEqual(list(b[3]), [4, 0, 0])

        p0 = self.Point(7, 8, 0)
        p1 = self.Point(1, 1, 0)
        p2 = self.Point(3, 1, 0)
        p3 = self.Point(4, 0, 0)
        b = self.Bezier(p0, p1, p2, p3)
        cp[0] = b
        self.assertEqual(list(cp[0][0]), [7, 8, 0])
        self.assertEqual(list(cp[0][1]), [1, 1, 0])
        self.assertEqual(list(cp[0][2]), [3, 1, 0])
        self.assertEqual(list(cp[0][3]), [4, 0, 0])

    def test_append_3d(self):
        cp = self.CurvePad(ndim=3)
        self.assertEqual(cp.ndim, 3)
        self.assertEqual(len(cp), 0)

        p0 = self.Point(0, 0, 0)
        p1 = self.Point(1, 1, 0)
        p2 = self.Point(3, 1, 0)
        p3 = self.Point(4, 0, 0)
        cp.append(p0=p0, p1=p1, p2=p2, p3=p3)
        self.assertEqual(len(cp), 1)

        self.assertEqual(cp.x0_at(0), 0)
        self.assertEqual(cp.y0_at(0), 0)
        self.assertEqual(cp.z0_at(0), 0)
        self.assertEqual(cp.x1_at(0), 1)
        self.assertEqual(cp.y1_at(0), 1)
        self.assertEqual(cp.z1_at(0), 0)
        self.assertEqual(cp.x2_at(0), 3)
        self.assertEqual(cp.y2_at(0), 1)
        self.assertEqual(cp.z2_at(0), 0)
        self.assertEqual(cp.x3_at(0), 4)
        self.assertEqual(cp.y3_at(0), 0)
        self.assertEqual(cp.z3_at(0), 0)

        p0 = self.Point(7, 8, -3)
        p1 = self.Point(1, 1, 0)
        p2 = self.Point(3, 1, 0)
        p3 = self.Point(4, 0, 0)
        b = self.Bezier(p0, p1, p2, p3)
        cp[0] = b
        self.assertEqual(list(cp[0][0]), [7, 8, -3])
        self.assertEqual(list(cp[0][1]), [1, 1, 0])
        self.assertEqual(list(cp[0][2]), [3, 1, 0])
        self.assertEqual(list(cp[0][3]), [4, 0, 0])

        b2 = self.Bezier(
            p0=self.Point(0, 0, 1),
            p1=self.Point(1.3, 1.921, 2),
            p2=self.Point(3.2, 1.224, 3),
            p3=self.Point(4.87, 0.12, 4))
        cp.append(c=b2)
        self.assertEqual(len(cp), 2)
        # Assert the equality between value array and PointPad
        self.assert_allclose(list(cp.x0), list(cp.p0.x))
        self.assert_allclose(list(cp.y0), list(cp.p0.y))
        self.assert_allclose(list(cp.z0), list(cp.p0.z))
        self.assert_allclose(list(cp.x1), list(cp.p1.x))
        self.assert_allclose(list(cp.y1), list(cp.p1.y))
        self.assert_allclose(list(cp.z1), list(cp.p1.z))
        self.assert_allclose(list(cp.x2), list(cp.p2.x))
        self.assert_allclose(list(cp.y2), list(cp.p2.y))
        self.assert_allclose(list(cp.z2), list(cp.p2.z))
        self.assert_allclose(list(cp.x3), list(cp.p3.x))
        self.assert_allclose(list(cp.y3), list(cp.p3.y))
        self.assert_allclose(list(cp.z3), list(cp.p3.z))
        # Check the value
        self.assert_allclose(list(cp.x0), [7, 0])
        self.assert_allclose(list(cp.y0), [8, 0])
        self.assert_allclose(list(cp.z0), [-3, 1])
        self.assert_allclose(list(cp.x1), [1, 1.3])
        self.assert_allclose(list(cp.y1), [1, 1.921])
        self.assert_allclose(list(cp.z1), [0, 2])
        self.assert_allclose(list(cp.x2), [3, 3.2])
        self.assert_allclose(list(cp.y2), [1, 1.224])
        self.assert_allclose(list(cp.z2), [0, 3])
        self.assert_allclose(list(cp.x3), [4, 4.87])
        self.assert_allclose(list(cp.y3), [0, 0.12])
        self.assert_allclose(list(cp.z3), [0, 4])

    def test_sample_2d(self):
        CurvePad = self.CurvePad
        Point = self.Point
        Bezier = self.Bezier

        cp = CurvePad(ndim=3)
        p0 = Point(0, 0, 0)
        p1 = Point(1, 1, 0)
        p2 = Point(3, 1, 0)
        p3 = Point(4, 0, 0)
        cp.append(p0=p0, p1=p1, p2=p2, p3=p3)
        self.assertEqual(len(cp), 1)
        p4 = Point(5, 0, 0)
        p5 = Point(5.5, 1, 0)
        p6 = Point(6.5, 1, 0)
        p7 = Point(7, 0, 0)
        c = Bezier(p0=p4, p1=p5, p2=p6, p3=p7)
        cp.append(c)
        self.assertEqual(len(cp), 2)

        # Sample to create segment pad
        sp = cp.sample(length=0.5)
        self.assertEqual(len(sp), 10)

        # The connectivity of the first curve
        self.assertEqual(p0, sp[0].p0)
        self.assertEqual(sp[0].p1, sp[1].p0)
        self.assertEqual(sp[1].p1, sp[2].p0)
        self.assertEqual(sp[2].p1, sp[3].p0)
        self.assertEqual(sp[3].p1, sp[4].p0)
        self.assertEqual(sp[4].p1, sp[5].p0)
        self.assertEqual(sp[5].p1, sp[6].p0)
        self.assertEqual(sp[6].p1, p3)

        # The connectivity of the second curve
        self.assertEqual(p4, sp[7].p0)
        self.assertEqual(sp[7].p1, sp[8].p0)
        self.assertEqual(sp[8].p1, sp[9].p0)
        self.assertEqual(sp[9].p1, p7)

        # Test for the segment coordinates of the first curve
        self.assert_allclose(list(sp[0].p0),
                             [0.0, 0.0, 0.0])
        self.assert_allclose(list(sp[0].p1),
                             [0.48396501457725954, 0.3673469387755103, 0.0])
        self.assert_allclose(list(sp[1].p0),
                             [0.48396501457725954, 0.3673469387755103, 0.0])
        self.assert_allclose(list(sp[1].p1),
                             [1.0553935860058308, 0.6122448979591837, 0.0])
        self.assert_allclose(list(sp[2].p0),
                             [1.0553935860058308, 0.6122448979591837, 0.0])
        self.assert_allclose(list(sp[2].p1),
                             [1.6793002915451893, 0.7346938775510203, 0.0])
        self.assert_allclose(list(sp[3].p0),
                             [1.6793002915451893, 0.7346938775510203, 0.0])
        self.assert_allclose(list(sp[3].p1),
                             [2.3206997084548107, 0.7346938775510206, 0.0])
        self.assert_allclose(list(sp[4].p0),
                             [2.3206997084548107, 0.7346938775510206, 0.0])
        self.assert_allclose(list(sp[4].p1),
                             [2.944606413994169, 0.6122448979591837, 0.0])
        self.assert_allclose(list(sp[5].p0),
                             [2.944606413994169, 0.6122448979591837, 0.0])
        self.assert_allclose(list(sp[5].p1),
                             [3.5160349854227406, 0.36734693877551033, 0.0])
        self.assert_allclose(list(sp[6].p0),
                             [3.5160349854227406, 0.36734693877551033, 0.0])
        self.assert_allclose(list(sp[6].p1),
                             [4.0, 0.0, 0.0])

        # Test for the segment coordinates of the second curve
        self.assert_allclose(list(sp[7].p0),
                             [5.0, 0.0, 0.0])
        self.assert_allclose(list(sp[7].p1),
                             [5.6296296296296315, 0.6666666666666667, 0.0])
        self.assert_allclose(list(sp[8].p0),
                             [5.6296296296296315, 0.6666666666666667, 0.0])
        self.assert_allclose(list(sp[8].p1),
                             [6.370370370370371, 0.6666666666666667, 0.0])
        self.assert_allclose(list(sp[9].p0),
                             [6.370370370370371, 0.6666666666666667, 0.0])
        self.assert_allclose(list(sp[9].p1),
                             [7.0, 0.0, 0.0])

    def _make_curve_pad(self, ndim, ncurve):
        cp = self.CurvePad(ndim=ndim)
        for it in range(ncurve):
            cp.append(self.Point(it, 0, 0), self.Point(it, 1, 0),
                      self.Point(it, 2, 0), self.Point(it, 3, 0))
        return cp

    def test_getitem_index(self):
        cp = self._make_curve_pad(2, 4)

        self.assert_allclose(list(cp[0][0]), [0, 0, 0])
        self.assertEqual(cp[-1], cp[3])
        self.assertEqual(cp[-4], cp[0])

        with self.assertRaisesRegex(
                IndexError, "CurvePad: index 4 is out of bounds with size 4"):
            cp[4]
        with self.assertRaisesRegex(
                IndexError, "CurvePad: index -5 is out of bounds with size 4"):
            cp[-5]

        empty = self.CurvePad(ndim=3)
        with self.assertRaisesRegex(
                IndexError, "CurvePad: index 0 is out of bounds with size 0"):
            empty[0]
        with self.assertRaisesRegex(
                IndexError, "CurvePad: index -1 is out of bounds with size 0"):
            empty[-1]

    def test_setitem_index(self):
        Point = self.Point
        cp = self._make_curve_pad(3, 4)

        c = self.Bezier(Point(9, 0, 0), Point(9, 1, 0),
                        Point(9, 2, 0), Point(9, 3, 0))
        cp[-1] = c
        self.assertEqual(cp[3], c)

        with self.assertRaisesRegex(
                IndexError, "CurvePad: index -5 is out of bounds with size 4"):
            cp[-5] = c

    def test_getitem_slice(self):
        cp = self._make_curve_pad(3, 4)

        full = cp[:]
        self.assertIsInstance(full, type(cp))
        self.assertEqual(full.ndim, 3)
        self.assertEqual(len(full), 4)
        for it in range(4):
            self.assertEqual(full[it], cp[it])

        strided = cp[1::2]
        self.assertEqual(strided.ndim, 3)
        self.assertEqual(len(strided), 2)
        self.assertEqual(strided[0], cp[1])
        self.assertEqual(strided[1], cp[3])

        flipped = cp[::-1]
        self.assert_allclose(list(flipped.x0), [3, 2, 1, 0])

        # A slice copies, so writing to it leaves the source alone.
        part = cp[1:3]
        part[0] = cp[0]
        self.assertEqual(part[0], cp[0])
        self.assert_allclose(cp.x0_at(1), 1.0)

        # A 2D pad keeps its dimensionality and its empty z arrays.
        cp2d = self._make_curve_pad(2, 1)
        sliced2d = cp2d[:]
        self.assertEqual(sliced2d.ndim, 2)
        self.assertEqual(len(sliced2d.z0), 0)
        self.assertEqual(sliced2d[0], cp2d[0])

        empty = self.CurvePad(ndim=3)
        self.assertEqual(len(empty[:]), 0)
        self.assertEqual(empty[:].ndim, 3)
        self.assertEqual(len(cp[3:1]), 0)

    def test_mirror(self):
        CurvePad = self.CurvePad
        Point = self.Point

        cp = CurvePad(ndim=3)
        cp.append(Point(1, 2, 3), Point(4, 5, 6),
                  Point(7, 8, 9), Point(10, 11, 12))
        cp.append(Point(-1, -2, -3), Point(-4, -5, -6),
                  Point(-7, -8, -9), Point(-10, -11, -12))

        cp.mirror('x')
        self.assert_allclose(list(cp.x0), [1, -1])
        self.assert_allclose(list(cp.y0), [-2, 2])
        self.assert_allclose(list(cp.z0), [-3, 3])
        self.assert_allclose(list(cp.x1), [4, -4])
        self.assert_allclose(list(cp.y1), [-5, 5])
        self.assert_allclose(list(cp.z1), [-6, 6])
        self.assert_allclose(list(cp.x2), [7, -7])
        self.assert_allclose(list(cp.y2), [-8, 8])
        self.assert_allclose(list(cp.z2), [-9, 9])
        self.assert_allclose(list(cp.x3), [10, -10])
        self.assert_allclose(list(cp.y3), [-11, 11])
        self.assert_allclose(list(cp.z3), [-12, 12])

        cp.mirror('y')
        self.assert_allclose(list(cp.x0), [-1, 1])
        self.assert_allclose(list(cp.y0), [-2, 2])
        self.assert_allclose(list(cp.z0), [3, -3])
        self.assert_allclose(list(cp.x1), [-4, 4])
        self.assert_allclose(list(cp.y1), [-5, 5])
        self.assert_allclose(list(cp.z1), [6, -6])
        self.assert_allclose(list(cp.x2), [-7, 7])
        self.assert_allclose(list(cp.y2), [-8, 8])
        self.assert_allclose(list(cp.z2), [9, -9])
        self.assert_allclose(list(cp.x3), [-10, 10])
        self.assert_allclose(list(cp.y3), [-11, 11])
        self.assert_allclose(list(cp.z3), [12, -12])

        cp.mirror('Z')
        self.assert_allclose(list(cp.x0), [1, -1])
        self.assert_allclose(list(cp.y0), [2, -2])
        self.assert_allclose(list(cp.z0), [3, -3])
        self.assert_allclose(list(cp.x1), [4, -4])
        self.assert_allclose(list(cp.y1), [5, -5])
        self.assert_allclose(list(cp.z1), [6, -6])
        self.assert_allclose(list(cp.x2), [7, -7])
        self.assert_allclose(list(cp.y2), [8, -8])
        self.assert_allclose(list(cp.z2), [9, -9])
        self.assert_allclose(list(cp.x3), [10, -10])
        self.assert_allclose(list(cp.y3), [11, -11])
        self.assert_allclose(list(cp.z3), [12, -12])

        with self.assertRaisesRegex(
                ValueError, "CurvePad::mirror: axis must be 'x', 'y', or 'z'"):
            cp.mirror('w')


class CurvePadFp32TC(CurvePadTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float32'
        self.SimpleArray = solvcon.SimpleArrayFloat32
        self.Point = solvcon.Point3dFp32
        self.PointPad = solvcon.PointPadFp32
        self.Segment = solvcon.Segment3dFp32
        self.SegmentPad = solvcon.SegmentPadFp32
        self.Bezier = solvcon.Bezier3dFp32
        self.CurvePad = solvcon.CurvePadFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.5e-7
        return super().assert_allclose(*args, **kw)


class CurvePadFp64TC(CurvePadTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float64'
        self.SimpleArray = solvcon.SimpleArrayFloat64
        self.Point = solvcon.Point3dFp64
        self.PointPad = solvcon.PointPadFp64
        self.Segment = solvcon.Segment3dFp64
        self.SegmentPad = solvcon.SegmentPadFp64
        self.Bezier = solvcon.Bezier3dFp64
        self.CurvePad = solvcon.CurvePadFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
