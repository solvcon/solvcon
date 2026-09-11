# Copyright (c) 2025, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

import numpy as np

import solvcon
import solvcon as sc
from solvcon import testing


class Triangle3dTB(testing.TestBase):

    def test_construct(self):
        Point = self.Point
        Triangle = self.Triangle

        t = Triangle(p0=Point(x=0, y=0, z=0),
                     p1=Point(x=1, y=1, z=1),
                     p2=Point(x=2, y=2, z=2))
        self.assertEqual(len(t), 3)
        self.assertEqual(tuple(t.p0), (0.0, 0.0, 0.0))
        self.assertEqual(tuple(t.p1), (1.0, 1.0, 1.0))
        self.assertEqual(tuple(t.p2), (2.0, 2.0, 2.0))

        t.p0 = Point(x=3, y=7, z=0)
        t.p1 = Point(x=-1, y=-4, z=9)
        t.p2 = Point(x=5, y=6, z=-2)
        self.assertEqual(t.x0, 3)
        self.assertEqual(t.y0, 7)
        self.assertEqual(t.z0, 0)
        self.assertEqual(t.x1, -1)
        self.assertEqual(t.y1, -4)
        self.assertEqual(t.z1, 9)
        self.assertEqual(t.x2, 5)
        self.assertEqual(t.y2, 6)
        self.assertEqual(t.z2, -2)

        t = Triangle(Point(x=3.1, y=7.4, z=0.6),
                     Point(x=-1.2, y=-4.1, z=9.2),
                     Point(x=5.5, y=6.6, z=-2.3))
        self.assert_allclose(tuple(t.p0), (3.1, 7.4, 0.6))
        self.assert_allclose(tuple(t.p1), (-1.2, -4.1, 9.2))
        self.assert_allclose(tuple(t.p2), (5.5, 6.6, -2.3))

    def test_indexing(self):
        Point = self.Point
        Triangle = self.Triangle

        t = Triangle(Point(1, 2, 3), Point(4, 5, 6), Point(7, 8, 9))
        self.assertEqual(list(t[0]), [1, 2, 3])
        self.assertEqual(list(t[1]), [4, 5, 6])
        self.assertEqual(list(t[2]), [7, 8, 9])

        t.p0 = Point(10, 11, 12)
        self.assertEqual(list(t.p0), [10, 11, 12])
        t.p1 = Point(13, 14, 15)
        self.assertEqual(list(t.p1), [13, 14, 15])
        t.p2 = Point(16, 17, 18)
        self.assertEqual(list(t.p2), [16, 17, 18])

    def test_equality(self):
        Point = self.Point
        Triangle = self.Triangle

        t1 = Triangle(Point(1, 2, 3), Point(4, 5, 6), Point(7, 8, 9))
        t2 = Triangle(Point(1, 2, 3), Point(4, 5, 6), Point(7, 8, 9))
        t3 = Triangle(Point(1, 2, 3), Point(4, 5, 6), Point(7, 8, 10))

        self.assertEqual(t1, t2)
        self.assertNotEqual(t1, t3)

    def test_mirror(self):
        Point = self.Point
        Triangle = self.Triangle

        t1 = Triangle(Point(1, 2, 3), Point(4, 5, 6), Point(7, 8, 9))
        t1.mirror('x')
        self.assertEqual(list(t1.p0), [1, -2, -3])
        self.assertEqual(list(t1.p1), [4, -5, -6])
        self.assertEqual(list(t1.p2), [7, -8, -9])

        t2 = Triangle(Point(1, 2, 3), Point(4, 5, 6), Point(7, 8, 9))
        t2.mirror('y')
        self.assertEqual(list(t2.p0), [-1, 2, -3])
        self.assertEqual(list(t2.p1), [-4, 5, -6])
        self.assertEqual(list(t2.p2), [-7, 8, -9])

        t3 = Triangle(Point(1, 2, 3), Point(4, 5, 6), Point(7, 8, 9))
        t3.mirror('z')
        self.assertEqual(list(t3.p0), [-1, -2, 3])
        self.assertEqual(list(t3.p1), [-4, -5, 6])
        self.assertEqual(list(t3.p2), [-7, -8, 9])

        t4 = Triangle(Point(1, 2, 3), Point(4, 5, 6), Point(7, 8, 9))
        t4.mirror('Y')
        self.assertEqual(list(t4.p0), [-1, 2, -3])
        self.assertEqual(list(t4.p1), [-4, 5, -6])
        self.assertEqual(list(t4.p2), [-7, 8, -9])

        with self.assertRaisesRegex(
                ValueError,
                "Triangle3d::mirror: axis must be 'x', 'y', or 'z'"):
            Triangle(Point(1, 2, 3), Point(4, 5, 6),
                     Point(7, 8, 9)).mirror('w')


class Triangle3dFp32TC(Triangle3dTB, unittest.TestCase):

    def setUp(self):
        self.Point = solvcon.Point3dFp32
        self.Triangle = solvcon.Triangle3dFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)

    def test_repr_str(self):
        t = solvcon.Triangle3dFp32(solvcon.Point3dFp32(1.5, 2.5, 3.5),
                                   solvcon.Point3dFp32(4.5, 5.5, 6.5),
                                   solvcon.Point3dFp32(7.5, 8.5, 9.5))
        golden = ("Triangle3dFp32(Point3dFp32(1.5, 2.5, 3.5), "
                  "Point3dFp32(4.5, 5.5, 6.5), "
                  "Point3dFp32(7.5, 8.5, 9.5))")
        self.assertEqual(repr(t), golden)
        self.assertEqual(str(t), golden)
        e = eval(golden, vars(solvcon))
        self.assertEqual(t, e)


class Triangle3dFp64TC(Triangle3dTB, unittest.TestCase):

    def setUp(self):
        self.Point = solvcon.Point3dFp64
        self.Triangle = solvcon.Triangle3dFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

    def test_repr_str(self):
        t = solvcon.Triangle3dFp64(solvcon.Point3dFp64(1.5, 2.5, 3.5),
                                   solvcon.Point3dFp64(4.5, 5.5, 6.5),
                                   solvcon.Point3dFp64(7.5, 8.5, 9.5))
        golden = ("Triangle3dFp64(Point3dFp64(1.5, 2.5, 3.5), "
                  "Point3dFp64(4.5, 5.5, 6.5), "
                  "Point3dFp64(7.5, 8.5, 9.5))")
        self.assertEqual(repr(t), golden)
        self.assertEqual(str(t), golden)
        e = eval(golden, vars(solvcon))
        self.assertEqual(t, e)


class TrianglePadTB(testing.TestBase):

    def test_ndim(self):
        tp2d = self.TrianglePad(ndim=2)
        self.assertEqual(tp2d.ndim, 2)
        tp3d = self.TrianglePad(ndim=3)
        self.assertEqual(tp3d.ndim, 3)

        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 0 < 2"):
            self.TrianglePad(ndim=0)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 0 < 2"):
            self.TrianglePad(ndim=0, nelem=2)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 1 < 2"):
            self.TrianglePad(ndim=1)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 1 < 2"):
            self.TrianglePad(ndim=1, nelem=3)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 4 > 3"):
            self.TrianglePad(ndim=4)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 4 > 3"):
            self.TrianglePad(ndim=4, nelem=5)

    def test_construct_2d(self):
        x0arr = self.SimpleArray(
            array=np.array([1, 2, 3], dtype=self.dtype))
        y0arr = self.SimpleArray(
            array=np.array([4, 5, 6], dtype=self.dtype))
        x1arr = self.SimpleArray(
            array=np.array([7, 8, 9], dtype=self.dtype))
        y1arr = self.SimpleArray(
            array=np.array([10, 11, 12], dtype=self.dtype))
        x2arr = self.SimpleArray(
            array=np.array([13, 14, 15], dtype=self.dtype))
        y2arr = self.SimpleArray(
            array=np.array([16, 17, 18], dtype=self.dtype))
        tp = self.TrianglePad(x0=x0arr, y0=y0arr, x1=x1arr, y1=y1arr,
                              x2=x2arr, y2=y2arr, clone=False)
        self.assertEqual(tp.ndim, 2)
        self.assert_allclose(tp.x0, [1, 2, 3])
        self.assert_allclose(tp.y0, [4, 5, 6])
        self.assert_allclose(tp.x1, [7, 8, 9])
        self.assert_allclose(tp.y1, [10, 11, 12])
        self.assert_allclose(tp.x2, [13, 14, 15])
        self.assert_allclose(tp.y2, [16, 17, 18])
        self.assertEqual(len(tp.z0), 0)
        self.assertEqual(len(tp.z1), 0)
        self.assertEqual(len(tp.z2), 0)

        tp.y0[0] = -700.3
        tp.y1[0] = 900.5
        tp.y2[0] = 1100.7
        tp.x0[1] = 200.2
        tp.x1[1] = -800.4
        tp.x2[1] = -1000.6
        self.assert_allclose(
            list(tp[0]), [[1, -700.3, 0], [7, 900.5, 0], [13, 1100.7, 0]])
        self.assert_allclose(
            list(tp[1]),
            [[200.2, 5, 0], [-800.4, 11, 0], [-1000.6, 17, 0]])
        self.assert_allclose(
            list(tp[2]), [[3, 6, 0], [9, 12, 0], [15, 18, 0]])  # no change

        tp2 = self.TrianglePad(ndim=2, nelem=3)
        for i in range(len(tp)):
            tp2.set_at(i, tp.get_at(i).x0, tp.get_at(i).y0,
                       tp.get_at(i).x1, tp.get_at(i).y1,
                       tp.get_at(i).x2, tp.get_at(i).y2)
        self.assert_allclose(tp2.x0, [1, 200.2, 3])
        self.assert_allclose(tp2.y0, [-700.3, 5, 6])
        self.assert_allclose(tp2.x1, [7, -800.4, 9])
        self.assert_allclose(tp2.y1, [900.5, 11, 12])
        self.assert_allclose(tp2.x2, [13, -1000.6, 15])
        self.assert_allclose(tp2.y2, [1100.7, 17, 18])
        self.assertEqual(len(tp2.z0), 0)
        self.assertEqual(len(tp2.z1), 0)
        self.assertEqual(len(tp2.z2), 0)

        packed = tp2.pack_array().ndarray
        self.assertEqual(packed.shape, (3, 6))
        self.assert_allclose(
            list(packed[0]), (1, -700.3, 7, 900.5, 13, 1100.7))
        self.assert_allclose(
            list(packed[1]), (200.2, 5, -800.4, 11, -1000.6, 17))
        self.assert_allclose(
            list(packed[2]), (3, 6, 9, 12, 15, 18))

    def test_construct_3d(self):
        x0arr = self.SimpleArray(
            array=np.array([1, 2, 3], dtype=self.dtype))
        y0arr = self.SimpleArray(
            array=np.array([4, 5, 6], dtype=self.dtype))
        z0arr = self.SimpleArray(
            array=np.array([7, 8, 9], dtype=self.dtype))
        x1arr = self.SimpleArray(
            array=np.array([10, 11, 12], dtype=self.dtype))
        y1arr = self.SimpleArray(
            array=np.array([13, 14, 15], dtype=self.dtype))
        z1arr = self.SimpleArray(
            array=np.array([16, 17, 18], dtype=self.dtype))
        x2arr = self.SimpleArray(
            array=np.array([19, 20, 21], dtype=self.dtype))
        y2arr = self.SimpleArray(
            array=np.array([22, 23, 24], dtype=self.dtype))
        z2arr = self.SimpleArray(
            array=np.array([25, 26, 27], dtype=self.dtype))
        tp = self.TrianglePad(
            x0=x0arr, y0=y0arr, z0=z0arr,
            x1=x1arr, y1=y1arr, z1=z1arr,
            x2=x2arr, y2=y2arr, z2=z2arr,
            clone=False)
        self.assertEqual(tp.ndim, 3)
        self.assert_allclose(tp.x0, [1, 2, 3])
        self.assert_allclose(tp.y0, [4, 5, 6])
        self.assert_allclose(tp.z0, [7, 8, 9])
        self.assert_allclose(tp.x1, [10, 11, 12])
        self.assert_allclose(tp.y1, [13, 14, 15])
        self.assert_allclose(tp.z1, [16, 17, 18])
        self.assert_allclose(tp.x2, [19, 20, 21])
        self.assert_allclose(tp.y2, [22, 23, 24])
        self.assert_allclose(tp.z2, [25, 26, 27])

        tp.y0[0] = -300.3
        tp.y1[0] = 600.6
        tp.y2[0] = -900.9
        tp.x0[1] = 200.2
        tp.x1[1] = -500.5
        tp.x2[1] = 800.8
        tp.z0[2] = 400.4
        tp.z1[2] = -700.7
        tp.z2[2] = 1000.1
        self.assert_allclose(
            list(tp[0]),
            [[1, -300.3, 7], [10, 600.6, 16], [19, -900.9, 25]])  # y changed
        self.assert_allclose(
            list(tp[1]),
            [[200.2, 5, 8], [-500.5, 14, 17], [800.8, 23, 26]])  # x changed
        self.assert_allclose(
            list(tp[2]),
            [[3, 6, 400.4], [12, 15, -700.7], [21, 24, 1000.1]])  # z changed

        tp2 = self.TrianglePad(ndim=3, nelem=3)
        for i in range(len(tp)):
            tp2.set_at(
                i,
                tp.get_at(i).x0, tp.get_at(i).y0, tp.get_at(i).z0,
                tp.get_at(i).x1, tp.get_at(i).y1, tp.get_at(i).z1,
                tp.get_at(i).x2, tp.get_at(i).y2, tp.get_at(i).z2)
        self.assert_allclose(tp2.x0, [1, 200.2, 3])
        self.assert_allclose(tp2.y0, [-300.3, 5, 6])
        self.assert_allclose(tp2.z0, [7, 8, 400.4])
        self.assert_allclose(tp2.x1, [10, -500.5, 12])
        self.assert_allclose(tp2.y1, [600.6, 14, 15])
        self.assert_allclose(tp2.z1, [16, 17, -700.7])
        self.assert_allclose(tp2.x2, [19, 800.8, 21])
        self.assert_allclose(tp2.y2, [-900.9, 23, 24])
        self.assert_allclose(tp2.z2, [25, 26, 1000.1])

        packed = tp2.pack_array().ndarray
        self.assertEqual(packed.shape, (3, 9))
        self.assert_allclose(
            list(packed[0]),
            (1, -300.3, 7, 10, 600.6, 16, 19, -900.9, 25))
        self.assert_allclose(
            list(packed[1]),
            (200.2, 5, 8, -500.5, 14, 17, 800.8, 23, 26))
        self.assert_allclose(
            list(packed[2]),
            (3, 6, 400.4, 12, 15, -700.7, 21, 24, 1000.1))

    def test_append_2d(self):
        tp = self.TrianglePad(ndim=2)
        self.assertEqual(tp.ndim, 2)
        tp.append(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        tp.append(
            self.Point(7.0, 8.0),
            self.Point(9.0, 10.0),
            self.Point(11.0, 12.0))
        tp.append(self.Triangle(
            self.Point(13.0, 14.0),
            self.Point(15.0, 16.0),
            self.Point(17.0, 18.0)))

        self.assertEqual(tp.x0_at(0), 1.0)
        self.assertEqual(tp.y0_at(0), 2.0)
        self.assertEqual(tp.x1_at(0), 3.0)
        self.assertEqual(tp.y1_at(0), 4.0)
        self.assertEqual(tp.x2_at(0), 5.0)
        self.assertEqual(tp.y2_at(0), 6.0)

        self.assertEqual(tp.x0_at(1), 7.0)
        self.assertEqual(tp.y0_at(1), 8.0)
        self.assertEqual(tp.x1_at(1), 9.0)
        self.assertEqual(tp.y1_at(1), 10.0)
        self.assertEqual(tp.x2_at(1), 11.0)
        self.assertEqual(tp.y2_at(1), 12.0)

        self.assertEqual(tp.x0_at(2), 13.0)
        self.assertEqual(tp.y0_at(2), 14.0)
        self.assertEqual(tp.x1_at(2), 15.0)
        self.assertEqual(tp.y1_at(2), 16.0)
        self.assertEqual(tp.x2_at(2), 17.0)
        self.assertEqual(tp.y2_at(2), 18.0)

    def test_append_3d(self):
        tp = self.TrianglePad(ndim=3)
        self.assertEqual(tp.ndim, 3)
        tp.append(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        tp.append(
            self.Point(10.0, 11.0, 12.0),
            self.Point(13.0, 14.0, 15.0),
            self.Point(16.0, 17.0, 18.0))
        tp.append(self.Triangle(
            self.Point(19.0, 20.0, 21.0),
            self.Point(22.0, 23.0, 24.0),
            self.Point(25.0, 26.0, 27.0)))

        self.assertEqual(tp.x0_at(0), 1.0)
        self.assertEqual(tp.y0_at(0), 2.0)
        self.assertEqual(tp.z0_at(0), 3.0)
        self.assertEqual(tp.x1_at(0), 4.0)
        self.assertEqual(tp.y1_at(0), 5.0)
        self.assertEqual(tp.z1_at(0), 6.0)
        self.assertEqual(tp.x2_at(0), 7.0)
        self.assertEqual(tp.y2_at(0), 8.0)
        self.assertEqual(tp.z2_at(0), 9.0)

        self.assertEqual(tp.x0_at(1), 10.0)
        self.assertEqual(tp.y0_at(1), 11.0)
        self.assertEqual(tp.z0_at(1), 12.0)
        self.assertEqual(tp.x1_at(1), 13.0)
        self.assertEqual(tp.y1_at(1), 14.0)
        self.assertEqual(tp.z1_at(1), 15.0)
        self.assertEqual(tp.x2_at(1), 16.0)
        self.assertEqual(tp.y2_at(1), 17.0)
        self.assertEqual(tp.z2_at(1), 18.0)

        self.assertEqual(tp.x0_at(2), 19.0)
        self.assertEqual(tp.y0_at(2), 20.0)
        self.assertEqual(tp.z0_at(2), 21.0)
        self.assertEqual(tp.x1_at(2), 22.0)
        self.assertEqual(tp.y1_at(2), 23.0)
        self.assertEqual(tp.z1_at(2), 24.0)
        self.assertEqual(tp.x2_at(2), 25.0)
        self.assertEqual(tp.y2_at(2), 26.0)
        self.assertEqual(tp.z2_at(2), 27.0)

    def test_mirror_2d_x(self):
        TrianglePad = self.TrianglePad

        tp = TrianglePad(ndim=2)
        tp.append(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

        tp.mirror('x')

        self.assert_allclose(tp.x0_at(0), 1.0)
        self.assert_allclose(tp.y0_at(0), -2.0)
        self.assert_allclose(tp.x1_at(0), 3.0)
        self.assert_allclose(tp.y1_at(0), -4.0)
        self.assert_allclose(tp.x2_at(0), 5.0)
        self.assert_allclose(tp.y2_at(0), -6.0)

    def test_mirror_2d_y(self):
        TrianglePad = self.TrianglePad

        tp = TrianglePad(ndim=2)
        tp.append(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)

        tp.mirror('y')

        self.assert_allclose(tp.x0_at(0), -1.0)
        self.assert_allclose(tp.y0_at(0), 2.0)
        self.assert_allclose(tp.x1_at(0), -3.0)
        self.assert_allclose(tp.y1_at(0), 4.0)
        self.assert_allclose(tp.x2_at(0), -5.0)
        self.assert_allclose(tp.y2_at(0), 6.0)

    def test_mirror_3d_z(self):
        TrianglePad = self.TrianglePad

        tp = TrianglePad(ndim=3)
        tp.append(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)

        tp.mirror('z')

        self.assert_allclose(tp.x0_at(0), -1.0)
        self.assert_allclose(tp.y0_at(0), -2.0)
        self.assert_allclose(tp.z0_at(0), 3.0)
        self.assert_allclose(tp.x1_at(0), -4.0)
        self.assert_allclose(tp.y1_at(0), -5.0)
        self.assert_allclose(tp.z1_at(0), 6.0)
        self.assert_allclose(tp.x2_at(0), -7.0)
        self.assert_allclose(tp.y2_at(0), -8.0)
        self.assert_allclose(tp.z2_at(0), 9.0)

        with self.assertRaisesRegex(
                ValueError,
                "TrianglePad::mirror: axis must be 'x', 'y', or 'z'"):
            tp.mirror('w')

    def test_getitem_index(self):
        TrianglePad = self.TrianglePad

        tp2d = TrianglePad(ndim=2)
        tp2d.append(0, 1, 2, 3, 4, 5)
        tp2d.append(6, 7, 8, 9, 10, 11)
        tp2d.append(12, 13, 14, 15, 16, 17)
        tp2d.append(18, 19, 20, 21, 22, 23)

        self.assert_allclose(list(tp2d[0]),
                             [[0, 1, 0], [2, 3, 0], [4, 5, 0]])
        self.assert_allclose(list(tp2d[-1]),
                             [[18, 19, 0], [20, 21, 0], [22, 23, 0]])
        self.assert_allclose(list(tp2d[-4]),
                             [[0, 1, 0], [2, 3, 0], [4, 5, 0]])

        with self.assertRaisesRegex(
                IndexError,
                "TrianglePad: index 4 is out of bounds with size 4"):
            tp2d[4]
        with self.assertRaisesRegex(
                IndexError,
                "TrianglePad: index -5 is out of bounds with size 4"):
            tp2d[-5]

        empty = TrianglePad(ndim=3)
        with self.assertRaisesRegex(
                IndexError,
                "TrianglePad: index 0 is out of bounds with size 0"):
            empty[0]
        with self.assertRaisesRegex(
                IndexError,
                "TrianglePad: index -1 is out of bounds with size 0"):
            empty[-1]

    def test_getitem_slice(self):
        TrianglePad = self.TrianglePad

        tp3d = TrianglePad(ndim=3)
        tp3d.append(0, 1, 2, 3, 4, 5, 6, 7, 8)
        tp3d.append(9, 10, 11, 12, 13, 14, 15, 16, 17)
        tp3d.append(18, 19, 20, 21, 22, 23, 24, 25, 26)
        tp3d.append(27, 28, 29, 30, 31, 32, 33, 34, 35)

        full = tp3d[:]
        self.assertIsInstance(full, type(tp3d))
        self.assertEqual(full.ndim, 3)
        self.assertEqual(len(full), 4)
        for it in range(4):
            self.assertEqual(full[it], tp3d[it])

        strided = tp3d[1::2]
        self.assertEqual(strided.ndim, 3)
        self.assertEqual(len(strided), 2)
        self.assertEqual(strided[0], tp3d[1])
        self.assertEqual(strided[1], tp3d[3])

        flipped = tp3d[::-1]
        self.assert_allclose(list(flipped.x0), [27, 18, 9, 0])

        negative = tp3d[-2:-1]
        self.assertEqual(len(negative), 1)
        self.assertEqual(negative[0], tp3d[2])

        # A slice copies, so writing to it leaves the source alone.
        part = tp3d[1:3]
        part.set_at(0, -1.0, -2.0, -3.0, -4.0, -5.0,
                    -6.0, -7.0, -8.0, -9.0)
        self.assert_allclose(part.x0_at(0), -1.0)
        self.assert_allclose(tp3d.x0_at(1), 9.0)

        # A 2D pad keeps its dimensionality and its empty z arrays.
        tp2d = TrianglePad(ndim=2)
        tp2d.append(1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
        sliced2d = tp2d[:]
        self.assertEqual(sliced2d.ndim, 2)
        self.assertEqual(len(sliced2d.z0), 0)
        self.assertEqual(len(sliced2d.z1), 0)
        self.assertEqual(len(sliced2d.z2), 0)
        self.assertEqual(sliced2d[0], tp2d[0])

        empty = TrianglePad(ndim=3)
        self.assertEqual(len(empty[:]), 0)
        self.assertEqual(empty[:].ndim, 3)
        self.assertEqual(len(tp3d[3:1]), 0)


class TrianglePadFp32TC(TrianglePadTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float32'
        self.SimpleArray = solvcon.SimpleArrayFloat32
        self.Point = solvcon.Point3dFp32
        self.Triangle = solvcon.Triangle3dFp32
        self.TrianglePad = solvcon.TrianglePadFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)


class TrianglePadFp64TC(TrianglePadTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float64'
        self.SimpleArray = solvcon.SimpleArrayFloat64
        self.Point = solvcon.Point3dFp64
        self.Triangle = solvcon.Triangle3dFp64
        self.TrianglePad = solvcon.TrianglePadFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)


class Polygon3dTB(testing.TestBase):

    def test_polygon_pad_basic(self):
        """Test PolygonPad with basic operations."""
        pad = self.PolygonPad(ndim=2)
        self.assertEqual(pad.ndim, 2)
        self.assertEqual(pad.num_polygons, 0)
        self.assertEqual(pad.num_points, 0)

        nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        polygon = pad.add_polygon(nodes)

        self.assertEqual(pad.num_polygons, 1)
        self.assertEqual(pad.num_points, 4)
        self.assertEqual(polygon.nnode, 4)

        node0 = polygon.get_node(0)
        self.assert_allclose([node0.x, node0.y], [0.0, 0.0])

        node1 = polygon.get_node(1)
        self.assert_allclose([node1.x, node1.y], [1.0, 0.0])

    def test_polygon_handle_operations(self):
        """Test Polygon3d handle operations."""
        pad = self.PolygonPad(ndim=2)
        nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        polygon = pad.add_polygon(nodes)

        self.assertEqual(polygon.ndim, 2)
        self.assertEqual(polygon.polygon_id, 0)
        self.assertEqual(polygon.nnode, 4)

        node0 = polygon.get_node(0)
        self.assert_allclose([node0.x, node0.y], [0.0, 0.0])

        edge0 = polygon.get_edge(0)
        self.assert_allclose([edge0.x0, edge0.y0], [0.0, 0.0])
        self.assert_allclose([edge0.x1, edge0.y1], [1.0, 0.0])

        edge3 = polygon.get_edge(3)
        self.assert_allclose([edge3.x0, edge3.y0], [0.0, 1.0])
        self.assert_allclose([edge3.x1, edge3.y1], [0.0, 0.0])

    def test_right_hand_rule_counter_clockwise(self):
        """
        Test right-hand rule validation for counter-clockwise
        (positive area) polygon.
        """
        pad = self.PolygonPad(ndim=2)
        nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        polygon = pad.add_polygon(nodes)

        area = polygon.compute_signed_area()
        self.assertGreater(area, 0.0)
        self.assertTrue(polygon.is_counter_clockwise())

    def test_right_hand_rule_clockwise(self):
        """Test right-hand rule: clockwise nodes should have negative area."""
        pad = self.PolygonPad(ndim=2)
        nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(0.0, 1.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(1.0, 0.0, 0.0)
        ]
        polygon = pad.add_polygon(nodes)

        area = polygon.compute_signed_area()
        self.assertLess(area, 0.0)
        self.assertFalse(polygon.is_counter_clockwise())

    def test_multiple_polygons_in_pad(self):
        """Test storing multiple polygons in one PolygonPad."""
        pad = self.PolygonPad(ndim=2)

        square = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        triangle = [
            self.Point(2.0, 0.0, 0.0),
            self.Point(3.0, 0.0, 0.0),
            self.Point(2.5, 1.0, 0.0)
        ]

        poly1 = pad.add_polygon(square)
        poly2 = pad.add_polygon(triangle)

        self.assertEqual(pad.num_polygons, 2)
        self.assertEqual(pad.num_points, 7)

        self.assertEqual(poly1.nnode, 4)
        self.assertEqual(poly2.nnode, 3)

        self.assertGreater(poly1.compute_signed_area(), 0.0)
        self.assertGreater(poly2.compute_signed_area(), 0.0)

        retrieved_poly1 = pad.get_polygon(0)
        retrieved_poly2 = pad.get_polygon(1)

        self.assertEqual(retrieved_poly1.polygon_id, 0)
        self.assertEqual(retrieved_poly2.polygon_id, 1)

    def test_polygon_pad_from_segments(self):
        """Test adding polygon from SegmentPad."""
        segment_pad = self.SegmentPad(ndim=2)
        segment_pad.append(0.0, 0.0, 1.0, 0.0)
        segment_pad.append(1.0, 0.0, 1.0, 1.0)
        segment_pad.append(1.0, 1.0, 0.0, 1.0)
        segment_pad.append(0.0, 1.0, 0.0, 0.0)

        pad = self.PolygonPad(ndim=2)
        polygon = pad.add_polygon_from_segments(segment_pad)

        self.assertEqual(pad.num_polygons, 1)
        self.assertEqual(polygon.nnode, 4)

        node0 = polygon.get_node(0)
        self.assert_allclose([node0.x, node0.y], [0.0, 0.0])

    def test_polygon_pad_rtree_search(self):
        """Test RTree search across multiple polygons in PolygonPad."""
        pad = self.PolygonPad(ndim=2)

        square1 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        square2 = [
            self.Point(5.0, 5.0, 0.0),
            self.Point(6.0, 5.0, 0.0),
            self.Point(6.0, 6.0, 0.0),
            self.Point(5.0, 6.0, 0.0)
        ]

        pad.add_polygon(square1)
        pad.add_polygon(square2)

        BoundBox = (sc.BoundBox3dFp32 if self.dtype == 'float32'
                    else sc.BoundBox3dFp64)

        search_box1 = BoundBox(-0.5, -0.5, -0.5, 1.5, 1.5, 0.5)
        results1 = pad.search_segments(search_box1)
        self.assertEqual(len(results1), 4)

        search_box2 = BoundBox(4.5, 4.5, -0.5, 6.5, 6.5, 0.5)
        results2 = pad.search_segments(search_box2)
        self.assertEqual(len(results2), 4)

        search_box3 = BoundBox(-1.0, -1.0, -0.5, 7.0, 7.0, 0.5)
        results3 = pad.search_segments(search_box3)
        self.assertEqual(len(results3), 8)

    def test_polygon_equality_same_coordinates(self):
        """Test that polygons with same coordinates are equal."""
        pad1 = self.PolygonPad(ndim=2)
        pad2 = self.PolygonPad(ndim=2)

        nodes1 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        nodes2 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]

        polygon1 = pad1.add_polygon(nodes1)
        polygon2 = pad2.add_polygon(nodes2)

        self.assertTrue(polygon1 == polygon2)
        self.assertFalse(polygon1 != polygon2)

    def test_polygon_equality_different_coordinates(self):
        """Test that polygons with different coordinates are not equal."""
        pad1 = self.PolygonPad(ndim=2)
        pad2 = self.PolygonPad(ndim=2)

        nodes1 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        nodes2 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(2.0, 2.0, 0.0),
            self.Point(0.0, 2.0, 0.0)
        ]

        polygon1 = pad1.add_polygon(nodes1)
        polygon2 = pad2.add_polygon(nodes2)

        self.assertFalse(polygon1 == polygon2)
        self.assertTrue(polygon1 != polygon2)

    def test_polygon_equality_different_node_count(self):
        """Test that polygons with different node counts are not equal."""
        pad1 = self.PolygonPad(ndim=2)
        pad2 = self.PolygonPad(ndim=2)

        nodes1 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        nodes2 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(0.5, 1.0, 0.0)
        ]

        polygon1 = pad1.add_polygon(nodes1)
        polygon2 = pad2.add_polygon(nodes2)

        self.assertFalse(polygon1 == polygon2)
        self.assertTrue(polygon1 != polygon2)

    def test_polygon_identity_same_pad_same_id(self):
        """Test identity: same pad and same polygon_id."""
        pad = self.PolygonPad(ndim=2)

        nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]

        polygon1 = pad.add_polygon(nodes)
        polygon2 = pad.get_polygon(0)

        self.assertTrue(polygon1.is_same(polygon2))
        self.assertFalse(polygon1.is_not_same(polygon2))

    def test_polygon_identity_different_pads(self):
        """Test identity: different pads with same coordinates."""
        pad1 = self.PolygonPad(ndim=2)
        pad2 = self.PolygonPad(ndim=2)

        nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]

        polygon1 = pad1.add_polygon(nodes)
        polygon2 = pad2.add_polygon(nodes)

        self.assertTrue(polygon1 == polygon2)
        self.assertFalse(polygon1.is_same(polygon2))
        self.assertTrue(polygon1.is_not_same(polygon2))

    def test_polygon_identity_same_pad_different_id(self):
        """Test identity: same pad but different polygon_id."""
        pad = self.PolygonPad(ndim=2)

        nodes1 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        nodes2 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]

        polygon1 = pad.add_polygon(nodes1)
        polygon2 = pad.add_polygon(nodes2)

        self.assertTrue(polygon1 == polygon2)
        self.assertFalse(polygon1.is_same(polygon2))
        self.assertTrue(polygon1.is_not_same(polygon2))

    def _compute_total_area(self, result_pad):
        """Helper to compute total unsigned area of all polygons in a pad."""
        total = 0.0
        for i in range(result_pad.num_polygons):
            total += abs(result_pad.get_polygon(i).compute_signed_area())
        return total

    def _assert_all_ccw(self, result_pad):
        """Assert all result polygons have counter-clockwise winding."""
        for i in range(result_pad.num_polygons):
            area = result_pad.get_polygon(i).compute_signed_area()
            self.assertGreaterEqual(area, 0.0,
                                    f"Polygon {i} has negative area {area}"
                                    " (clockwise winding)")

    def test_boolean_union_simple(self):
        """Test polygon boolean union with two overlapping squares."""
        pad = self.PolygonPad(ndim=2)

        # First square: (0,0) to (2,2), area = 4
        square1_nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(2.0, 2.0, 0.0),
            self.Point(0.0, 2.0, 0.0)
        ]
        polygon1 = pad.add_polygon(square1_nodes)

        # Second square: (1,1) to (3,3), area = 4
        # Overlap region: (1,1) to (2,2), area = 1
        # Union area = 4 + 4 - 1 = 7
        square2_nodes = [
            self.Point(1.0, 1.0, 0.0),
            self.Point(3.0, 1.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(1.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(square2_nodes)

        result = pad.boolean_union(polygon1, polygon2)
        self.assertIsInstance(result, self.PolygonPad)
        self.assertGreater(result.num_polygons, 0)
        total_area = self._compute_total_area(result)
        self.assert_allclose([total_area], [7.0], rtol=1e-6)

    def test_boolean_intersection_simple(self):
        """Test polygon boolean intersection with two overlapping squares."""
        pad = self.PolygonPad(ndim=2)

        # First square: (0,0) to (2,2)
        square1_nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(2.0, 2.0, 0.0),
            self.Point(0.0, 2.0, 0.0)
        ]
        polygon1 = pad.add_polygon(square1_nodes)

        # Second square: (1,1) to (3,3)
        # Intersection should be (1,1) to (2,2), area = 1
        square2_nodes = [
            self.Point(1.0, 1.0, 0.0),
            self.Point(3.0, 1.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(1.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(square2_nodes)

        result = pad.boolean_intersection(polygon1, polygon2)
        self.assertIsInstance(result, self.PolygonPad)
        self.assertGreater(result.num_polygons, 0)
        total_area = self._compute_total_area(result)
        self.assert_allclose([total_area], [1.0], rtol=1e-6)

    def test_boolean_difference_simple(self):
        """Test polygon boolean difference with two overlapping squares."""
        pad = self.PolygonPad(ndim=2)

        # First square: (0,0) to (2,2), area = 4
        square1_nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(2.0, 2.0, 0.0),
            self.Point(0.0, 2.0, 0.0)
        ]
        polygon1 = pad.add_polygon(square1_nodes)

        # Second square: (1,1) to (3,3)
        # Difference (polygon1 - polygon2) = L-shaped region, area = 4 - 1 = 3
        square2_nodes = [
            self.Point(1.0, 1.0, 0.0),
            self.Point(3.0, 1.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(1.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(square2_nodes)

        result = pad.boolean_difference(polygon1, polygon2)
        self.assertIsInstance(result, self.PolygonPad)
        self.assertGreater(result.num_polygons, 0)
        total_area = self._compute_total_area(result)
        self.assert_allclose([total_area], [3.0], rtol=1e-6)

    def test_boolean_union_non_overlapping(self):
        """Test polygon boolean union with two non-overlapping squares."""
        pad = self.PolygonPad(ndim=2)

        # First square: (0,0) to (1,1), area = 1
        square1_nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        polygon1 = pad.add_polygon(square1_nodes)

        # Second square: (2,2) to (3,3), area = 1
        # No overlap, union area = 2
        square2_nodes = [
            self.Point(2.0, 2.0, 0.0),
            self.Point(3.0, 2.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(2.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(square2_nodes)

        result = pad.boolean_union(polygon1, polygon2)
        self.assertIsInstance(result, self.PolygonPad)
        total_area = self._compute_total_area(result)
        self.assert_allclose([total_area], [2.0], rtol=1e-6)

    def test_boolean_intersection_non_overlapping(self):
        """Test polygon boolean intersection with two non-overlapping squares.
        """
        pad = self.PolygonPad(ndim=2)

        # First square: (0,0) to (1,1)
        square1_nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        polygon1 = pad.add_polygon(square1_nodes)

        # Second square: (2,2) to (3,3) - no overlap
        # Intersection should be empty
        square2_nodes = [
            self.Point(2.0, 2.0, 0.0),
            self.Point(3.0, 2.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(2.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(square2_nodes)

        result = pad.boolean_intersection(polygon1, polygon2)
        self.assertIsInstance(result, self.PolygonPad)
        self.assertEqual(result.num_polygons, 0)

    def test_boolean_operations_triangle(self):
        """Test polygon boolean operations with triangular polygons."""
        pad = self.PolygonPad(ndim=2)

        # Triangle 1: (0,0)-(2,0)-(0,2), area = 2
        triangle1_nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(0.0, 2.0, 0.0)
        ]
        polygon1 = pad.add_polygon(triangle1_nodes)

        # Triangle 2: (1,1)-(3,1)-(1,3), area = 2
        triangle2_nodes = [
            self.Point(1.0, 1.0, 0.0),
            self.Point(3.0, 1.0, 0.0),
            self.Point(1.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(triangle2_nodes)

        # Intersection: the overlap region is x>=1, y>=1, x+y<=2.
        # Since x>=1 and y>=1 imply x+y>=2, the only solution is the single
        # point (1,1). The two triangles touch at exactly one point, so
        # the intersection area is 0.
        result_intersection = pad.boolean_intersection(polygon1, polygon2)
        self.assertIsInstance(result_intersection, self.PolygonPad)
        intersection_area = self._compute_total_area(result_intersection)
        self.assertAlmostEqual(intersection_area, 0.0, places=5)

        # Union area = 2 + 2 - 0 = 4
        result_union = pad.boolean_union(polygon1, polygon2)
        self.assertIsInstance(result_union, self.PolygonPad)
        union_area = self._compute_total_area(result_union)
        self.assert_allclose([union_area], [4.0], rtol=1e-6)

        # Difference area = 2 - 0 = 2
        result_difference = pad.boolean_difference(polygon1, polygon2)
        self.assertIsInstance(result_difference, self.PolygonPad)
        diff_area = self._compute_total_area(result_difference)
        self.assert_allclose([diff_area], [2.0], rtol=1e-6)

    def test_boolean_containment(self):
        """Test boolean operations when one polygon contains the other."""
        pad = self.PolygonPad(ndim=2)

        # Large square: (0,0) to (4,4), area = 16
        large_nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(4.0, 0.0, 0.0),
            self.Point(4.0, 4.0, 0.0),
            self.Point(0.0, 4.0, 0.0)
        ]
        polygon1 = pad.add_polygon(large_nodes)

        # Small square: (1,1) to (3,3), area = 4
        small_nodes = [
            self.Point(1.0, 1.0, 0.0),
            self.Point(3.0, 1.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(1.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(small_nodes)

        # Union = large square, area = 16
        result_union = pad.boolean_union(polygon1, polygon2)
        union_area = self._compute_total_area(result_union)
        self.assert_allclose([union_area], [16.0], rtol=1e-6)

        # Intersection = small square, area = 4
        result_inter = pad.boolean_intersection(polygon1, polygon2)
        inter_area = self._compute_total_area(result_inter)
        self.assert_allclose([inter_area], [4.0], rtol=1e-6)

        # Difference = large - small = 12
        result_diff = pad.boolean_difference(polygon1, polygon2)
        diff_area = self._compute_total_area(result_diff)
        self.assert_allclose([diff_area], [12.0], rtol=1e-6)

    def test_boolean_identical_polygons(self):
        """Test boolean operations when both polygons are identical."""
        pad = self.PolygonPad(ndim=2)

        nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(3.0, 0.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(0.0, 3.0, 0.0)
        ]
        polygon1 = pad.add_polygon(nodes)
        polygon2 = pad.add_polygon(nodes)

        # Union(P, P) = P, area = 9
        result_union = pad.boolean_union(polygon1, polygon2)
        self.assert_allclose(
            [self._compute_total_area(result_union)], [9.0], rtol=1e-6)

        # Intersection(P, P) = P, area = 9
        result_inter = pad.boolean_intersection(polygon1, polygon2)
        self.assert_allclose(
            [self._compute_total_area(result_inter)], [9.0], rtol=1e-6)

        # Difference(P, P) = empty
        result_diff = pad.boolean_difference(polygon1, polygon2)
        self.assertEqual(result_diff.num_polygons, 0)

    def test_boolean_shared_edge(self):
        """Test boolean operations with two squares sharing an edge."""
        pad = self.PolygonPad(ndim=2)

        # Square 1: (0,0)-(1,1)
        nodes1 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(1.0, 0.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(0.0, 1.0, 0.0)
        ]
        polygon1 = pad.add_polygon(nodes1)

        # Square 2: (1,0)-(2,1), shares edge at x=1
        nodes2 = [
            self.Point(1.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(2.0, 1.0, 0.0),
            self.Point(1.0, 1.0, 0.0)
        ]
        polygon2 = pad.add_polygon(nodes2)

        # Union = (0,0)-(2,1), area = 2
        result_union = pad.boolean_union(polygon1, polygon2)
        self.assert_allclose(
            [self._compute_total_area(result_union)], [2.0], rtol=1e-6)

        # Intersection: touching at edge only, area = 0
        result_inter = pad.boolean_intersection(polygon1, polygon2)
        inter_area = self._compute_total_area(result_inter)
        self.assertAlmostEqual(inter_area, 0.0, places=5)

        # Difference(A, B) = square 1, area = 1
        result_diff = pad.boolean_difference(polygon1, polygon2)
        self.assert_allclose(
            [self._compute_total_area(result_diff)], [1.0], rtol=1e-6)

    def test_boolean_commutativity(self):
        """Test that union and intersection are commutative,
        and difference is non-commutative."""
        pad = self.PolygonPad(ndim=2)

        nodes1 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(2.0, 2.0, 0.0),
            self.Point(0.0, 2.0, 0.0)
        ]
        polygon1 = pad.add_polygon(nodes1)

        nodes2 = [
            self.Point(1.0, 1.0, 0.0),
            self.Point(3.0, 1.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(1.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(nodes2)

        # Union(A, B) == Union(B, A)
        union_ab = self._compute_total_area(
            pad.boolean_union(polygon1, polygon2))
        union_ba = self._compute_total_area(
            pad.boolean_union(polygon2, polygon1))
        self.assert_allclose([union_ab], [union_ba], rtol=1e-6)

        # Intersection(A, B) == Intersection(B, A)
        inter_ab = self._compute_total_area(
            pad.boolean_intersection(polygon1, polygon2))
        inter_ba = self._compute_total_area(
            pad.boolean_intersection(polygon2, polygon1))
        self.assert_allclose([inter_ab], [inter_ba], rtol=1e-6)

        # Difference is non-commutative: A-B=3, B-A=3 in area but
        # they are different regions. Verify areas are as expected.
        diff_ab = self._compute_total_area(
            pad.boolean_difference(polygon1, polygon2))
        diff_ba = self._compute_total_area(
            pad.boolean_difference(polygon2, polygon1))
        self.assert_allclose([diff_ab], [3.0], rtol=1e-6)
        self.assert_allclose([diff_ba], [3.0], rtol=1e-6)

    def test_boolean_partial_vertical_overlap(self):
        """Test with rectangles that partially overlap in Y."""
        pad = self.PolygonPad(ndim=2)

        # Rectangle 1: (0,0)-(2,3), area = 6
        nodes1 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(2.0, 3.0, 0.0),
            self.Point(0.0, 3.0, 0.0)
        ]
        polygon1 = pad.add_polygon(nodes1)

        # Rectangle 2: (1,1)-(3,2), area = 2
        # Overlap region: (1,1)-(2,2), area = 1
        nodes2 = [
            self.Point(1.0, 1.0, 0.0),
            self.Point(3.0, 1.0, 0.0),
            self.Point(3.0, 2.0, 0.0),
            self.Point(1.0, 2.0, 0.0)
        ]
        polygon2 = pad.add_polygon(nodes2)

        # Union = 6 + 2 - 1 = 7
        result_union = pad.boolean_union(polygon1, polygon2)
        self._assert_all_ccw(result_union)
        self.assert_allclose(
            [self._compute_total_area(result_union)], [7.0], rtol=1e-6)

        # Intersection = 1
        result_inter = pad.boolean_intersection(polygon1, polygon2)
        self._assert_all_ccw(result_inter)
        self.assert_allclose(
            [self._compute_total_area(result_inter)], [1.0], rtol=1e-6)

        # Difference = 6 - 1 = 5
        result_diff = pad.boolean_difference(polygon1, polygon2)
        self._assert_all_ccw(result_diff)
        self.assert_allclose(
            [self._compute_total_area(result_diff)], [5.0], rtol=1e-6)

    def test_boolean_concave_polygon(self):
        """Test boolean operations with a concave (L-shaped) polygon."""
        pad = self.PolygonPad(ndim=2)

        # L-shaped polygon (concave):
        # (0,0)-(2,0)-(2,1)-(1,1)-(1,2)-(0,2), area = 3
        l_nodes = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(2.0, 1.0, 0.0),
            self.Point(1.0, 1.0, 0.0),
            self.Point(1.0, 2.0, 0.0),
            self.Point(0.0, 2.0, 0.0)
        ]
        polygon1 = pad.add_polygon(l_nodes)

        # Square overlapping the L: (0.5, 0.5)-(1.5, 1.5), area = 1
        sq_nodes = [
            self.Point(0.5, 0.5, 0.0),
            self.Point(1.5, 0.5, 0.0),
            self.Point(1.5, 1.5, 0.0),
            self.Point(0.5, 1.5, 0.0)
        ]
        polygon2 = pad.add_polygon(sq_nodes)

        # The square partially extends outside the L's notch:
        #   Band [0.5, 1.0]: full square width (0.5 to 1.5), height 0.5 -> 0.5
        #   Band [1.0, 1.5]: only x in [0.5, 1.0] (L's left column), -> 0.25
        # Intersection area = 0.75
        result_inter = pad.boolean_intersection(polygon1, polygon2)
        inter_area = self._compute_total_area(result_inter)
        self.assert_allclose([inter_area], [0.75], rtol=1e-6)

        # Union = L + square - intersection = 3 + 1 - 0.75 = 3.25
        result_union = pad.boolean_union(polygon1, polygon2)
        union_area = self._compute_total_area(result_union)
        self.assert_allclose([union_area], [3.25], rtol=1e-6)

        # Difference(L, square) = L - intersection = 3 - 0.75 = 2.25
        result_diff = pad.boolean_difference(polygon1, polygon2)
        diff_area = self._compute_total_area(result_diff)
        self.assert_allclose([diff_area], [2.25], rtol=1e-6)

    def test_decomposition_edge_order(self):
        """Test that polygon decomposition produces non-inverted trapezoids.
        """
        pad = self.PolygonPad(ndim=2)

        # Triangle 1: apex at bottom y=0, base at y=2
        # Vertices CCW: (0,2) -> (4,2) -> (2,0)
        nodes1 = [
            self.Point(0.0, 2.0, 0.0),
            self.Point(4.0, 2.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
        ]
        _ = pad.add_polygon(nodes1)

        # Triangle 2: shifted right, apex at bottom y=0
        # Vertices CCW: (1,3) -> (5,3) -> (3,0)
        nodes2 = [
            self.Point(1.0, 3.0, 0.0),
            self.Point(5.0, 3.0, 0.0),
            self.Point(3.0, 0.0, 0.0),
        ]
        _ = pad.add_polygon(nodes2)

        # Verify decomposition produces non-inverted trapezoids
        # The apex of each triangles is at y=0,
        # so the x value at y=0 will be the same for both left and right edges.
        # Thus, the dxdy should be consider to prevent inverted trapezoids.
        trap_pad = pad.decomposed_trapezoids()
        for trap_id in range(trap_pad.size):
            # At the top of each trapezoid, left_x (p3) <= right_x (p2)
            self.assertLessEqual(
                trap_pad.x3(trap_id), trap_pad.x2(trap_id),
                f"Inverted trapezoid {trap_id}:"
                f"top-left x={trap_pad.x3(trap_id)}"
                f" > top-right x={trap_pad.x2(trap_id)}")

    def test_boolean_result_ccw_winding(self):
        """Test that all result polygons from boolean operations have
        counter - clockwise winding(non - negative signed area)."""
        pad = self.PolygonPad(ndim=2)

        nodes1 = [
            self.Point(0.0, 0.0, 0.0),
            self.Point(2.0, 0.0, 0.0),
            self.Point(2.0, 2.0, 0.0),
            self.Point(0.0, 2.0, 0.0)
        ]
        polygon1 = pad.add_polygon(nodes1)

        nodes2 = [
            self.Point(1.0, 1.0, 0.0),
            self.Point(3.0, 1.0, 0.0),
            self.Point(3.0, 3.0, 0.0),
            self.Point(1.0, 3.0, 0.0)
        ]
        polygon2 = pad.add_polygon(nodes2)

        self._assert_all_ccw(pad.boolean_union(polygon1, polygon2))
        self._assert_all_ccw(pad.boolean_intersection(polygon1, polygon2))
        self._assert_all_ccw(pad.boolean_difference(polygon1, polygon2))
        self._assert_all_ccw(pad.boolean_difference(polygon2, polygon1))


class Polygon3dFp32TC(Polygon3dTB, unittest.TestCase):
    dtype = 'float32'
    Point = sc.Point3dFp32
    SegmentPad = sc.SegmentPadFp32
    CurvePad = sc.CurvePadFp32
    PolygonPad = sc.PolygonPadFp32
    SimpleArray = sc.SimpleArrayFloat32


class Polygon3dFp64TC(Polygon3dTB, unittest.TestCase):
    dtype = 'float64'
    Point = sc.Point3dFp64
    SegmentPad = sc.SegmentPadFp64
    CurvePad = sc.CurvePadFp64
    PolygonPad = sc.PolygonPadFp64
    SimpleArray = sc.SimpleArrayFloat64


class TrapezoidalDecomposerTB(testing.TestBase):

    def _make_2d_points(self, *coords):
        """Helper to create Point objects from 2D coordinates.

        Args:
            *coords: Variable number of (x, y) tuples

        Returns:
            List of Point objects with z=0.0
        """
        return [self.Point(x, y, 0.0) for x, y in coords]

    def test_simple_rectangle(self):
        """Test decomposition of a simple rectangle."""
        decomposer = self.TrapezoidalDecomposer(2)

        rectangle = self._make_2d_points(
            (0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (0.0, 1.0))

        begin_index, end_index = decomposer.decompose(0, rectangle)

        self.assertEqual(decomposer.num_trapezoids(0), 1)
        self.assertEqual(begin_index, 0)
        self.assertEqual(end_index, 1)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 1)

        self.assert_allclose(trapezoids.x0(0), 0.0)
        self.assert_allclose(trapezoids.y0(0), 0.0)
        self.assert_allclose(trapezoids.x1(0), 2.0)
        self.assert_allclose(trapezoids.y1(0), 0.0)
        self.assert_allclose(trapezoids.x2(0), 2.0)
        self.assert_allclose(trapezoids.y2(0), 1.0)
        self.assert_allclose(trapezoids.x3(0), 0.0)
        self.assert_allclose(trapezoids.y3(0), 1.0)

    def test_simple_triangle(self):
        """Test decomposition of a simple triangle."""
        decomposer = self.TrapezoidalDecomposer(2)

        triangle = self._make_2d_points((0.0, 0.0), (2.0, 0.0), (1.0, 2.0))

        begin_index, end_index = decomposer.decompose(0, triangle)

        self.assertEqual(decomposer.num_trapezoids(0), 1)
        self.assertEqual(begin_index, 0)
        self.assertEqual(end_index, 1)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 1)

        self.assert_allclose(trapezoids.y0(0), 0.0)
        self.assert_allclose(trapezoids.y1(0), 0.0)
        self.assert_allclose(trapezoids.x0(0), 0.0)
        self.assert_allclose(trapezoids.x1(0), 2.0)

        self.assert_allclose(trapezoids.y2(0), 2.0)
        self.assert_allclose(trapezoids.y3(0), 2.0)
        self.assert_allclose(trapezoids.x2(0), 1.0)
        self.assert_allclose(trapezoids.x3(0), 1.0)

    def test_trapezoid_shape(self):
        """Test decomposition of a trapezoid shape."""
        decomposer = self.TrapezoidalDecomposer(2)

        trapezoid = self._make_2d_points(
            (0.0, 0.0), (4.0, 0.0), (3.0, 2.0), (1.0, 2.0))

        begin_index, end_index = decomposer.decompose(0, trapezoid)

        self.assertEqual(decomposer.num_trapezoids(0), 1)
        self.assertEqual(begin_index, 0)
        self.assertEqual(end_index, 1)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 1)

        self.assert_allclose(trapezoids.y0(0), 0.0)
        self.assert_allclose(trapezoids.y1(0), 0.0)
        self.assert_allclose(trapezoids.y2(0), 2.0)
        self.assert_allclose(trapezoids.y3(0), 2.0)

        self.assert_allclose(trapezoids.x0(0), 0.0)
        self.assert_allclose(trapezoids.x1(0), 4.0)
        self.assert_allclose(trapezoids.x2(0), 3.0)
        self.assert_allclose(trapezoids.x3(0), 1.0)

    def test_pentagon(self):
        """Test decomposition of a pentagon."""
        decomposer = self.TrapezoidalDecomposer(2)

        pentagon = self._make_2d_points(
            (0.0, 0.0), (3.0, 0.0), (4.0, 2.0), (1.5, 3.0), (-1.0, 2.0))

        begin_index, end_index = decomposer.decompose(0, pentagon)

        self.assertEqual(decomposer.num_trapezoids(0), 2)
        self.assertEqual(begin_index, 0)
        self.assertEqual(end_index, 2)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 2)

    def test_concave_polygon(self):
        """Test decomposition of a concave polygon."""
        decomposer = self.TrapezoidalDecomposer(2)

        concave = self._make_2d_points(
            (0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (2.0, 1.5), (0.0, 3.0))

        begin_index, end_index = decomposer.decompose(0, concave)

        self.assertEqual(decomposer.num_trapezoids(0), 3)
        self.assertEqual(begin_index, 0)
        self.assertEqual(end_index, 3)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 3)

    def test_multiple_polygons(self):
        """Test decomposition of multiple polygons."""
        decomposer = self.TrapezoidalDecomposer(2)

        square1 = self._make_2d_points(
            (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))

        square2 = self._make_2d_points(
            (2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0))

        begin1, end1 = decomposer.decompose(0, square1)
        begin2, end2 = decomposer.decompose(1, square2)

        self.assertEqual(decomposer.num_trapezoids(0), 1)
        self.assertEqual(decomposer.num_trapezoids(1), 1)

        self.assertEqual(begin1, 0)
        self.assertEqual(end1, 1)
        self.assertEqual(begin2, 1)
        self.assertEqual(end2, 2)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 2)

    def test_cached_decomposition(self):
        """Test that decomposition results are cached."""
        decomposer = self.TrapezoidalDecomposer(2)

        square = self._make_2d_points(
            (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))

        begin1, end1 = decomposer.decompose(0, square)
        begin2, end2 = decomposer.decompose(0, square)

        self.assertEqual(begin1, begin2)
        self.assertEqual(end1, end2)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 1)

    def test_clear(self):
        """Test clearing decomposition results."""
        decomposer = self.TrapezoidalDecomposer(2)

        square = self._make_2d_points(
            (0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))

        decomposer.decompose(0, square)
        self.assertEqual(decomposer.num_trapezoids(0), 1)

        decomposer.clear()

        self.assertEqual(decomposer.num_trapezoids(0), 0)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 0)

    def test_horizontal_edges_ignored(self):
        """Test that horizontal edges are ignored during decomposition."""
        decomposer = self.TrapezoidalDecomposer(2)

        shape_with_horizontal = self._make_2d_points(
            (0.0, 0.0), (2.0, 0.0), (3.0, 1.0), (2.0, 1.0), (0.0, 1.0))

        begin_index, end_index = decomposer.decompose(0, shape_with_horizontal)

        self.assertEqual(decomposer.num_trapezoids(0), 1)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 1)

    def test_floating_point_precision(self):
        """Test decomposition with different floating point precisions."""
        decomposer_fp32 = sc.TrapezoidalDecomposerFp32(2)
        decomposer_fp64 = self.TrapezoidalDecomposer(2)

        triangle_fp32 = [
            sc.Point3dFp32(0.0, 0.0, 0.0),
            sc.Point3dFp32(1.0, 0.0, 0.0),
            sc.Point3dFp32(0.5, 1.0, 0.0)
        ]

        triangle_fp64 = self._make_2d_points(
            (0.0, 0.0), (1.0, 0.0), (0.5, 1.0))

        begin32, end32 = decomposer_fp32.decompose(0, triangle_fp32)
        begin64, end64 = decomposer_fp64.decompose(0, triangle_fp64)

        self.assertEqual(decomposer_fp32.num_trapezoids(0),
                         decomposer_fp64.num_trapezoids(0))

    def test_complex_polygon_with_many_vertices(self):
        """Test decomposition of a complex polygon with many vertices."""
        decomposer = self.TrapezoidalDecomposer(2)

        octagon = self._make_2d_points(
            (1.0, 0.0), (2.0, 0.0), (3.0, 1.0), (3.0, 2.0),
            (2.0, 3.0), (1.0, 3.0), (0.0, 2.0), (0.0, 1.0))

        begin_index, end_index = decomposer.decompose(0, octagon)

        self.assertEqual(decomposer.num_trapezoids(0), 3)
        self.assertEqual(begin_index, 0)
        self.assertEqual(end_index, 3)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 3)

    def test_l_shaped_polygon(self):
        """Test decomposition of an L-shaped polygon."""
        decomposer = self.TrapezoidalDecomposer(2)

        l_shape = self._make_2d_points(
            (0.0, 0.0), (2.0, 0.0), (2.0, 1.0),
            (1.0, 1.0), (1.0, 3.0), (0.0, 3.0))

        begin_index, end_index = decomposer.decompose(0, l_shape)

        self.assertEqual(decomposer.num_trapezoids(0), 2)
        self.assertEqual(begin_index, 0)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 2)

    def test_narrow_spike(self):
        """Test decomposition of a narrow spike."""
        decomposer = self.TrapezoidalDecomposer(2)

        spike = self._make_2d_points((0.0, 0.0), (1.0, 0.0), (0.5, 10.0))

        begin_index, end_index = decomposer.decompose(0, spike)

        self.assertEqual(decomposer.num_trapezoids(0), 1)

        trapezoids = decomposer.trapezoids()
        self.assertEqual(trapezoids.size, 1)


class TrapezoidalDecomposerFp32TC(TrapezoidalDecomposerTB, unittest.TestCase):

    def setUp(self):
        self.Point = sc.Point3dFp32
        self.TrapezoidalDecomposer = sc.TrapezoidalDecomposerFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)


class TrapezoidalDecomposerFp64TC(TrapezoidalDecomposerTB, unittest.TestCase):

    def setUp(self):
        self.Point = sc.Point3dFp64
        self.TrapezoidalDecomposer = sc.TrapezoidalDecomposerFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
