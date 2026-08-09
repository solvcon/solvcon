# Copyright (c) 2023, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

import numpy as np

import solvcon
from solvcon import testing


class BernsteinTB(testing.TestBase):
    pass


class BernsteinPolynomialTC(BernsteinTB, unittest.TestCase):

    def test_degree1(self):
        # linear basis, degree n = 1
        f = solvcon.calc_bernstein_polynomial

        self.assertEqual(1.0, f(t=0.0, i=0, n=1))
        self.assertEqual(0.0, f(t=1.0, i=0, n=1))
        self.assertEqual(0.0, f(t=0.0, i=1, n=1))
        self.assertEqual(1.0, f(t=1.0, i=1, n=1))

        def _check(t):
            self.assert_allclose((1 - t), f(t, 0, 1))
            self.assert_allclose(t, f(t, 1, 1))
            vsum = f(t, 0, 1) + f(t, 1, 1)
            self.assert_allclose(1.0, vsum)

        _check(t=0.1)
        _check(t=0.3)
        _check(t=0.5)
        _check(t=0.7)
        _check(t=0.9)

    def test_degree2(self):
        # quadratic basis, degree n = 2
        f = solvcon.calc_bernstein_polynomial

        self.assert_allclose(1.0, f(t=0.0, i=0, n=2))
        self.assert_allclose(0.0, f(t=1.0, i=0, n=2))
        self.assert_allclose(0.0, f(t=0.0, i=2, n=2))
        self.assert_allclose(1.0, f(t=1.0, i=2, n=2))

        def _check(t):
            self.assert_allclose((1 - t) ** 2, f(t, 0, 2))
            self.assert_allclose(2 * (1 - t) * t, f(t, 1, 2))
            self.assert_allclose(t ** 2, f(t, 2, 2))
            vsum = f(t, 0, 2) + f(t, 1, 2) + f(t, 2, 2)
            self.assert_allclose(1.0, vsum)

        _check(t=0.1)
        _check(t=0.3)
        _check(t=0.5)
        _check(t=0.7)
        _check(t=0.9)

    def test_degree3(self):
        # cubic basis, degree n = 3
        f = solvcon.calc_bernstein_polynomial

        self.assertEqual(1.0, f(t=0.0, i=0, n=3))
        self.assertEqual(0.0, f(t=1.0, i=0, n=3))
        self.assertEqual(0.0, f(t=0.0, i=3, n=3))
        self.assertEqual(1.0, f(t=1.0, i=3, n=3))

        def _check(t):
            self.assert_allclose((1 - t) ** 3, f(t, 0, 3))
            self.assert_allclose(3 * ((1 - t) ** 2) * t, f(t, 1, 3))
            self.assert_allclose(3 * (1 - t) * (t ** 2), f(t, 2, 3))
            self.assert_allclose(t ** 3, f(t, 3, 3))
            vsum = f(t, 0, 3) + f(t, 1, 3) + f(t, 2, 3) + f(t, 3, 3)
            self.assert_allclose(1.0, vsum)

        _check(t=0.1)
        _check(t=0.3)
        _check(t=0.5)
        _check(t=0.7)
        _check(t=0.9)


class BernsteinInterpolationTC(BernsteinTB, unittest.TestCase):

    def test_degree1(self):
        # linear basis, degree n = 1
        f = solvcon.interpolate_bernstein

        def _check(t, values):
            golden = values[0] * (1 - t) + values[1] * t,
            self.assert_allclose(golden, f(t=t, values=values, n=1))

        values = [1.0, 2.0]
        self.assertEqual(values[0], f(t=0.0, values=values, n=1))
        self.assertEqual(values[1], f(t=1.0, values=values, n=1))
        _check(t=0.1, values=values)
        _check(t=0.3, values=values)
        _check(t=0.5, values=values)
        _check(t=0.7, values=values)
        _check(t=0.9, values=values)

    def test_degree2(self):
        # quadratic basis, degree n = 2
        f = solvcon.interpolate_bernstein

        def _check(t, values):
            golden = values[0] * (1 - t) ** 2
            golden += values[1] * 2 * (1 - t) * t
            golden += values[2] * t ** 2
            self.assert_allclose(golden, f(t=t, values=values, n=2))

        values = [1.0, 2.0, 3.0]
        self.assertEqual(values[0], f(t=0.0, values=values, n=2))
        self.assertEqual(values[2], f(t=1.0, values=values, n=2))
        _check(t=0.1, values=values)
        _check(t=0.3, values=values)
        _check(t=0.5, values=values)
        _check(t=0.7, values=values)
        _check(t=0.9, values=values)

    def test_degree3(self):
        # cubic basis, degree n = 3
        f = solvcon.interpolate_bernstein

        def _check(t, values):
            golden = values[0] * (1 - t) ** 3
            golden += values[1] * 3 * ((1 - t) ** 2) * t
            golden += values[2] * 3 * (1 - t) * (t ** 2)
            golden += values[3] * t ** 3
            self.assert_allclose(golden, f(t=t, values=values, n=3))

        values = [1.0, 2.0, 3.0, 4.0]
        self.assertEqual(values[0], f(t=0.0, values=values, n=3))
        self.assertEqual(values[3], f(t=1.0, values=values, n=3))
        _check(t=0.1, values=values)
        _check(t=0.3, values=values)
        _check(t=0.5, values=values)
        _check(t=0.7, values=values)
        _check(t=0.9, values=values)


class Point3dTB(testing.TestBase):

    def test_construct(self):
        Point = self.Point

        # Construct using positional arguments
        pnt = Point(1, 2, 3)
        self.assertEqual(pnt.x, 1.0)
        self.assertEqual(pnt.y, 2.0)
        self.assertEqual(pnt.z, 3.0)

        # Construct using keyword arguments
        pnt = Point(x=2.2, y=5.8, z=-9.22)
        self.assert_allclose(pnt, [2.2, 5.8, -9.22])
        self.assert_allclose(pnt[0], 2.2)
        self.assert_allclose(pnt[1], 5.8)
        self.assert_allclose(pnt[2], -9.22)
        self.assertEqual(len(pnt), 3)

        # Range error in C++
        with self.assertRaisesRegex(IndexError, "Point3d: i 3 >= size 3"):
            pnt[3]

    def test_fill(self):
        Point = self.Point

        pnt = Point(1, 2, 3)
        pnt.fill(10.0)
        self.assertEqual(list(pnt), [10, 10, 10])

    def test_arithmetic(self):
        Point = self.Point
        p1 = Point(1, 2, 3)
        p2 = Point(4, 5, 6)

        p1 += p2
        self.assertEqual(list(p1), [5, 7, 9])
        p1 -= p2
        self.assertEqual(list(p1), [1, 2, 3])

        p1 += 4
        self.assertEqual(list(p1), [5, 6, 7])
        p1 -= 2
        self.assertEqual(list(p1), [3, 4, 5])

        p1 *= 8
        self.assertEqual(list(p1), [24, 32, 40])
        p1 /= 4
        self.assertEqual(list(p1), [6, 8, 10])

    def test_mirror(self):
        Point = self.Point

        p1 = Point(1, 2, 3)
        p1.mirror(axis='x')
        self.assertEqual(list(p1), [1, -2, -3])

        p2 = Point(1, 2, 3)
        p2.mirror('y')
        self.assertEqual(list(p2), [-1, 2, -3])

        p3 = Point(1, 2, 3)
        p3.mirror('z')
        self.assertEqual(list(p3), [-1, -2, 3])

        p4 = Point(1, 2, 3)
        p4.mirror('X')
        self.assertEqual(list(p4), [1, -2, -3])

        with self.assertRaisesRegex(
                ValueError, "Point3d::mirror: axis must be 'x', 'y', or 'z'"):
            Point(1, 2, 3).mirror('w')


class Point3dFp32TC(Point3dTB, unittest.TestCase):

    def setUp(self):
        self.Point = solvcon.Point3dFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)

    def test_type(self):
        self.assertIs(solvcon.Point3dFp32, self.Point)

    def test_repr_str(self):
        p = solvcon.Point3dFp32(607.7, -64.2, 0)
        golden = "Point3dFp32(607.7, -64.2, 0)"
        # __repr__ is the same as __str__ for Point3d
        self.assertEqual(repr(p), golden)
        self.assertEqual(str(p), golden)
        # Evaluate the string and test the result
        e = eval(golden, vars(solvcon))
        self.assertEqual(p, e)


class Point3dFp64TC(Point3dTB, unittest.TestCase):

    def setUp(self):
        self.Point = solvcon.Point3dFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

    def test_type(self):
        self.assertIs(solvcon.Point3dFp64, self.Point)

    def test_repr_str(self):
        p = solvcon.Point3dFp64(607.7, -64.2, 0)
        golden = "Point3dFp64(607.7, -64.2, 0)"
        # __repr__ is the same as __str__ for Point3d
        self.assertEqual(repr(p), golden)
        self.assertEqual(str(p), golden)
        # Evaluate the string and test the result
        e = eval(golden, vars(solvcon))
        self.assertEqual(p, e)


class PointPadTB(testing.TestBase):

    def test_ndim(self):
        pp2d = self.PointPad(ndim=2)
        self.assertEqual(pp2d.ndim, 2)
        pp3d = self.PointPad(ndim=3)
        self.assertEqual(pp3d.ndim, 3)

        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 0 < 2"):
            self.PointPad(ndim=0)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 0 < 2"):
            self.PointPad(ndim=0, nelem=2)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 1 < 2"):
            self.PointPad(ndim=1)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 1 < 2"):
            self.PointPad(ndim=1, nelem=3)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 4 > 3"):
            self.PointPad(ndim=4)
        with self.assertRaisesRegex(
                ValueError, "PointPad::PointPad: ndim = 4 > 3"):
            self.PointPad(ndim=4, nelem=5)

    def test_construct_2d(self):
        xarr = self.SimpleArray(array=np.array([1, 2, 3], dtype=self.dtype))
        yarr = self.SimpleArray(array=np.array([4, 5, 6], dtype=self.dtype))
        pp = self.PointPad(x=xarr, y=yarr, clone=False)
        self.assertEqual(pp.ndim, 2)
        self.assert_allclose(pp.x, [1, 2, 3])
        self.assert_allclose(pp.y, [4, 5, 6])
        self.assertEqual(len(pp.z), 0)

        # Test zero-copy writing
        pp.x[1] = 200.2
        pp.y[0] = -700.3
        self.assert_allclose(list(pp[0]), (1, -700.3, 0))
        self.assert_allclose(list(pp[1]), (200.2, 5, 0))
        self.assert_allclose(list(pp[2]), (3, 6, 0))

        pp2 = self.PointPad(ndim=2, nelem=3)
        for i in range(len(pp)):
            pp2.set_at(i, pp.get_at(i).x, pp.get_at(i).y)
        self.assert_allclose(pp2.x, [1, 200.2, 3])
        self.assert_allclose(pp2.y, [-700.3, 5, 6])
        self.assertEqual(len(pp2.z), 0)

        packed = pp2.pack_array().ndarray
        self.assertEqual(packed.shape, (3, 2))
        self.assert_allclose(list(packed[0]), (1, -700.3))
        self.assert_allclose(list(packed[1]), (200.2, 5))
        self.assert_allclose(list(packed[2]), (3, 6))

    def test_construct_3d(self):
        xarr = self.SimpleArray(array=np.array([1, 2, 3], dtype=self.dtype))
        yarr = self.SimpleArray(array=np.array([4, 5, 6], dtype=self.dtype))
        zarr = self.SimpleArray(array=np.array([7, 8, 9], dtype=self.dtype))
        pp = self.PointPad(x=xarr, y=yarr, z=zarr, clone=False)
        self.assertEqual(pp.ndim, 3)
        self.assert_allclose(pp.x, [1, 2, 3])
        self.assert_allclose(pp.y, [4, 5, 6])
        self.assert_allclose(pp.z, [7, 8, 9])

        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            pp.x_at(3)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            pp.y_at(3)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            pp.z_at(3)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            pp.get_at(3)
        with self.assertRaisesRegex(
                IndexError,
                "SimpleCollector: index 3 is out of bounds with size 3"):
            pp.set_at(3, self.Point(0, 0, 0))

        # Test zero-copy writing
        pp.x[1] = 200.2
        pp.y[0] = -700.3
        pp.z[2] = 213.9
        self.assert_allclose(list(pp[0]), (1, -700.3, 7))
        self.assert_allclose(list(pp[1]), (200.2, 5, 8))
        self.assert_allclose(list(pp[2]), (3, 6, 213.9))

        pp2 = self.PointPad(ndim=3, nelem=3)
        for i in range(len(pp)):
            pp2.set_at(i, pp.get_at(i).x, pp.get_at(i).y, pp.get_at(i).z)
        self.assert_allclose(pp2.x, [1, 200.2, 3])
        self.assert_allclose(pp2.y, [-700.3, 5, 6])
        self.assert_allclose(pp2.z, [7, 8, 213.9])

        packed = pp2.pack_array().ndarray
        self.assertEqual(packed.shape, (3, 3))
        self.assert_allclose(list(packed[0]), (1, -700.3, 7))
        self.assert_allclose(list(packed[1]), (200.2, 5, 8))
        self.assert_allclose(list(packed[2]), (3, 6, 213.9))

    def test_append_2d(self):
        pp = self.PointPad(ndim=2)
        self.assertEqual(pp.ndim, 2)
        self.assertEqual(len(pp), 0)
        pp.append(1.1, 2.2)
        self.assertEqual(len(pp), 1)
        self.assert_allclose(pp.x_at(0), 1.1)
        self.assert_allclose(pp.y_at(0), 2.2)
        pp.append(1.1 * 3, 2.2 * 3)
        self.assertEqual(len(pp), 2)
        self.assert_allclose(pp.x_at(1), 1.1 * 3)
        self.assert_allclose(pp.y_at(1), 2.2 * 3)
        pp.append(1.1 * 3.1, 2.2 * 3.1)
        self.assertEqual(len(pp), 3)
        self.assert_allclose(pp.x_at(2), 1.1 * 3.1)
        self.assert_allclose(pp.y_at(2), 2.2 * 3.1)

        with self.assertRaisesRegex(
                IndexError, "PointPad::append: ndim must be 3 but is 2"):
            pp.append(3.2, 4.1, 5.7)
        self.assertEqual(len(pp), 3)

        # Test batch interface
        self.assert_allclose(pp.x, [1.1, 1.1 * 3, 1.1 * 3.1])
        self.assert_allclose(pp.y, [2.2, 2.2 * 3, 2.2 * 3.1])
        pp.x[0] = -10.9
        pp.x.ndarray[2] = -13.2
        self.assert_allclose(pp.x_at(0), -10.9)
        self.assert_allclose(pp.x_at(1), 1.1 * 3)
        self.assert_allclose(pp.x_at(2), -13.2)
        pp.y[1] = -0.93
        pp.y.ndarray[2] = 29.1
        self.assert_allclose(pp.y_at(0), 2.2)
        self.assert_allclose(pp.y_at(1), -0.93)
        self.assert_allclose(pp.y_at(2), 29.1)
        self.assertEqual(len(pp.z), 0)

    def test_append_3d(self):
        pp = self.PointPad(ndim=3)
        self.assertEqual(pp.ndim, 3)
        self.assertEqual(len(pp), 0)
        pp.append(1.1, 2.2, 3.3)
        self.assertEqual(len(pp), 1)
        self.assert_allclose(pp.x_at(0), 1.1)
        self.assert_allclose(pp.y_at(0), 2.2)
        self.assert_allclose(pp.z_at(0), 3.3)
        pp.append(1.1 * 5, 2.2 * 5, 3.3 * 5)
        self.assertEqual(len(pp), 2)
        self.assert_allclose(pp.x_at(1), 1.1 * 5)
        self.assert_allclose(pp.y_at(1), 2.2 * 5)
        self.assert_allclose(pp.z_at(1), 3.3 * 5)
        pp.append(1.1 * 5.1, 2.2 * 5.1, 3.3 * 5.1)
        self.assertEqual(len(pp), 3)
        self.assert_allclose(pp.x_at(2), 1.1 * 5.1)
        self.assert_allclose(pp.y_at(2), 2.2 * 5.1)
        self.assert_allclose(pp.z_at(2), 3.3 * 5.1)

        with self.assertRaisesRegex(
                IndexError, "PointPad::append: ndim must be 2 but is 3"):
            pp.append(3.2, 4.1)
        self.assertEqual(len(pp), 3)

        # Test batch interface
        self.assert_allclose(pp.x, [1.1, 1.1 * 5, 1.1 * 5.1])
        self.assert_allclose(pp.y, [2.2, 2.2 * 5, 2.2 * 5.1])
        self.assert_allclose(pp.z, [3.3, 3.3 * 5, 3.3 * 5.1])
        pp.x[0] = -10.9
        pp.x.ndarray[2] = -13.2
        self.assert_allclose(pp.x_at(0), -10.9)
        self.assert_allclose(pp.x_at(1), 1.1 * 5)
        self.assert_allclose(pp.x_at(2), -13.2)
        pp.y[1] = -0.93
        pp.y.ndarray[2] = 29.1
        self.assert_allclose(pp.y_at(0), 2.2)
        self.assert_allclose(pp.y_at(1), -0.93)
        self.assert_allclose(pp.y_at(2), 29.1)
        pp.z[0] = 2.31
        pp.z.ndarray[1] = 8.23
        self.assert_allclose(pp.z_at(0), 2.31)
        self.assert_allclose(pp.z_at(1), 8.23)
        self.assert_allclose(pp.z_at(2), 3.3 * 5.1)

    def test_getitem_index(self):
        pp = self.PointPad(ndim=2)
        for it in range(4):
            pp.append(float(it), float(it) * 10)

        self.assert_allclose(list(pp[0]), (0, 0, 0))
        self.assert_allclose(list(pp[-1]), (3, 30, 0))
        self.assert_allclose(list(pp[-4]), (0, 0, 0))

        with self.assertRaisesRegex(
                IndexError, "PointPad: index 4 is out of bounds with size 4"):
            pp[4]
        with self.assertRaisesRegex(
                IndexError, "PointPad: index -5 is out of bounds with size 4"):
            pp[-5]

        empty = self.PointPad(ndim=3)
        with self.assertRaisesRegex(
                IndexError, "PointPad: index 0 is out of bounds with size 0"):
            empty[0]
        with self.assertRaisesRegex(
                IndexError, "PointPad: index -1 is out of bounds with size 0"):
            empty[-1]

    def test_getitem_slice(self):
        pp = self.PointPad(ndim=3)
        for it in range(4):
            pp.append(float(it), float(it) * 10, float(it) * 100)

        full = pp[:]
        self.assertIsInstance(full, type(pp))
        self.assertEqual(full.ndim, 3)
        self.assertEqual(len(full), 4)
        self.assert_allclose(full.x, [0, 1, 2, 3])
        self.assert_allclose(full.y, [0, 10, 20, 30])
        self.assert_allclose(full.z, [0, 100, 200, 300])

        strided = pp[1::2]
        self.assertEqual(strided.ndim, 3)
        self.assert_allclose(strided.x, [1, 3])
        self.assert_allclose(strided.y, [10, 30])
        self.assert_allclose(strided.z, [100, 300])

        flipped = pp[::-1]
        self.assert_allclose(flipped.x, [3, 2, 1, 0])

        # A slice copies, so writing to it leaves the source alone.
        part = pp[1:3]
        part.set_at(0, -1.0, -2.0, -3.0)
        self.assert_allclose(list(part[0]), (-1, -2, -3))
        self.assert_allclose(list(pp[1]), (1, 10, 100))

        # A 2D pad keeps its dimensionality and its empty z array.
        pp2d = self.PointPad(ndim=2)
        pp2d.append(1.0, 2.0)
        sliced2d = pp2d[:]
        self.assertEqual(sliced2d.ndim, 2)
        self.assertEqual(len(sliced2d.z), 0)
        self.assert_allclose(list(sliced2d[0]), (1, 2, 0))

        empty = self.PointPad(ndim=3)
        self.assertEqual(len(empty[:]), 0)
        self.assertEqual(empty[:].ndim, 3)
        self.assertEqual(len(pp[3:1]), 0)

    def test_mirror_2d(self):
        PointPad = self.PointPad

        pp = PointPad(ndim=2)
        pp.append(1.0, 2.0)
        pp.append(3.0, 4.0)
        pp.append(5.0, 6.0)

        pp.mirror('x')
        self.assert_allclose(pp.x_at(0), 1.0)
        self.assert_allclose(pp.y_at(0), -2.0)
        self.assert_allclose(pp.x_at(1), 3.0)
        self.assert_allclose(pp.y_at(1), -4.0)

        pp.mirror('y')
        self.assert_allclose(pp.x_at(0), -1.0)
        self.assert_allclose(pp.y_at(0), -2.0)
        self.assert_allclose(pp.x_at(1), -3.0)
        self.assert_allclose(pp.y_at(1), -4.0)

    def test_mirror_3d(self):
        PointPad = self.PointPad

        pp = PointPad(ndim=3)
        pp.append(1.0, 2.0, 3.0)
        pp.append(4.0, 5.0, 6.0)

        pp.mirror('z')
        self.assert_allclose(pp.x_at(0), -1.0)
        self.assert_allclose(pp.y_at(0), -2.0)
        self.assert_allclose(pp.z_at(0), 3.0)
        self.assert_allclose(pp.x_at(1), -4.0)
        self.assert_allclose(pp.y_at(1), -5.0)
        self.assert_allclose(pp.z_at(1), 6.0)

        pp.mirror('X')
        self.assert_allclose(pp.x_at(0), -1.0)
        self.assert_allclose(pp.y_at(0), 2.0)
        self.assert_allclose(pp.z_at(0), -3.0)
        self.assert_allclose(pp.x_at(1), -4.0)
        self.assert_allclose(pp.y_at(1), 5.0)
        self.assert_allclose(pp.z_at(1), -6.0)

        with self.assertRaisesRegex(
                ValueError, "PointPad::mirror: axis must be 'x', 'y', or 'z'"):
            pp.mirror('w')

    def test_alignment(self):
        # Test alignment property for PointPad with various configurations
        pp2d = self.PointPad(ndim=2)
        self.assertEqual(pp2d.alignment, 0)

        # Test alignment with valid values (16, 32, 64)
        pp2d_aligned = self.PointPad(ndim=2, nelem=8, alignment=32)
        self.assertEqual(pp2d_aligned.alignment, 32)

        pp3d_aligned = self.PointPad(ndim=3, nelem=16, alignment=64)
        self.assertEqual(pp3d_aligned.alignment, 64)

        # Test invalid alignment values
        invalid_alignments = [
            1, 2, 4, 8, 128, 256, 1024, 15, 17, 31, 33, 63, 65
        ]
        for alignment in invalid_alignments:
            with self.assertRaisesRegex(
                    ValueError,
                    "PointPad::PointPad: alignment must be 0, 16, 32, or 64"):  # noqa E501
                self.PointPad(ndim=2, nelem=4, alignment=alignment)

        # Test alignment with arrays and cloning
        xarr = self.SimpleArray(
            array=np.array([1, 2, 3, 4], dtype=self.dtype))
        yarr = self.SimpleArray(
            array=np.array([5, 6, 7, 8], dtype=self.dtype))
        pp_arr_aligned = self.PointPad(
            x=xarr, y=yarr, clone=True, alignment=16)
        self.assertEqual(pp_arr_aligned.alignment, 16)
        self.assertEqual(pp_arr_aligned.ndim, 2)

        # Test invalid alignment values during array initialization
        with self.assertRaisesRegex(
                ValueError,
                "PointPad::PointPad: alignment must be 0, 16, 32, or 64"):  # noqa E501
            self.PointPad(x=xarr, y=yarr, clone=True, alignment=12)

        with self.assertRaisesRegex(
                ValueError,
                "PointPad::PointPad: alignment must be 0, 16, 32, or 64"):  # noqa E501
            self.PointPad(x=xarr, y=yarr, clone=False, alignment=48)

        # Test alignment for 3D arrays
        zarr = self.SimpleArray(
            array=np.array([9, 10, 11, 12, 13, 14, 15, 16],
                           dtype=self.dtype))
        xarr2 = self.SimpleArray(
            array=np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=self.dtype))
        yarr2 = self.SimpleArray(
            array=np.array([9, 10, 11, 12, 13, 14, 15, 16],
                           dtype=self.dtype))
        pp_arr_3d_aligned = self.PointPad(
            x=xarr2, y=yarr2, z=zarr, clone=True, alignment=32)
        self.assertEqual(pp_arr_3d_aligned.alignment, 32)
        self.assertEqual(pp_arr_3d_aligned.ndim, 3)

        # Test invalid alignment values for 3D arrays
        with self.assertRaisesRegex(
                ValueError,
                "PointPad::PointPad: alignment must be 0, 16, 32, or 64"):  # noqa E501
            self.PointPad(
                x=xarr2, y=yarr2, z=zarr, clone=True, alignment=100)

        with self.assertRaisesRegex(
                ValueError,
                "PointPad::PointPad: alignment must be 0, 16, 32, or 64"):  # noqa E501
            self.PointPad(
                x=xarr2, y=yarr2, z=zarr, clone=False, alignment=512)

    def test_alignment_preservation(self):
        # Test that alignment is preserved during operations
        # like append and expand
        element_size = 4 if self.dtype == 'float32' else 8

        # Test alignment preservation for 2D PointPad
        nelem_32 = 32 // element_size
        pp2d_32 = self.PointPad(ndim=2, nelem=nelem_32, alignment=32)
        self.assertEqual(pp2d_32.alignment, 32)

        pp2d_32.append(1.0, 2.0)
        pp2d_32.append(3.0, 4.0)
        self.assertEqual(pp2d_32.alignment, 32)

        pp2d_32.expand(64)
        self.assertEqual(pp2d_32.alignment, 32)

        packed = pp2d_32.pack_array()
        self.assertEqual(packed.nbytes % 32, 0)

        # Test alignment preservation for 3D PointPad
        nelem_64 = 64 // element_size
        pp3d_64 = self.PointPad(ndim=3, nelem=nelem_64, alignment=64)
        self.assertEqual(pp3d_64.alignment, 64)

        pp3d_64.append(1.0, 2.0, 3.0)
        pp3d_64.append(4.0, 5.0, 6.0)
        self.assertEqual(pp3d_64.alignment, 64)

        pp3d_64.expand(128)
        self.assertEqual(pp3d_64.alignment, 64)

        packed_3d = pp3d_64.pack_array()
        self.assertEqual(packed_3d.nbytes % 64, 0)

        # Test alignment preservation for exceeding initial size
        nelem_16 = 16 // element_size
        pp2d_16 = self.PointPad(ndim=2, nelem=nelem_16, alignment=16)
        for i in range(10):
            pp2d_16.append(float(i), float(i + 1))
        self.assertEqual(pp2d_16.alignment, 16)

        pp2d_16.expand(200)
        self.assertEqual(pp2d_16.alignment, 16)

        packed_2d_16 = pp2d_16.pack_array()
        self.assertEqual(packed_2d_16.nbytes % 16, 0)

        # Test unaligned PointPad
        pp2d_unaligned = self.PointPad(ndim=2, nelem=5)
        self.assertEqual(pp2d_unaligned.alignment, 0)

        pp2d_unaligned.append(1.0, 2.0)
        self.assertEqual(pp2d_unaligned.alignment, 0)

        pp2d_unaligned.expand(20)
        self.assertEqual(pp2d_unaligned.alignment, 0)

    def test_alignment_valid_values(self):
        # Test valid alignment values (0, 16, 32, 64) for PointPad
        element_size = 4 if self.dtype == 'float32' else 8

        for alignment in [0, 16, 32, 64]:
            if alignment == 0:
                nelem = 4
            else:
                nelem = alignment // element_size

            pp2d = self.PointPad(
                ndim=2, nelem=nelem, alignment=alignment)
            self.assertEqual(pp2d.alignment, alignment)

            nelem_3d = max(nelem, 8)
            pp3d = self.PointPad(
                ndim=3, nelem=nelem_3d, alignment=alignment)
            self.assertEqual(pp3d.alignment, alignment)

            if alignment == 0:
                arr_nelem = 3
            else:
                arr_nelem = max(alignment // element_size, 2)

            xarr = self.SimpleArray(
                array=np.arange(arr_nelem, dtype=self.dtype))
            yarr = self.SimpleArray(
                array=np.arange(arr_nelem, arr_nelem * 2,
                                dtype=self.dtype))
            pp_arr = self.PointPad(
                x=xarr, y=yarr, clone=True, alignment=alignment)
            self.assertEqual(pp_arr.alignment, alignment)

            zarr = self.SimpleArray(
                array=np.arange(arr_nelem * 2, arr_nelem * 3,
                                dtype=self.dtype))
            pp_arr_3d = self.PointPad(
                x=xarr, y=yarr, z=zarr, clone=True, alignment=alignment)
            self.assertEqual(pp_arr_3d.alignment, alignment)

    def test_alignment_size_validation(self):
        # Test that alignment size validation works correctly
        element_size = 4 if self.dtype == 'float32' else 8
        test_cases = [
            (4, [
                (2, 5, 16,
                 f"BufferExpander::allocate: size {4 * 5} must be a multiple of alignment 16"),  # noqa E501
                (2, 7, 32,
                 f"BufferExpander::allocate: size {4 * 7} must be a multiple of alignment 32"),  # noqa E501
                (3, 9, 64,
                 f"BufferExpander::allocate: size {4 * 9} must be a multiple of alignment 64"),  # noqa E501
                (2, 11, 16,
                 f"BufferExpander::allocate: size {4 * 11} must be a multiple of alignment 16"),  # noqa E501
                (3, 13, 32,
                 f"BufferExpander::allocate: size {4 * 13} must be a multiple of alignment 32"),  # noqa E501
            ]),
            (8, [
                (2, 3, 32,
                 f"BufferExpander::allocate: size {8 * 3} must be a multiple of alignment 32"),  # noqa E501
                (2, 5, 64,
                 f"BufferExpander::allocate: size {8 * 5} must be a multiple of alignment 64"),  # noqa E501
                (3, 7, 64,
                 f"BufferExpander::allocate: size {8 * 7} must be a multiple of alignment 64"),  # noqa E501
                (2, 9, 16,
                 f"BufferExpander::allocate: size {8 * 9} must be a multiple of alignment 16"),  # noqa E501
                (3, 11, 32,
                 f"BufferExpander::allocate: size {8 * 11} must be a multiple of alignment 32"),  # noqa E501
            ]),
        ]

        for size, cases in test_cases:
            if element_size == size:
                for ndim, nelem, alignment, error_msg in cases:
                    with self.assertRaisesRegex(ValueError, error_msg):
                        self.PointPad(
                            ndim=ndim, nelem=nelem, alignment=alignment)


class PointPadFp32TC(PointPadTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float32'
        self.SimpleArray = solvcon.SimpleArrayFloat32
        self.Point = solvcon.Point3dFp32
        self.PointPad = solvcon.PointPadFp32

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-7
        return super().assert_allclose(*args, **kw)


class PointPadFp64TC(PointPadTB, unittest.TestCase):

    def setUp(self):
        self.dtype = 'float64'
        self.SimpleArray = solvcon.SimpleArrayFloat64
        self.Point = solvcon.Point3dFp64
        self.PointPad = solvcon.PointPadFp64

    def assert_allclose(self, *args, **kw):
        if 'rtol' not in kw:
            kw['rtol'] = 1.e-15
        return super().assert_allclose(*args, **kw)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
