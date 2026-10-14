# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Tests for the Bowyer-Watson ``DelaunayBuilder``."""

import unittest

from solvcon import core


class DelaunayBuilderTC(unittest.TestCase):

    def make_pad(self, points, super_triangle):
        pad = core.PointPadFp64(ndim=2)
        for x, y in list(points) + list(super_triangle):
            pad.append(x, y)
        return pad

    def make_builder(self, pad, npt):
        builder = core.DelaunayBuilder()
        rr, ux, uy, uz = pad.circumcircle(npt, npt + 1, npt + 2)
        builder.add(ia=npt, ib=npt + 1, ic=npt + 2,
                    rr=rr, ux=ux, uy=uy, uz=uz)
        return builder

    def test_relative_tolerance(self):
        builder = core.DelaunayBuilder()
        self.assertEqual(builder.relative_tolerance, 1.e-12)
        builder.relative_tolerance = 1.e-9
        self.assertEqual(builder.relative_tolerance, 1.e-9)

    def test_add_and_len(self):
        builder = core.DelaunayBuilder()
        self.assertEqual(len(builder), 0)
        builder.add(ia=0, ib=1, ic=2, rr=1.0, ux=0.0, uy=0.0)
        self.assertEqual(len(builder), 1)

    def test_populate_square(self):
        # Four corners of a unit square give two triangles.
        corners = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        pad = self.make_pad(corners, [(-64.0, -32.0), (65.0, -32.0),
                                      (0.5, 64.0)])
        builder = self.make_builder(pad, 4)
        builder.populate_from_points(pad=pad, first=0, last=4)
        tris = builder.compute_triangle_nodes(pad=pad, nnode=4)
        self.assertEqual(tris.shape, (2, 3))
        nodes = tris.ndarray
        self.assertEqual(sorted(set(nodes.ravel().tolist())), [0, 1, 2, 3])
        # The builder still holds triangles on the super triangle (indices 4
        # to 6), but none of them is returned.
        self.assertGreater(len(builder), 2)
        self.assertLess(nodes.max(), 4)
        # Every triangle is counter-clockwise and the areas fill the square.
        total = 0.0
        for ia, ib, ic in nodes.tolist():
            ax, ay = corners[ia]
            bx, by = corners[ib]
            cx, cy = corners[ic]
            area = ((bx - ax) * (cy - ay) - (cx - ax) * (by - ay)) / 2.0
            self.assertGreater(area, 0.0)
            total += area
        self.assertAlmostEqual(total, 1.0)

    @staticmethod
    def square_pad():
        pad = core.PointPadFp64(ndim=2)
        for x, y in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
            pad.append(x=x, y=y)
        return pad

    def test_triangulate_is_repeatable(self):
        builder = core.DelaunayBuilder(pad=self.square_pad())
        first = builder.triangulate().ndarray.copy()
        second = builder.triangulate().ndarray
        self.assertEqual(first.tolist(), second.tolist())


if __name__ == '__main__':
    unittest.main()

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
