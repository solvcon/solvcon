# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The point probe of the reflection: the mean value coordinates it
interpolates the nodes by, and the three readings of a cell against the
analytic answer."""

import unittest

import numpy as np
from numpy.testing import assert_almost_equal

from solvcon.apps.obsrefl import ObliqueShock, Probe, Reflection
from solvcon.apps.obsrefl import _probe


def linear(points):
    """The linear field every interpolation check reproduces.

    :param points: The points, ``[..., 2]``.
    :return: The field at the points, ``[...]``.
    """
    points = np.asarray(points, dtype='float64')
    return 1.5 + 0.7 * points[..., 0] - 0.3 * points[..., 1]


class MeanValueWeightsTC(unittest.TestCase):
    """The weights the probe interpolates the node values of a cell by."""

    TRIANGLE = ((0.0, 0.0), (2.0, 0.0), (0.5, 1.5))
    PENTAGON = ((0.0, 0.0), (2.0, -0.2), (2.6, 1.1), (1.2, 2.0),
                (-0.3, 1.0))

    def test_a_triangle_takes_the_barycentric_coordinates(self):
        point = (0.8, 0.5)
        weights = _probe.mean_value_weights(self.TRIANGLE, point)
        # Solve x = sum w_i v_i with sum w_i = 1 for the barycentric ones.
        corners = np.array(self.TRIANGLE, dtype='float64')
        system = np.vstack([corners.T, np.ones(3, dtype='float64')])
        assert_almost_equal(weights,
                            np.linalg.solve(system, (0.8, 0.5, 1.0)))

    def test_a_linear_field_comes_back_on_convex_polygons(self):
        # A quad of the structured mesh and a pentagon of no cell type: the
        # rule needs no code per type, which is why it was taken.
        square = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
        for corners, point in ((square, (0.25, 0.6)),
                               (self.PENTAGON, (1.1, 0.9))):
            weights = _probe.mean_value_weights(corners, point)
            assert_almost_equal(weights.sum(), 1.0)
            self.assertTrue((weights > 0.0).all())
            assert_almost_equal(weights @ linear(corners), linear(point))

    def test_a_corner_takes_its_own_value(self):
        weights = _probe.mean_value_weights(self.PENTAGON, self.PENTAGON[2])
        assert_almost_equal(weights, (0.0, 0.0, 1.0, 0.0, 0.0))

    def test_a_side_is_cut_between_its_two_ends(self):
        # The general formula divides by zero on a side, so the rule falls
        # back to the line between the two corners.
        a = np.array(self.PENTAGON[1], dtype='float64')
        b = np.array(self.PENTAGON[2], dtype='float64')
        weights = _probe.mean_value_weights(self.PENTAGON, a + 0.25 * (b - a))
        assert_almost_equal(weights, (0.0, 0.75, 0.25, 0.0, 0.0))

    def test_a_point_on_the_line_of_a_side_keeps_finite_weights(self):
        # A pick in float32 can land just past a cell, on the line of one
        # of its sides, where that side spans no angle and the general
        # formula reads 0 / 0.
        a = np.array(self.PENTAGON[1], dtype='float64')
        b = np.array(self.PENTAGON[2], dtype='float64')
        weights = _probe.mean_value_weights(self.PENTAGON, b + 0.1 * (b - a))
        self.assertTrue(np.isfinite(weights).all())
        assert_almost_equal(weights.sum(), 1.0)


class _ProbeTB:
    """A reflection run seeded with a known field, then probed."""

    CELL_TYPE = 'quad'

    def setUp(self):
        self.shock = ObliqueShock()
        self.shock.build_constant()
        self.shock.build_numerical(cell_type=self.CELL_TYPE, nx=32, ny=8)
        self.analysis = Reflection(self.shock)
        self.probe = Probe(self.analysis)
        self.field = self.analysis.field

    def interior_cell(self):
        """A cell whose nodes all sit off the boundary of the domain."""
        mesher = self.shock.mesher
        for cell in range(len(self.field.centroid)):
            _, corners = self.analysis.cell_corners(cell)
            if ((corners[:, 0] > mesher.x0 + 1.e-9).all()
                    and (corners[:, 0] < mesher.x1 - 1.e-9).all()
                    and (corners[:, 1] > mesher.y0 + 1.e-9).all()
                    and (corners[:, 1] < mesher.y1 - 1.e-9).all()):
                return cell
        raise AssertionError("no interior cell")


class _ReadingTB(_ProbeTB):
    """The three readings of a cell under a linear density."""

    def setUp(self):
        super().setUp()
        cons = self.field.conserved
        cons[:, 0] = linear(self.field.solution_point)
        self.field.gradient[:] = 0.0
        self.field.gradient[:, 0, :] = (0.7, -0.3)
        self.cell = self.interior_cell()
        _, corners = self.analysis.cell_corners(self.cell)
        self.point = 0.6 * corners[0] + 0.4 * corners[2]

    def test_each_reading_comes_from_its_own_source(self):
        reading = self.probe.read(self.cell, self.point, 'density')
        self.assertEqual((self.cell, 'density'),
                         (reading.cell, reading.name))
        assert_almost_equal(reading.point, self.point)
        sp = self.field.solution_point[self.cell]
        assert_almost_equal(reading.solution_point, sp)
        # The cell value holds at its solution point only, while the
        # element carries the slope to the point read.
        assert_almost_equal(reading.at_solution_point, linear(sp))
        assert_almost_equal(reading.by_element, linear(self.point))
        self.assertGreater(abs(reading.by_element
                               - reading.at_solution_point), 1.e-3)

    def test_the_mesh_reading_interpolates_the_node_values(self):
        # Node values on a plane come back at any point between them.
        planar = linear(self.analysis._node_coord())
        self.analysis.node_field = lambda name: planar
        nodes, corners = self.analysis.cell_corners(self.cell)
        reading = self.probe.read(self.cell, self.point, 'density')
        assert_almost_equal(reading.by_mesh, linear(self.point))
        # At a node the reading is the node value itself.
        at_node = self.probe.read(self.cell, corners[1], 'density')
        assert_almost_equal(at_node.by_mesh, planar[nodes[1]])

    def test_only_a_body_cell_is_read(self):
        # The rows of the ghost cells come before the body cells.
        for cell in (-1, self.shock.mesh.ncell):
            with self.subTest(cell=cell):
                with self.assertRaises(IndexError):
                    self.analysis.cell_corners(cell)


class ProbeQuadTC(_ReadingTB, unittest.TestCase):
    """The readings on the quads, where a horizontal cut runs through the
    vertical sides of a cell."""

    def test_the_mesh_reading_agrees_with_the_profile_on_the_cut(self):
        # Where the cut crosses a side of the cell, the probe and the
        # profile both cut that side between its two node values.
        _, corners = self.analysis.cell_corners(self.cell)
        height = float(corners[:, 1].min() + corners[:, 1].max()) / 2.0
        left = float(corners[:, 0].min())
        profile = self.analysis.profile(height, 'density')
        station = int(np.argmin(np.abs(profile.x - left)))
        reading = self.probe.read(
            self.cell, (profile.x[station], height), 'density')
        assert_almost_equal(reading.by_mesh, profile.computed[station])


class ProbeTriangleTC(_ReadingTB, unittest.TestCase):
    """The same readings on the triangles, where the solution point and the
    centroid part."""

    CELL_TYPE = 'triangle'

    def setUp(self):
        super().setUp()
        _, corners = self.analysis.cell_corners(self.cell)
        self.point = corners.mean(axis=0) + 0.1 * (corners[0] - corners[1])


class ProbeExactTC(_ProbeTB, unittest.TestCase):
    """The exact reading against the analytic zones."""

    def test_the_exact_reading_is_the_zone_value(self):
        zones = self.analysis.zone_field('pressure')
        masks = self.analysis.zone_masks()
        for zone in range(3):
            cell = int(np.nonzero(masks[zone])[0][0])
            reading = self.probe.read(
                cell, self.field.centroid[cell], 'pressure')
            assert_almost_equal(reading.exact, zones[zone])

    def test_a_field_without_an_exact_value_reads_none(self):
        cell = self.interior_cell()
        reading = self.probe.read(cell, self.field.centroid[cell], 'cfl')
        self.assertIsNone(reading.exact)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
