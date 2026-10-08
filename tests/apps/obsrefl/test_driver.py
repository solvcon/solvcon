# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The oblique-shock relation and the reflection driver.

The computing domain is a plain rectangle.  A uniform supersonic stream
enters horizontally from the left, the top boundary holds the state behind
an incident oblique shock, the bottom slip wall reflects the incident
shock, and the flow leaves through the non-reflective outflow on the right.
"""

import math
import unittest

from solvcon.pilot.apps.obsrefl import ObliqueShock, ObliqueShockRelation


class ObliqueShockRelationTC(unittest.TestCase):
    """The oblique-shock relation reproduces the reference values carried
    over from the doctests of the legacy solvcon gas parcel (Anderson,
    Modern Compressible Flow, chapter 4).
    """

    def setUp(self):
        self.ob = ObliqueShockRelation(gamma=1.4)

    def test_ratios(self):
        beta = math.radians(37.8)
        self.assertAlmostEqual(
            2.4204302545, self.ob.calc_density_ratio(3, beta), places=10)
        self.assertAlmostEqual(
            3.7777114257, self.ob.calc_pressure_ratio(3, beta), places=10)
        self.assertAlmostEqual(
            1.5607602899, self.ob.calc_temperature_ratio(3, beta), places=10)

    def test_gamma_changes_solution(self):
        self.ob.gamma = 1.2
        self.assertAlmostEqual(
            2.7793244902,
            self.ob.calc_density_ratio(3, math.radians(37.8)), places=10)

    def test_downstream_mach(self):
        self.assertAlmostEqual(
            0.4751909633, self.ob.calc_normal_dmach(3), places=10)
        self.assertAlmostEqual(
            1.9924827009,
            self.ob.calc_dmach(3, beta=math.radians(37.8)), places=10)
        self.assertAlmostEqual(
            1.9941316656,
            self.ob.calc_dmach(3, theta=math.radians(20)), places=10)

    def test_dmach_needs_one_angle(self):
        with self.assertRaises(ValueError):
            self.ob.calc_dmach(3)
        with self.assertRaises(ValueError):
            self.ob.calc_dmach(3, beta=0.2, theta=0.1)

    def test_detached_shock_is_rejected(self):
        # 40 degrees of deflection exceeds the maximum attached-shock
        # deflection at Mach 2 (about 23 degrees).
        with self.assertRaises(ValueError):
            self.ob.calc_shock_angle(2, math.radians(40))

    def test_angles_invert_each_other(self):
        # Example 4.6 of Anderson: M1 = 4 and theta = 32 degrees give a
        # weak-shock angle of about 48.2585 degrees, and the flow-angle
        # calculation inverts it.
        theta = math.radians(32)
        beta = self.ob.calc_shock_angle(4, theta, delta=1)
        self.assertAlmostEqual(48.2584798722, math.degrees(beta), places=10)
        self.assertAlmostEqual(
            32.0, math.degrees(self.ob.calc_flow_angle(4, beta)), places=6)


class _ObliqueShockDriverBase:
    """Base class to test for solver over each mesh flavor and marches a few
    steps; subclasses select the flavor.
    """

    CELL_TYPE = None
    # A coarse mesh keeps the driver tests fast.
    MESHER_KW = dict(nx=24, ny=8)

    def test_build_and_march(self):
        shock = ObliqueShock()
        shock.build_constant()
        shock.build_numerical(cell_type=self.CELL_TYPE, **self.MESHER_KW)
        # The core is built over the mesh with the right shape.
        self.assertEqual(shock.mesh.ncell, shock.svr.ncell)
        self.assertEqual(2, shock.svr.ndim)
        # The solution is not yet validated. Only make sure the solver runs
        # through.
        shock.march(10)


class ObliqueShockDriverQuadTC(_ObliqueShockDriverBase, unittest.TestCase):
    """The driver over the structured quadrilateral mesh."""

    CELL_TYPE = 'quad'


class ObliqueShockDriverTriangleTC(_ObliqueShockDriverBase,
                                   unittest.TestCase):
    """The driver over the structured triangular mesh."""

    CELL_TYPE = 'triangle'


class ObliqueShockDriverUnstructuredTC(_ObliqueShockDriverBase,
                                       unittest.TestCase):
    """The driver over the unstructured (Delaunay) triangular mesh."""

    CELL_TYPE = 'unstructured'


class ObliqueShockDriverTC(unittest.TestCase):

    def test_numerical_requires_constants(self):
        with self.assertRaises(ValueError):
            ObliqueShock().build_numerical()

    def test_constants_set_post_shock_state(self):
        shock = ObliqueShock()
        shock.build_constant(gamma=1.4, density=1.0, pressure=1.0, mach=3.0,
                             angle=10.0)
        relation = shock.relation
        beta = relation.calc_shock_angle(3.0, math.radians(10.0))
        self.assertAlmostEqual(beta, shock.shock_angle)
        # The imposed zone-2 state carries the analytical jumps, and its
        # velocity points 10 degrees below horizontal.
        self.assertAlmostEqual(relation.calc_density_ratio(3.0, beta),
                               shock.density2 / shock.density)
        self.assertAlmostEqual(relation.calc_pressure_ratio(3.0, beta),
                               shock.pressure2 / shock.pressure)
        vx, vy = shock.velocity2
        self.assertLess(vy, 0.0)
        self.assertAlmostEqual(math.radians(10.0), math.atan2(-vy, vx))
        speed2 = math.hypot(vx, vy)
        sos2 = math.sqrt(1.4 * shock.pressure2 / shock.density2)
        self.assertAlmostEqual(shock.mach2, speed2 / sos2)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
