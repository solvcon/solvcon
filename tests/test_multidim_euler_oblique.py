# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The oblique-shock reflection test.

A uniform supersonic stream enters from the left over a slip wall whose bottom
turns into a wedge inclined by a fixed angle.  The wedge deflects the flow, an
oblique shock forms at the wedge tip and reflects off the flat top slip wall,
and the flow leaves through the non-reflective outflow on the right.  The mesh
comes in three element flavors: one quadrilateral per grid box, each box cut
into two triangles, or a Delaunay triangulation of jittered interior points.
"""

import math
import unittest

from numpy.testing import assert_almost_equal

import solvcon
from solvcon.multidim.euler.oblique import (ObliqueShock, ObliqueShockMesher,
                                            ObliqueShockRelation)


class _ObliqueMeshBase:
    """The oblique-shock mesh builds and its boundary classifies cleanly.

    The boundary checks are identical for all three element flavors because
    every flavor shares the same boundary node layout (interior-only changes
    in cell shape never touch the boundary faces); subclasses set the flavor,
    and the box-based flavors also set the cell count per grid box.
    """

    NX = 24
    NY = 8
    LL = (0.0, 0.0)
    UR = (3.0, 1.0)
    CELL_TYPE = None
    CELLS_PER_BOX = None
    CLTPN = None

    @classmethod
    def setUpClass(cls):
        cls.mesher = ObliqueShockMesher(nx=cls.NX, ny=cls.NY, ll=cls.LL,
                                        ur=cls.UR, x_ramp=0.5,
                                        wedge_angle=15.0)
        cls.mesh = cls.mesher.make_mesh(cell_type=cls.CELL_TYPE)

    def _boundary_faces(self):
        return {ifc for ifc in range(self.mesh.nface)
                if self.mesh.fccls[ifc, 1] < 0}

    def _signed_area2(self, icl):
        # Twice the signed shoelace area of cell icl, straight from
        # ndcrd/clnds via element access (the .ndarray views carry
        # prepended ghost rows).
        mh = self.mesh
        nnd = mh.clnds[icl, 0]
        xy = [(mh.ndcrd[mh.clnds[icl, 1 + it], 0],
               mh.ndcrd[mh.clnds[icl, 1 + it], 1])
              for it in range(nnd)]
        return sum(xy[it][0] * xy[(it + 1) % nnd][1]
                   - xy[(it + 1) % nnd][0] * xy[it][1]
                   for it in range(nnd))

    def test_mesh_shape(self):
        mh = self.mesh
        nbox = self.NX * self.NY
        self.assertEqual(2, mh.ndim)
        self.assertEqual(self.CELLS_PER_BOX * nbox, mh.ncell)
        # Element access reads the body cells; the .ndarray views also
        # carry the ghost cells that build_ghost prepends.
        self.assertTrue(all(mh.cltpn[icl] == self.CLTPN
                            for icl in range(mh.ncell)))
        # A structured quad grid has 2*nx*ny + nx + ny faces; each box split
        # adds one cell and one (diagonal, interior) face.
        self.assertEqual((self.CELLS_PER_BOX + 1) * nbox + self.NX + self.NY,
                         mh.nface)
        # The boundary is the four edges of the logical grid either way.
        self.assertEqual(2 * (self.NX + self.NY), len(self._boundary_faces()))

    def test_classification_partitions_boundary(self):
        inlet, walls, outflow = self.mesher.classify_boundary(self.mesh)
        # Every role is present.
        self.assertTrue(inlet)
        self.assertTrue(walls)
        self.assertTrue(outflow)
        si, sw, so = set(inlet), set(walls), set(outflow)
        # The three roles are pairwise disjoint.
        self.assertEqual(set(), si & sw)
        self.assertEqual(set(), si & so)
        self.assertEqual(set(), sw & so)
        # Together they cover every boundary face and nothing else.
        self.assertEqual(self._boundary_faces(), si | sw | so)

    def test_classification_counts(self):
        inlet, walls, outflow = self.mesher.classify_boundary(self.mesh)
        # The left and right edges carry one face per cell row; the top and
        # bottom edges carry one per cell column.
        self.assertEqual(self.NY, len(inlet))
        self.assertEqual(self.NY, len(outflow))
        self.assertEqual(2 * self.NX, len(walls))

    def test_inlet_outflow_geometry(self):
        mh = self.mesh
        inlet, walls, outflow = self.mesher.classify_boundary(mh)
        # The real domain spans [LL[0], UR[0]]; ndcrd would include ghost-node
        # coordinates that overshoot the edges, so use the construction
        # bounds.
        xmin, xmax = self.LL[0], self.UR[0]
        # Inlet faces sit on x == xmin with the outward normal pointing in -x.
        for ifc in inlet:
            assert_almost_equal(mh.fccnd[ifc, 0], xmin, decimal=12)
            assert_almost_equal(
                [mh.fcnml[ifc, 0], mh.fcnml[ifc, 1]], [-1.0, 0.0], decimal=12)
        # Outflow faces sit on x == xmax with the normal pointing in +x.
        for ifc in outflow:
            assert_almost_equal(mh.fccnd[ifc, 0], xmax, decimal=12)
            assert_almost_equal(
                [mh.fcnml[ifc, 0], mh.fcnml[ifc, 1]], [1.0, 0.0], decimal=12)
        # Wall faces are the top and bottom edges: their centres are strictly
        # interior in x and their outward normals are vertical-dominant.
        for ifc in walls:
            xc = mh.fccnd[ifc, 0]
            self.assertGreater(xc, xmin)
            self.assertLess(xc, xmax)
            self.assertGreater(abs(mh.fcnml[ifc, 1]), abs(mh.fcnml[ifc, 0]))

    def test_wedge_is_present(self):
        # The defining feature versus a plain channel: some slip-wall faces
        # are inclined, so their outward normal has a non-zero x component.
        mh = self.mesh
        _, walls, _ = self.mesher.classify_boundary(mh)
        inclined = [ifc for ifc in walls if abs(mh.fcnml[ifc, 0]) > 1e-6]
        self.assertTrue(inclined)

    def test_cell_winding_and_volume(self):
        # clvol alone cannot pin the winding: build_interior repairs
        # inverted faces and accumulates absolute values, so it stays
        # positive even for clockwise connectivity.  Check the signed
        # shoelace area instead, and keep the clvol check for degeneracy
        # (zero or NaN volume).
        mh = self.mesh
        for icl in range(mh.ncell):
            self.assertGreater(self._signed_area2(icl), 0.0)
            self.assertGreater(mh.clvol[icl], 0.0)

    def test_cells_tile_the_domain(self):
        # The cells tile the domain exactly (no overlap, no gap): the
        # signed cell areas must sum to the area enclosed by the boundary
        # faces, computed independently through the divergence theorem
        # (area = 1/2 * sum of fcara * (fccnd . fcnml) over the boundary).
        mh = self.mesh
        total = sum(self._signed_area2(icl) for icl in range(mh.ncell)) / 2.0
        enclosed = sum(mh.fcara[ifc] * (mh.fccnd[ifc, 0] * mh.fcnml[ifc, 0]
                                        + mh.fccnd[ifc, 1] * mh.fcnml[ifc, 1])
                       for ifc in self._boundary_faces()) / 2.0
        assert_almost_equal(total, enclosed, decimal=10)


class _SingleBoundaryFaceTB:
    """No cell touches the domain boundary with more than one face.

    Mixed into the triangular flavors only: a corner ``'quad'`` cell always
    owns its two adjacent boundary edges, so quads are exempt.
    """

    def test_at_most_one_boundary_face_per_cell(self):
        # Each cell keeps at least two interior neighbours for the CESE
        # solver.  fccls(ifc, 0) is the interior cell of boundary face ifc,
        # so tallying it counts the boundary faces each cell owns.
        mh = self.mesh
        per_cell = {}
        for ifc in self._boundary_faces():
            icl = mh.fccls[ifc, 0]
            per_cell[icl] = per_cell.get(icl, 0) + 1
        self.assertTrue(per_cell)
        self.assertLessEqual(max(per_cell.values()), 1)


class ObliqueShockQuadMeshTC(_ObliqueMeshBase, unittest.TestCase):
    """One quadrilateral per grid box."""

    CELL_TYPE = 'quad'
    CELLS_PER_BOX = 1
    CLTPN = solvcon.StaticMesh.QUADRILATERAL


class ObliqueShockTriangleMeshTC(_ObliqueMeshBase, _SingleBoundaryFaceTB,
                                 unittest.TestCase):
    """Each grid box cut along its diagonal into two triangles."""

    CELL_TYPE = 'triangle'
    CELLS_PER_BOX = 2
    CLTPN = solvcon.StaticMesh.TRIANGLE


class ObliqueShockUnstructuredMeshTC(_ObliqueMeshBase, _SingleBoundaryFaceTB,
                                     unittest.TestCase):
    """Delaunay triangulation of the wedge: the structured boundary layout
    with jittered interior points.  The boundary tests of the base apply
    unchanged because all flavors share the boundary node layout.
    """

    CELL_TYPE = 'unstructured'
    CELLS_PER_BOX = None  # not box-based; test_mesh_shape is overridden
    CLTPN = solvcon.StaticMesh.TRIANGLE

    def _body_connectivity(self, mh):
        # The body cells as (type, node-id tuple) pairs, plus the node
        # coordinates, via element access -- the canonical identity of a
        # mesh, free of the ghost rows and uninitialised trailing clnds
        # columns that the raw .ndarray views carry.
        cells = [(mh.cltpn[icl],
                  tuple(mh.clnds[icl, 1 + it]
                        for it in range(mh.clnds[icl, 0])))
                 for icl in range(mh.ncell)]
        nodes = [(mh.ndcrd[ind, 0], mh.ndcrd[ind, 1])
                 for ind in range(mh.nnode)]
        return cells, nodes

    def test_mesh_shape(self):
        # The counts are not box-based, but any triangulation using every
        # vertex of a convex region obeys exact combinatorics: with nb
        # boundary and ni interior vertices it has 2*ni + nb - 2 cells,
        # and the Euler characteristic of a disk (V - E + F = 1) pins the
        # face count.
        mh = self.mesh
        nbnd = 2 * (self.NX + self.NY)
        self.assertEqual(2, mh.ndim)
        self.assertTrue(all(mh.cltpn[icl] == self.CLTPN
                            for icl in range(mh.ncell)))
        self.assertEqual(nbnd, len(self._boundary_faces()))
        self.assertEqual(2 * (mh.nnode - nbnd) + nbnd - 2, mh.ncell)
        self.assertEqual(1, mh.nnode - mh.nface + mh.ncell)

    def test_triangulation_is_deterministic(self):
        # The flavor advertises a reproducible mesh (RNG-free jitter): a
        # second build of the same parameters must reproduce the node
        # coordinates and the triangle connectivity exactly.  Guards against
        # a future switch to an unseeded RNG, which the count/Euler/tiling
        # invariants would not catch.
        again = self.mesher.make_mesh(cell_type=self.CELL_TYPE)
        self.assertEqual(self._body_connectivity(self.mesh),
                         self._body_connectivity(again))


class ObliqueShockMesherTC(unittest.TestCase):

    def test_unknown_cell_type(self):
        with self.assertRaises(ValueError):
            ObliqueShockMesher(nx=2, ny=2).make_mesh(cell_type='hexagon')

    def test_wedge_top_must_stay_below_ly(self):
        # Past atan(height / (ur[0] - x_ramp)) the grid columns invert and
        # the mesh would silently self-overlap; the constructor refuses
        # instead.  Here the limit is atan(1.0 / 2.5) ~ 21.8 degrees, so 30
        # degrees overflows.
        with self.assertRaises(ValueError):
            ObliqueShockMesher(nx=2, ny=2, ll=(0.0, 0.0), ur=(3.0, 1.0),
                               x_ramp=0.5, wedge_angle=30.0)


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
    MESHER_KW = dict(nx=24, ny=8, x_ramp=1.0)

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


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
