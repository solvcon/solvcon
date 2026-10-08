# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Tests for the rectangular-domain mesher in ``solvcon.mesh``.

The mesh comes in three element flavors: one quadrilateral per grid box,
each box cut into two triangles, or a Delaunay triangulation of jittered
interior points.
"""

import unittest

from numpy.testing import assert_almost_equal

import solvcon
from solvcon import mesh as meshmod


class _RectMeshBase:
    """The rectangular-domain mesh builds and its boundary classifies cleanly.

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
        cls.mesher = meshmod.RectDomainMesher(
            nx=cls.NX, ny=cls.NY, ll=cls.LL, ur=cls.UR)
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
        edges = self.mesher.classify_boundary(self.mesh)
        # Every edge is present.
        for faces in edges:
            self.assertTrue(faces)
        sets = [set(faces) for faces in edges]
        # The four edges are pairwise disjoint.
        for it in range(len(sets)):
            for jt in range(it + 1, len(sets)):
                self.assertEqual(set(), sets[it] & sets[jt])
        # Together they cover every boundary face and nothing else.
        self.assertEqual(self._boundary_faces(),
                         sets[0] | sets[1] | sets[2] | sets[3])

    def test_classification_counts(self):
        left, top, bottom, right = self.mesher.classify_boundary(self.mesh)
        # The left and right edges carry one face per cell row; the top and
        # bottom edges carry one per cell column.
        self.assertEqual(self.NY, len(left))
        self.assertEqual(self.NY, len(right))
        self.assertEqual(self.NX, len(top))
        self.assertEqual(self.NX, len(bottom))

    def test_named_boundary_groups(self):
        mh = self.mesh
        names = meshmod.RectDomainMesher.BOUNDARY_NAMES
        # One group per domain edge and no catch-all leftover.
        self.assertEqual(list(names), [bnd.name for bnd in mh.bcs])
        edges = self.mesher.classify_boundary(mh)
        for name, faces in zip(names, edges):
            self.assertEqual(faces, mh.bc(name).facn.ndarray[:, 0].tolist())
        # Every boundary face carries its group index in bndfcs.
        byface = {ifc: ibc for ibc, faces in enumerate(edges)
                  for ifc in faces}
        for ifc, ibc in mh.bndfcs.ndarray[:, :2].tolist():
            self.assertEqual(byface[ifc], ibc)

    def test_boundary_geometry(self):
        mh = self.mesh
        left, top, bottom, right = self.mesher.classify_boundary(mh)
        # The real domain spans LL to UR; ndcrd would include ghost-node
        # coordinates that overshoot the edges, so use the construction
        # bounds.
        (xmin, ymin), (xmax, ymax) = self.LL, self.UR
        # Each edge sits on its domain boundary with the outward normal
        # exactly axis-aligned; the wedge-free domain has no inclined face.
        groups = [(left, 0, xmin, [-1.0, 0.0]),
                  (right, 0, xmax, [1.0, 0.0]),
                  (top, 1, ymax, [0.0, 1.0]),
                  (bottom, 1, ymin, [0.0, -1.0])]
        for faces, axis, coord, normal in groups:
            for ifc in faces:
                assert_almost_equal(mh.fccnd[ifc, axis], coord, decimal=12)
                assert_almost_equal(
                    [mh.fcnml[ifc, 0], mh.fcnml[ifc, 1]], normal, decimal=12)

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


class RectDomainQuadMeshTC(_RectMeshBase, unittest.TestCase):
    """One quadrilateral per grid box."""

    CELL_TYPE = 'quad'
    CELLS_PER_BOX = 1
    CLTPN = solvcon.StaticMesh.QUADRILATERAL


class RectDomainTriangleMeshTC(_RectMeshBase, _SingleBoundaryFaceTB,
                               unittest.TestCase):
    """Each grid box cut along its diagonal into two triangles."""

    CELL_TYPE = 'triangle'
    CELLS_PER_BOX = 2
    CLTPN = solvcon.StaticMesh.TRIANGLE


class RectDomainUnstructuredMeshTC(_RectMeshBase, _SingleBoundaryFaceTB,
                                   unittest.TestCase):
    """Delaunay triangulation of the rectangle: the structured boundary
    layout with jittered interior points.  The boundary tests of the base
    apply unchanged because all flavors share the boundary node layout.
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


class RectDomainMesherTC(unittest.TestCase):

    def test_unknown_cell_type(self):
        with self.assertRaises(ValueError):
            mesher = meshmod.RectDomainMesher(nx=2, ny=2)
            mesher.make_mesh(cell_type='hexagon')

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
