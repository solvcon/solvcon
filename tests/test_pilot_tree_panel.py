# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot.panel import _tree_panel
except ImportError:
    pilot = None


def _make_sample_mesh():
    """
    Two triangles and one quadrilateral; ``build_ghost`` adds ghost cells
    and nodes whose presence the panel must not count.
    """
    core = solvcon.core
    T = core.StaticMesh.TRIANGLE
    Q = core.StaticMesh.QUADRILATERAL
    mh = core.StaticMesh(ndim=2, nnode=6, nface=0, ncell=3)
    mh.ndcrd.ndarray[:, :] = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1)]
    mh.cltpn.ndarray[:] = [T, T, Q]
    mh.clnds.ndarray[:, :5] = [(3, 0, 3, 2, -1), (3, 0, 1, 3, -1),
                               (4, 1, 4, 5, 3)]
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    return mh


def _section_map(sections):
    """Map each section name to its ``{property: value}`` dict.

    Counts and Ghost share property names (node, face, cell), so the rows
    cannot be flattened into one namespace.
    """
    return {name: dict(rows) for name, rows in sections}


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class MakeMeshInfoTC(unittest.TestCase):
    def test_excludes_ghost_entities(self):
        info = _section_map(
            _tree_panel.MeshInfoTree.make_mesh_info(_make_sample_mesh()))
        self.assertEqual(info["Counts"]["dim"], "2")
        self.assertEqual(info["Counts"]["node"], "6")
        self.assertEqual(info["Counts"]["cell"], "3")
        # The ghost cells must not inflate the cell-type counts.
        self.assertEqual(info["Cell types"]["triangle"], "2")
        self.assertEqual(info["Cell types"]["quadrilateral"], "1")
        # The bounding box must come from the body nodes only.
        self.assertEqual(info["Bounding box"]["x"], "[0, 2]")
        self.assertEqual(info["Bounding box"]["y"], "[0, 1]")

    def test_boundary_info_groups_every_face(self):
        mh = _make_sample_mesh()
        binfo = _tree_panel.MeshInfoTree.make_boundary_info(mh)
        # With no add_bc, build_boundary gathers every boundary face into a
        # single catch-all set, so the one row must report all of them.
        self.assertGreater(mh.nbound, 0)
        self.assertEqual(binfo, [[0, mh.nbound]])


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
