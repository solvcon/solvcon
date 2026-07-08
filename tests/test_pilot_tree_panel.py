# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot import _mesh, _mesh_info, _entity_tree, _tree_panel
except ImportError:
    pilot = None

GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', False)


def _make_sample_mesh():
    """Two triangles and one quadrilateral, as the mesh-info panel expects."""
    core = solvcon.core
    tri = core.StaticMesh.TRIANGLE
    quad = core.StaticMesh.QUADRILATERAL
    mh = core.StaticMesh(ndim=2, nnode=6, nface=0, ncell=3)
    mh.ndcrd.ndarray[:, :] = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (2, 1)]
    mh.cltpn.ndarray[:] = [tri, tri, quad]
    mh.clnds.ndarray[:, :5] = [(3, 0, 3, 2, -1), (3, 0, 1, 3, -1),
                               (4, 1, 4, 5, 3)]
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    return mh


def _crossing_world():
    """Two lines crossing at (1, 1)."""
    world = solvcon.WorldFp64()
    world.add_line(0, 0, 2, 2)
    world.add_line(0, 2, 2, 0)
    return world


def _all_item_texts(tree):
    """Every item's text in the tree, walked depth first."""
    texts = []

    def walk(item):
        texts.append(item.text(0))
        for it in range(item.childCount()):
            walk(item.child(it))

    for it in range(tree.topLevelItemCount()):
        walk(tree.topLevelItem(it))
    return texts


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class TreePanelBaseTC(unittest.TestCase):
    def test_both_trees_share_the_base(self):
        # The mesh tree and the entity tree are the same kind of panel, so
        # they descend from the one shared base.
        self.assertTrue(
            issubclass(_mesh_info.MeshInfoTree, _tree_panel.TreePanelBase))
        self.assertTrue(
            issubclass(_entity_tree.EntityTreeWidget,
                       _tree_panel.TreePanelBase))


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class TreePanelTC(unittest.TestCase):
    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        self.status = _mesh.MeshStyleStatus(mgr=self.mgr)

    def _panel_on(self):
        feature = _entity_tree.TreePanel(
            mgr=self.mgr, style_status=self.status)
        feature.populate_menu()
        feature._action.setChecked(True)
        return feature

    def test_one_toggle_under_view_panels(self):
        feature = _entity_tree.TreePanel(
            mgr=self.mgr, style_status=self.status)
        feature.populate_menu()
        panels = self.mgr.menu_model.menu("View/Panels")
        self.assertIn(feature._action, panels.actions())

    def test_mesh_viewer_shows_mesh_tree(self):
        widget = self.mgr.add3DWidget()
        widget.updateMesh(_make_sample_mesh())
        feature = self._panel_on()
        self.assertIs(feature._stack.currentWidget(), feature._mesh_tree)
        texts = _all_item_texts(feature._mesh_tree._tree)
        self.assertIn("StaticMesh (2D)", texts)
        self.assertIn("cell: 3", texts)

    def test_canvas_shows_entity_tree(self):
        widget = self.mgr.add2DWidget()
        widget.updateWorld(_crossing_world())
        feature = self._panel_on()
        self.assertIs(feature._stack.currentWidget(), feature._entity_tree)
        texts = _all_item_texts(feature._entity_tree._tree)
        self.assertIn("World (2D)", texts)
        self.assertIn("shape: 2", texts)

    def test_follows_active_subwindow(self):
        # With both a 3D viewer and a 2D canvas open, the panel shows the
        # tree that matches whichever sub-window is active.
        mdi = self.mgr.mainWindow.centralWidget()
        w3 = self.mgr.add3DWidget()
        w3.updateMesh(_make_sample_mesh())
        sub3 = mdi.subWindowList()[-1]
        w2 = self.mgr.add2DWidget()
        w2.updateWorld(_crossing_world())
        sub2 = mdi.subWindowList()[-1]
        feature = self._panel_on()

        mdi.setActiveSubWindow(sub2)
        feature._sync()
        self.assertIs(feature._stack.currentWidget(), feature._entity_tree)

        mdi.setActiveSubWindow(sub3)
        feature._sync()
        self.assertIs(feature._stack.currentWidget(), feature._mesh_tree)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
