# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon

try:
    from solvcon import pilot
    from solvcon.pilot import _mesh, _tree_panel
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
except ImportError:
    pilot = None

GITHUB_ACTIONS = os.getenv('GITHUB_ACTIONS', False)


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


def _make_single_triangle():
    """Return a one-cell mesh.

    Its ``ncell`` of 1 differs from the sample mesh's 3, so a test cannot
    pass on mesh state left in the shared ``RManager`` singleton.
    """
    core = solvcon.core
    mh = core.StaticMesh(ndim=2, nnode=3, nface=0, ncell=1)
    mh.ndcrd.ndarray[:, :] = [(0, 0), (1, 0), (0, 1)]
    mh.cltpn.ndarray[:] = [core.StaticMesh.TRIANGLE]
    mh.clnds.ndarray[:, :4] = [(3, 0, 1, 2)]
    mh.build_interior()
    mh.build_boundary()
    mh.build_ghost()
    return mh


def _crossing_world():
    """Two lines crossing at (1, 1): shape 0 and shape 1."""
    world = solvcon.WorldFp64()
    world.add_line(0, 0, 2, 2)
    world.add_line(0, 2, 2, 0)
    return world


def _section_map(sections):
    """Map each section name to its ``{property: value}`` dict.

    Counts and Ghost share property names (node, face, cell), so the rows
    cannot be flattened into one namespace.
    """
    return {name: dict(rows) for name, rows in sections}


def _tree_sections(tree):
    """Map each group under the mesh tree root to ``{property: value}``."""
    result = {}
    root = tree.topLevelItem(0)
    for i in range(root.childCount()):
        group = root.child(i)
        pairs = {}
        for j in range(group.childCount()):
            prop, value = group.child(j).text(0).split(": ", 1)
            pairs[prop] = value
        result[group.text(0)] = pairs
    return result


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


class _CountingWorld:
    """Wraps a real world, counting describe_state calls per level so a test
    can assert the panel caches instead of resweeping on every poll."""

    def __init__(self, real):
        self._real = real
        self.calls = {"basic": 0, "diagnostics": 0}

    def describe_state(self, level="basic"):
        self.calls[level] += 1
        return self._real.describe_state(level=level)


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


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class MeshInfoTreeTC(unittest.TestCase):
    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        # The View menu and the panel share one MeshStyleStatus.
        self.status = _mesh.MeshStyleStatus(mgr=self.mgr)

    def test_tree_holds_shared_style_status(self):
        # MeshInfoTree keeps the injected MeshStyleStatus as a public
        # attribute and reads its style rows from it, rather than owning a
        # status of its own.
        widget = self.mgr.add3DWidget()
        widget.updateMesh(_make_sample_mesh())
        tree = _tree_panel.MeshInfoTree(style_status=self.status,
                                        mh=_make_sample_mesh())
        self.assertIs(tree.style_status, self.status)
        self.assertEqual(tree._style_items["wireframe"].checkState(0),
                         Qt.Checked)


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class EntityTreeWidgetTC(unittest.TestCase):
    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()

    def test_level_selector_gates_diagnostics(self):
        tree = _tree_panel.EntityTreeWidget(_crossing_world())
        self.assertIn("Diagnostics", _all_item_texts(tree._tree))
        tree._levels["basic"].setChecked(True)
        texts = _all_item_texts(tree._tree)
        self.assertNotIn("Diagnostics", texts)
        self.assertIn("shape: 2", texts)  # geometry stays

    def test_diagnostics_cached_until_world_changes(self):
        real = _crossing_world()
        world = _CountingWorld(real)
        tree = _tree_panel.EntityTreeWidget()
        tree.set_world(world)
        tree.set_world(world)  # unchanged poll reuses the cache
        self.assertEqual(world.calls["diagnostics"], 1)
        real.add_line(5, 5, 6, 6)  # a real edit flips the fingerprint
        tree.set_world(world)
        self.assertEqual(world.calls["diagnostics"], 2)

    def test_poll_timer_runs_only_while_visible(self):
        tree = _tree_panel.EntityTreeWidget(_crossing_world())
        self.assertEqual(tree._timer.interval(), tree._POLL_MS)
        tree.show()
        QApplication.processEvents()
        self.assertTrue(tree._timer.isActive())
        tree.hide()
        QApplication.processEvents()
        self.assertFalse(tree._timer.isActive())


@unittest.skipIf(GITHUB_ACTIONS or not solvcon.HAS_PILOT,
                 "GUI is not available in GitHub Actions")
class TreePanelTC(unittest.TestCase):
    def setUp(self):
        self.mgr = pilot.RManager.instance.setUp()
        self.status = _mesh.MeshStyleStatus(mgr=self.mgr)

    def _panel_on(self):
        feature = _tree_panel.TreePanel(
            mgr=self.mgr, style_status=self.status)
        feature.populate_menu()
        feature._action.setChecked(True)
        return feature

    def test_current_r3dwidget_exposes_mesh(self):
        # The mesh must be reached through the pybind11 RDomainWidget rather
        # than QMdiSubWindow.widget(), which returns a bare QWidget.
        widget = self.mgr.add3DWidget()
        widget.updateMesh(_make_sample_mesh())
        current = self.mgr.currentR3DWidget()
        self.assertIsNotNone(current)
        self.assertIsNotNone(current.mesh)
        self.assertEqual(current.mesh.ncell, 3)

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

    def test_canvas_shows_diagnostics(self):
        # End to end: the active canvas's world reaches the panel through the
        # R2DWidget.world binding and renders geometry plus diagnostics.
        world = _crossing_world()
        world.add_triangle(10, 0, 11, 0, 12, 0)  # collinear: a degeneracy
        widget = self.mgr.add2DWidget()
        widget.updateWorld(world)
        feature = self._panel_on()
        texts = _all_item_texts(feature._entity_tree._tree)
        self.assertIn("shape: 3", texts)
        self.assertIn("shape 0 crosses shape 1 at (1, 1)", texts)
        self.assertIn("shape 2: triangle (collinear)", texts)

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

    def test_boundary_toggle_drives_viewer(self):
        widget = self.mgr.add3DWidget()
        widget.updateMesh(_make_sample_mesh())
        feature = self._panel_on()
        # Record the routed (ibc, checked) while still driving the real
        # viewer hook, so the highlight build is exercised too.
        calls = []
        inner = feature._mesh_tree.boundary_toggled

        def record(ibc, checked):
            calls.append((ibc, checked))
            inner(ibc, checked)
        feature._mesh_tree.boundary_toggled = record
        root = feature._mesh_tree._tree.topLevelItem(0)
        group = next(root.child(i) for i in range(root.childCount())
                     if root.child(i).text(0) == "Boundaries")
        item = group.child(0)
        self.assertEqual(item.checkState(0), Qt.Unchecked)  # default off
        item.setCheckState(0, Qt.Checked)
        item.setCheckState(0, Qt.Unchecked)
        # Each flip routes the set index and its new state to the viewer.
        self.assertEqual(calls, [(0, True), (0, False)])

    def test_style_toggle_drives_viewer(self):
        widget = self.mgr.add3DWidget()
        widget.updateMesh(_make_sample_mesh())
        panel = self._panel_on()._mesh_tree
        # Wireframe is on by default; surface and points are off.
        self.assertEqual(panel._style_items["wireframe"].checkState(0),
                         Qt.Checked)
        self.assertEqual(panel._style_items["surface"].checkState(0),
                         Qt.Unchecked)
        # Toggling a panel check box drives the active viewer's style, one
        # style at a time without disturbing the others.
        panel._style_items["surface"].setCheckState(0, Qt.Checked)
        self.assertTrue(widget.meshStyleShown("surface"))
        self.assertTrue(widget.meshStyleShown("wireframe"))
        panel._style_items["wireframe"].setCheckState(0, Qt.Unchecked)
        self.assertFalse(widget.meshStyleShown("wireframe"))
        self.assertTrue(widget.meshStyleShown("surface"))

    def test_menu_and_panel_stay_linked(self):
        widget = self.mgr.add3DWidget()
        widget.updateMesh(_make_sample_mesh())
        panel = self._panel_on()._mesh_tree
        # The status object owns the View menu now; build it too.
        self.status.populate_menu()
        # A toggle from the menu reaches the viewer and the panel check box.
        self.status._actions["surface"].setChecked(True)
        self.assertTrue(widget.meshStyleShown("surface"))
        self.assertEqual(panel._style_items["surface"].checkState(0),
                         Qt.Checked)
        # A toggle from the panel reaches the viewer and the menu action.
        panel._style_items["points"].setCheckState(0, Qt.Checked)
        self.assertTrue(widget.meshStyleShown("points"))
        self.assertTrue(self.status._actions["points"].isChecked())

    def test_overlay_toggles_drive_viewer(self):
        widget = self.mgr.add3DWidget()
        widget.updateMesh(_make_sample_mesh())
        feature = self._panel_on()
        root = feature._mesh_tree._tree.topLevelItem(0)
        for label, attr in (("feature edges", "edges_toggled"),
                            ("normals", "normals_toggled")):
            calls = []
            inner = getattr(feature._mesh_tree, attr)

            def record(checked, calls=calls, inner=inner):
                calls.append(checked)
                inner(checked)
            setattr(feature._mesh_tree, attr, record)
            item = next(root.child(i) for i in range(root.childCount())
                        if root.child(i).text(0) == label)
            self.assertEqual(item.checkState(0), Qt.Unchecked)  # default off
            item.setCheckState(0, Qt.Checked)
            item.setCheckState(0, Qt.Unchecked)
            self.assertEqual(calls, [True, False])

    def test_mesh_viewer_without_mesh_shows_placeholder(self):
        self.mgr.add3DWidget()  # fresh viewer becomes current, no mesh
        feature = self._panel_on()
        root = feature._mesh_tree._tree.topLevelItem(0)
        self.assertIn("No mesh", root.text(0))
        self.assertEqual(root.childCount(), 0)

    def test_canvas_without_world_shows_placeholder(self):
        self.mgr.add2DWidget()  # fresh canvas becomes current, no world
        feature = self._panel_on()
        root = feature._entity_tree._tree.topLevelItem(0)
        self.assertIn("No world", root.text(0))
        self.assertEqual(root.childCount(), 0)

    def test_panel_updates_on_menu_load(self):
        # Loading from a menu creates and activates the viewer before
        # updateMesh runs, so the re-select is deferred to the event loop.
        feature = self._panel_on()
        widget = self.mgr.add3DWidget()
        widget.updateMesh(_make_single_triangle())
        QApplication.processEvents()  # allow the deferred sync to run
        sections = _tree_sections(feature._mesh_tree._tree)
        self.assertEqual(sections["Counts"]["cell"], "1")
        self.assertEqual(sections["Cell types"]["triangle"], "1")


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
