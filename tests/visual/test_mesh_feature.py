# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import os
import unittest

import solvcon
from solvcon import testing


@testing.requires_pilot
class SampleMeshTC(unittest.TestCase):
    """The non-GUI sample-mesh data class and its name discovery."""

    def setUp(self):
        from solvcon.pilot.visual import _mesh
        self.samples = _mesh.SampleMesh()

    def test_every_discovered_name_builds_a_mesh(self):
        for name in self.samples.names():
            mh = self.samples.make(name)
            self.assertIn(mh.ndim, (2, 3))
            self.assertGreater(mh.ncell, 0)

    def test_dimensionality_of_named_samples(self):
        self.assertEqual(self.samples.make('triangle').ndim, 2)
        self.assertEqual(self.samples.make('tetrahedron').ndim, 3)
        self.assertEqual(self.samples.make('3dmix').ndim, 3)

    def test_names_derive_from_the_mesh_methods(self):
        self.assertEqual(
            self.samples.names(),
            ('triangle', 'tetrahedron', 'solvcon_2dtext',
             '2dmix_small', '2dmix_large', '3dmix',
             'rect_quad', 'rect_triangle', 'rect_unstructured'))

    def test_unknown_name_is_rejected(self):
        with self.assertRaises(AttributeError):
            self.samples.make('bogus')

    def test_rectangle_samples_are_the_mesher_flavors(self):
        # tests/mesh/test_regular_domain.py covers the meshes themselves.
        mesher = solvcon.mesh.RectDomainMesher()
        for cell_type in ('quad', 'triangle', 'unstructured'):
            with self.subTest(cell_type=cell_type):
                mh = self.samples.make(f'rect_{cell_type}')
                ref = mesher.make_mesh(cell_type=cell_type)
                self.assertEqual((mh.ncell, mh.nnode),
                                 (ref.ncell, ref.nnode))


def _make_rectangle():
    """A quad mesh with one named boundary group per domain edge."""
    return solvcon.mesh.RectDomainMesher(nx=4, ny=2).make_mesh(
        cell_type='quad')


class _FakeViewer:
    def __init__(self):
        self.mesh = None
        self.calls = []

    def updateMesh(self, mh):
        self.mesh = mh
        self.calls.append(('mesh', mh))

    def showAxis(self, flag):
        self.calls.append(('axis', flag))

    def showBoundary(self, ibc, flag):
        self.calls.append(('boundary', ibc, flag))


class _FakeConsole:
    def __init__(self):
        self.history = []

    def writeToHistory(self, text):
        self.history.append(text)


class _FakeManager:
    def __init__(self):
        self.viewers = []
        self.pycon = _FakeConsole()

    def add3DWidget(self):
        self.viewers.append(_FakeViewer())
        return self.viewers[-1]


@testing.requires_pilot
class MeshDrawerTC(unittest.TestCase):
    """The viewer sequence every mesh view shares, on a stand-in manager."""

    def setUp(self):
        from solvcon.pilot import visual
        self.mgr = _FakeManager()
        self.drawer = visual.MeshDrawer(self.mgr)
        self.mh = _make_rectangle()

    def test_show_ribbons_every_boundary_group(self):
        w = self.drawer.show(self.mh)
        self.assertEqual(self.mgr.viewers, [w])
        self.assertEqual(w.calls[:2], [('mesh', self.mh), ('axis', True)])
        self.assertEqual(
            w.calls[2:], [('boundary', i, True) for i in range(4)])
        # Only draw() writes the summary; show() keeps the console quiet.
        self.assertEqual(self.mgr.pycon.history, [])

    def test_draw_writes_one_summary(self):
        w = self.drawer.draw(self.mh, "quad")
        self.assertEqual(self.mgr.viewers, [w])
        self.assertEqual(len(self.mgr.pycon.history), 1)
        self.assertTrue(self.mgr.pycon.history[0].startswith("quad: "))

    def test_each_draw_opens_a_new_viewer(self):
        first = self.drawer.draw(self.mh, "a")
        second = self.drawer.draw(self.mh, "b")
        self.assertEqual(self.mgr.viewers, [first, second])

    def test_given_viewer_drops_the_old_ribbons(self):
        from solvcon.pilot import visual
        w = self.drawer.show(self.mh)
        tri = visual.SampleMesh().make('triangle')
        w.calls.clear()
        self.assertIs(self.drawer.show(tri, viewer=w), w)
        self.assertEqual(self.mgr.viewers, [w])
        # The four rectangle ribbons go before the triangle takes their place.
        self.assertEqual(
            w.calls, [('boundary', i, False) for i in range(4)]
            + [('mesh', tri), ('axis', True), ('boundary', 0, True)])

    def test_draw_into_a_given_viewer(self):
        w = self.drawer.draw(self.mh, "a")
        self.assertIs(self.drawer.draw(self.mh, "b", viewer=w), w)
        self.assertEqual(self.mgr.viewers, [w])
        self.assertEqual(len(self.mgr.pycon.history), 2)


@testing.requires_pilot
class MeshSummaryTC(unittest.TestCase):
    """The mesh summary every mesh view writes to the console."""

    def setUp(self):
        from solvcon.pilot import visual
        self.visual = visual

    def _summary(self, mh, label, notes=None):
        mgr = _FakeManager()
        self.visual.MeshDrawer(mgr).draw(mh, label, notes=notes)
        return mgr.pycon.history[-1]

    def test_unnamed_boundary_group_is_called_by_index(self):
        mh = self.visual.SampleMesh().make('triangle')
        self.assertEqual(
            self._summary(mh, "triangle"),
            f"triangle: 3 cells, {mh.nedge} edges\n"
            f"boundary ribbons: bc0 ({mh.bc(0).nbound} faces)\n")

    def test_boundary_groups_go_by_their_names(self):
        mh = _make_rectangle()
        text = self._summary(mh, "quad")
        self.assertEqual(len(text.splitlines()), 2)
        for ibc in range(mh.nbcs):
            bc = mh.bc(ibc)
            self.assertIn(f"{bc.name} ({bc.nbound} faces)", text)

    def test_notes_replace_the_names(self):
        mh = _make_rectangle()
        text = self._summary(mh, "quad", notes={'left': "red inlet"})
        self.assertIn(f"red inlet ({mh.bc('left').nbound} faces)", text)
        self.assertNotIn("left (", text)
        # A group without a note keeps its name.
        self.assertIn(f"top ({mh.bc('top').nbound} faces)", text)


@testing.requires_pilot
class MeshFeatureDrawTC(unittest.TestCase):
    """The sample meshes and the Gmsh file open in a real viewer."""

    def setUp(self):
        from solvcon import pilot
        self.mgr = pilot.RManager.instance.setUp()

    def _nwindow(self):
        return len(self.mgr.mdiArea.subWindowList())

    def test_every_sample_entry_opens_a_viewer_of_its_mesh(self):
        from solvcon.pilot import visual
        feature = visual.SampleMeshFeature(mgr=self.mgr)
        samples = visual.SampleMesh()
        entries = feature.mesh_sample_dialog_entries()
        self.assertEqual(len(entries), len(samples.names()))
        for (_cat, label, _tip, func), name in zip(entries, samples.names()):
            with self.subTest(label=label):
                before = self._nwindow()
                viewer = func()
                self.assertEqual(before + 1, self._nwindow())
                self.assertEqual(viewer.mesh.ncell,
                                 samples.make(name).ncell)

    def test_gmsh_file_opens_a_viewer(self):
        from solvcon.pilot import visual
        feature = visual.GmshFileDialog(mgr=self.mgr)
        path = os.path.join(os.path.dirname(__file__), os.pardir, "data",
                            "rectangle.msh")
        before = self._nwindow()
        viewer = feature._load_gmsh_file(path)
        self.assertEqual(before + 1, self._nwindow())
        self.assertGreater(viewer.mesh.nbcs, 0)

    def test_drawer_replaces_the_mesh_of_a_given_viewer(self):
        from solvcon.pilot import visual
        drawer = visual.MeshDrawer(self.mgr)
        viewer = drawer.draw(_make_rectangle(), "quad")
        before = self._nwindow()
        tri = visual.SampleMesh().make('triangle')
        self.assertIs(drawer.draw(tri, "triangle", viewer=viewer), viewer)
        self.assertEqual(before, self._nwindow())
        self.assertEqual(viewer.mesh.ncell, tri.ncell)


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
