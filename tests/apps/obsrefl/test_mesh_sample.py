# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The oblique-shock meshes offered to the sample mesh dialog."""

import unittest

import solvcon


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
class ObliqueShockMeshTC(unittest.TestCase):

    def setUp(self):
        from solvcon import pilot
        from solvcon.pilot.apps import obsrefl
        self.mgr = pilot.RManager.instance.setUp()
        self.feature = obsrefl.ObliqueShockMesh(mgr=self.mgr)

    def test_notes_follow_the_ribbon_order(self):
        from unittest import mock
        from solvcon.pilot import visual
        with mock.patch.object(visual.MeshDrawer, 'draw') as draw:
            self.feature.draw_quad_mesh()
        # The notes name the ribbon colors, which go by the group index.
        self.assertEqual(tuple(draw.call_args.kwargs['notes']),
                         solvcon.mesh.RectDomainMesher.BOUNDARY_NAMES)

    def test_every_entry_opens_a_viewer_of_its_mesh(self):
        entries = self.feature.mesh_sample_dialog_entries()
        self.assertEqual(len(entries), 3)
        for _cat, label, _tip, func in entries:
            with self.subTest(label=label):
                before = len(self.mgr.mdiArea.subWindowList())
                viewer = func()
                self.assertEqual(before + 1,
                                 len(self.mgr.mdiArea.subWindowList()))
                self.assertEqual(viewer.mesh.nbcs, 4)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
