# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


import unittest

import solvcon


@unittest.skipUnless(solvcon.HAS_PILOT, "Qt pilot is not built")
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
             '2dmix_small', '2dmix_large', '3dmix'))

    def test_unknown_name_is_rejected(self):
        with self.assertRaises(AttributeError):
            self.samples.make('bogus')


# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
