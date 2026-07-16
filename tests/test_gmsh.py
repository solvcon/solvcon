# Copyright (c) 2024, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING

import os

import unittest

import numpy as np

import solvcon as sc


class GmshTB(unittest.TestCase):
    TESTDIR = os.path.abspath(os.path.dirname(__file__))
    DATADIR = os.path.join(TESTDIR, "data")


class GmshTriangleTC(GmshTB):
    def test_gmsh_parsing(self):
        path = os.path.join(self.DATADIR, "gmsh_triangle.msh")

        data = open(path, 'rb').read()
        gmsh = sc.Gmsh(data)
        blk = gmsh.to_block()

        # Check nodes information
        self.assertEqual(blk.nnode, 4)
        self.assertEqual(blk.ngstnode, 3)
        np.testing.assert_almost_equal(
            [[blk.ndcrd[i, d] for d in range(blk.ndim)]
             for i in range(blk.nnode)],
            [[0.0, 0.0], [-1.0, -1.0],
             [1.0, -1.0], [0.0, 1.0]])
        # Check cells information
        self.assertEqual(blk.ncell, 3)
        self.assertEqual(list(blk.cltpn), [4, 4, 4])
        self.assertEqual(
            [[blk.clnds[i, j] for j in range(4)]
             for i in range(blk.ncell)],
            [[3, 0, 1, 2], [3, 0, 2, 3], [3, 0, 3, 1]])


class GmshRectangularTC(GmshTB):
    def setUp(self):
        # Read the Gmsh mesh file.
        path = os.path.join(self.DATADIR, "rectangle.msh")
        with open(path, 'rb') as fobj:
            data = fobj.read()
        # Create the Gmsh object.
        gmsh = sc.Gmsh(data)
        # Convert the Gmsh object to StaticMesh object.
        self.blk = gmsh.to_block()

    def test_shape(self):
        blk = self.blk
        # Test for shape data.
        self.assertEqual(blk.ndim, 2)
        self.assertEqual(blk.nbound, 40)
        self.assertEqual(blk.nnode, 104)
        self.assertEqual(blk.nface, 269)
        self.assertEqual(blk.ncell, 166)
        self.assertEqual(blk.ngstnode, 40)
        self.assertEqual(blk.ngstface, 80)
        self.assertEqual(blk.ngstcell, 40)

    @unittest.expectedFailure
    def test_type(self):
        blk = self.blk
        # TODO: all cell should be triangles including ghost cells.
        np.testing.assert_equal(blk.cltpn, 4)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
