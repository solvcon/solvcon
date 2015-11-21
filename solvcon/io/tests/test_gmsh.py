# -*- coding: UTF-8 -*-

import os
from unittest import TestCase

import numpy as np

from ...conf import env
from .. import gmsh
from ..gmsh import GmshIO

class TestGmshClass(TestCase):
    def test_load_physics(self):
        from io import BytesIO
        stream = BytesIO(b"""$PhysicalNames
        1
        2 1 "lower"
        $EndPhysicalNames""")
        # Check header.
        self.assertEqual(stream.readline(), b'$PhysicalNames\n')
        # Check physics body.
        res = gmsh.Gmsh._load_physics(stream)
        self.assertEqual(list(res.keys()), ['physics'])
        res = res['physics']
        self.assertEqual(len(res), 2)
        self.assertEqual(res[0], b'1')
        self.assertEqual(res[1], b'2 1 "lower"')
        # Check trailing.
        self.assertEqual(stream.readline(), b'')

    def test_load_periodic(self):
        from io import BytesIO
        stream = BytesIO(b"""$Periodic
        1
        0 1 3
        1
        1 3
        $EndPeriodic""") # a triangle.
        # Check header.
        self.assertEqual(stream.readline(), b'$Periodic\n')
        # Check periodic body.
        res = gmsh.Gmsh._load_periodic(stream)
        self.assertEqual(list(res.keys()), ['periodics'])
        res = res['periodics']
        self.assertEqual(len(res), 1)
        res = res[0]
        self.assertEqual(
            str([(key, res[key]) for key in sorted(res.keys())]),
            str([('mtag', 3), ('ndim', 0),
                 ('nodes', np.array([[1, 3]], dtype='int32')), ('stag', 1)]))
        # Check trailing.
        self.assertEqual(stream.readline(), b'')

sblk = GmshIO().load(os.path.join(env.datadir, 'gmsh_square.msh.gz'))
cblk = GmshIO().load(os.path.join(env.datadir, 'gmsh_cube.msh.gz'))

class TestGmshSquare(TestCase):
    def test_area(self):
        self.assertAlmostEqual(sblk.clvol.sum(), 4.0, 12)
    def test_top(self):
        bc = dict([(bc.name, bc) for bc in sblk.bclist])[b'top']
        self.assertAlmostEqual(sblk.fcara[bc.facn[:,0]].sum(), 2.0, 12)
        self.assertTrue((sblk.fccnd[bc.facn[:,0],1] == 1.0).all())
    def test_left(self):
        bc = dict([(bc.name, bc) for bc in sblk.bclist])[b'left']
        self.assertAlmostEqual(sblk.fcara[bc.facn[:,0]].sum(), 2.0, 12)
        self.assertTrue((sblk.fccnd[bc.facn[:,0],0] == -1.0).all())
    def test_bottom(self):
        bc = dict([(bc.name, bc) for bc in sblk.bclist])[b'bottom']
        self.assertAlmostEqual(sblk.fcara[bc.facn[:,0]].sum(), 2.0, 12)
        self.assertTrue((sblk.fccnd[bc.facn[:,0],1] == -1.0).all())
    def test_right(self):
        bc = dict([(bc.name, bc) for bc in sblk.bclist])[b'right']
        self.assertAlmostEqual(sblk.fcara[bc.facn[:,0]].sum(), 2.0, 12)
        self.assertTrue((sblk.fccnd[bc.facn[:,0],0] == 1.0).all())

class TestGmshCube(TestCase):
    def test_volume(self):
        self.assertAlmostEqual(cblk.clvol.sum(), 8.0, 12)
    def test_xnegative(self):
        bc = dict([(bc.name, bc) for bc in cblk.bclist])[b'xnegative']
        self.assertAlmostEqual(cblk.fcara[bc.facn[:,0]].sum(), 4.0, 12)
        self.assertTrue((cblk.fccnd[bc.facn[:,0],0] == -1.0).all())
    def test_xpositive(self):
        bc = dict([(bc.name, bc) for bc in cblk.bclist])[b'xpositive']
        self.assertAlmostEqual(cblk.fcara[bc.facn[:,0]].sum(), 4.0, 12)
        self.assertTrue((cblk.fccnd[bc.facn[:,0],0] == 1.0).all())
    def test_ynegative(self):
        bc = dict([(bc.name, bc) for bc in cblk.bclist])[b'ynegative']
        self.assertAlmostEqual(cblk.fcara[bc.facn[:,0]].sum(), 4.0, 12)
        self.assertTrue((cblk.fccnd[bc.facn[:,0],1] == -1.0).all())
    def test_ypositive(self):
        bc = dict([(bc.name, bc) for bc in cblk.bclist])[b'ypositive']
        self.assertAlmostEqual(cblk.fcara[bc.facn[:,0]].sum(), 4.0, 12)
        self.assertTrue((cblk.fccnd[bc.facn[:,0],1] == 1.0).all())
    def test_znegative(self):
        bc = dict([(bc.name, bc) for bc in cblk.bclist])[b'znegative']
        self.assertAlmostEqual(cblk.fcara[bc.facn[:,0]].sum(), 4.0, 12)
        self.assertTrue((cblk.fccnd[bc.facn[:,0],2] == -1.0).all())
    def test_zpositive(self):
        bc = dict([(bc.name, bc) for bc in cblk.bclist])[b'zpositive']
        self.assertAlmostEqual(cblk.fcara[bc.facn[:,0]].sum(), 4.0, 12)
        self.assertTrue((cblk.fccnd[bc.facn[:,0],2] == 1.0).all())
