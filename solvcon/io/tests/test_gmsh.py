# -*- coding: UTF-8 -*-


import os
from unittest import TestCase
import gzip
import tempfile

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

class TestGmshIO(TestCase):
    def test_plaintext_file(self):
        with tempfile.TemporaryDirectory() as wdir:
            sfname = os.path.join(env.datadir, 'gmsh_square.msh.gz')
            dfname = os.path.join(wdir, 'gmsh_square.msh')
            with gzip.open(sfname) as sfobj:
                data = sfobj.read()
                with open(dfname, 'wb') as dfobj:
                    dfobj.write(data)
            sblk = GmshIO().load(dfname)

    def test_gzip_file(self):
        sfname = os.path.join(env.datadir, 'gmsh_square.msh.gz')
        sblk = GmshIO().load(sfname)

class TestGmshSquare(TestCase):
    def setUp(self):
        sfname = os.path.join(env.datadir, 'gmsh_square.msh.gz')
        self.sblk = GmshIO().load(sfname)
    def test_area(self):
        self.assertAlmostEqual(self.sblk.clvol.sum(), 4.0, 12)
    def test_top(self):
        bc = dict([(bc.name, bc) for bc in self.sblk.bclist])['top']
        self.assertAlmostEqual(self.sblk.fcara[bc.facn[:,0]].sum(), 2.0, 12)
        self.assertTrue((self.sblk.fccnd[bc.facn[:,0],1] == 1.0).all())
    def test_left(self):
        bc = dict([(bc.name, bc) for bc in self.sblk.bclist])['left']
        self.assertAlmostEqual(self.sblk.fcara[bc.facn[:,0]].sum(), 2.0, 12)
        self.assertTrue((self.sblk.fccnd[bc.facn[:,0],0] == -1.0).all())
    def test_bottom(self):
        bc = dict([(bc.name, bc) for bc in self.sblk.bclist])['bottom']
        self.assertAlmostEqual(self.sblk.fcara[bc.facn[:,0]].sum(), 2.0, 12)
        self.assertTrue((self.sblk.fccnd[bc.facn[:,0],1] == -1.0).all())
    def test_right(self):
        bc = dict([(bc.name, bc) for bc in self.sblk.bclist])['right']
        self.assertAlmostEqual(self.sblk.fcara[bc.facn[:,0]].sum(), 2.0, 12)
        self.assertTrue((self.sblk.fccnd[bc.facn[:,0],0] == 1.0).all())

class TestGmshCube(TestCase):
    def setUp(self):
        sfname = os.path.join(env.datadir, 'gmsh_cube.msh.gz')
        self.cblk = GmshIO().load(sfname)
    def test_volume(self):
        self.assertAlmostEqual(self.cblk.clvol.sum(), 8.0, 12)
    def test_xnegative(self):
        bc = dict([(bc.name, bc) for bc in self.cblk.bclist])['xnegative']
        self.assertAlmostEqual(self.cblk.fcara[bc.facn[:,0]].sum(), 4.0, 12)
        self.assertTrue((self.cblk.fccnd[bc.facn[:,0],0] == -1.0).all())
    def test_xpositive(self):
        bc = dict([(bc.name, bc) for bc in self.cblk.bclist])['xpositive']
        self.assertAlmostEqual(self.cblk.fcara[bc.facn[:,0]].sum(), 4.0, 12)
        self.assertTrue((self.cblk.fccnd[bc.facn[:,0],0] == 1.0).all())
    def test_ynegative(self):
        bc = dict([(bc.name, bc) for bc in self.cblk.bclist])['ynegative']
        self.assertAlmostEqual(self.cblk.fcara[bc.facn[:,0]].sum(), 4.0, 12)
        self.assertTrue((self.cblk.fccnd[bc.facn[:,0],1] == -1.0).all())
    def test_ypositive(self):
        bc = dict([(bc.name, bc) for bc in self.cblk.bclist])['ypositive']
        self.assertAlmostEqual(self.cblk.fcara[bc.facn[:,0]].sum(), 4.0, 12)
        self.assertTrue((self.cblk.fccnd[bc.facn[:,0],1] == 1.0).all())
    def test_znegative(self):
        bc = dict([(bc.name, bc) for bc in self.cblk.bclist])['znegative']
        self.assertAlmostEqual(self.cblk.fcara[bc.facn[:,0]].sum(), 4.0, 12)
        self.assertTrue((self.cblk.fccnd[bc.facn[:,0],2] == -1.0).all())
    def test_zpositive(self):
        bc = dict([(bc.name, bc) for bc in self.cblk.bclist])['zpositive']
        self.assertAlmostEqual(self.cblk.fcara[bc.facn[:,0]].sum(), 4.0, 12)
        self.assertTrue((self.cblk.fccnd[bc.facn[:,0],2] == 1.0).all())
