# -*- coding: UTF-8 -*-

import os
from unittest import TestCase
from ...conf import env
from ..gmsh import GmshIO

sblk = GmshIO().load(os.path.join(env.datadir, 'gmsh_square.msh.gz'))
cblk = GmshIO().load(os.path.join(env.datadir, 'gmsh_cube.msh.gz'))

class TestGmshSquare(TestCase):
    def test_area(self):
        self.assertAlmostEqual(sblk.clvol.sum(), 4.0, 12)
    def test_top(self):
        bc = dict([(bc.name, bc) for bc in sblk.bclist])['top']
        self.assertAlmostEqual(sblk.fcara[bc.facn[:,0]].sum(), 2.0, 12)
        self.assertTrue((sblk.fccnd[bc.facn[:,0],1] == 1.0).all())
    def test_left(self):
        bc = dict([(bc.name, bc) for bc in sblk.bclist])['left']
        self.assertAlmostEqual(sblk.fcara[bc.facn[:,0]].sum(), 2.0, 12)
        self.assertTrue((sblk.fccnd[bc.facn[:,0],0] == -1.0).all())
    def test_bottom(self):
        bc = dict([(bc.name, bc) for bc in sblk.bclist])['bottom']
        self.assertAlmostEqual(sblk.fcara[bc.facn[:,0]].sum(), 2.0, 12)
        self.assertTrue((sblk.fccnd[bc.facn[:,0],1] == -1.0).all())
    def test_right(self):
        bc = dict([(bc.name, bc) for bc in sblk.bclist])['right']
        self.assertAlmostEqual(sblk.fcara[bc.facn[:,0]].sum(), 2.0, 12)
        self.assertTrue((sblk.fccnd[bc.facn[:,0],0] == 1.0).all())

class TestGmshCube(TestCase):
    def test_volume(self):
        self.assertAlmostEqual(cblk.clvol.sum(), 8.0, 12)
    def test_xnegative(self):
        bc = dict([(bc.name, bc) for bc in cblk.bclist])['xnegative']
        self.assertAlmostEqual(cblk.fcara[bc.facn[:,0]].sum(), 4.0, 12)
        self.assertTrue((cblk.fccnd[bc.facn[:,0],0] == -1.0).all())
    def test_xpositive(self):
        bc = dict([(bc.name, bc) for bc in cblk.bclist])['xpositive']
        self.assertAlmostEqual(cblk.fcara[bc.facn[:,0]].sum(), 4.0, 12)
        self.assertTrue((cblk.fccnd[bc.facn[:,0],0] == 1.0).all())
    def test_ynegative(self):
        bc = dict([(bc.name, bc) for bc in cblk.bclist])['ynegative']
        self.assertAlmostEqual(cblk.fcara[bc.facn[:,0]].sum(), 4.0, 12)
        self.assertTrue((cblk.fccnd[bc.facn[:,0],1] == -1.0).all())
    def test_ypositive(self):
        bc = dict([(bc.name, bc) for bc in cblk.bclist])['ypositive']
        self.assertAlmostEqual(cblk.fcara[bc.facn[:,0]].sum(), 4.0, 12)
        self.assertTrue((cblk.fccnd[bc.facn[:,0],1] == 1.0).all())
    def test_znegative(self):
        bc = dict([(bc.name, bc) for bc in cblk.bclist])['znegative']
        self.assertAlmostEqual(cblk.fcara[bc.facn[:,0]].sum(), 4.0, 12)
        self.assertTrue((cblk.fccnd[bc.facn[:,0],2] == -1.0).all())
    def test_zpositive(self):
        bc = dict([(bc.name, bc) for bc in cblk.bclist])['zpositive']
        self.assertAlmostEqual(cblk.fcara[bc.facn[:,0]].sum(), 4.0, 12)
        self.assertTrue((cblk.fccnd[bc.facn[:,0],2] == 1.0).all())
