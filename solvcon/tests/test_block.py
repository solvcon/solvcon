# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2012 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

from unittest import TestCase
from ..testing import get_blk_from_oblique_neu, get_blk_from_sample_neu

class TestCreation(TestCase):
    def test_import(self):
        from ..block import Block

    def test_use_centroid_default(self):
        from ..block import Block
        blk = Block()
        self.assertFalse(blk.use_incenter)
        self.assertEqual(str(blk),
            '[Block (0D/centroid): 0 nodes, 0 faces (0 BC), 0 cells]')

    def test_use_centroid(self):
        from ..block import Block
        blk = Block(use_incenter=False)
        self.assertFalse(blk.use_incenter)
        self.assertEqual(str(blk),
            '[Block (0D/centroid): 0 nodes, 0 faces (0 BC), 0 cells]')

    def test_use_incenter(self):
        from ..block import Block
        blk = Block(use_incenter=True)
        self.assertTrue(blk.use_incenter)
        self.assertEqual(str(blk),
            '[Block (0D/incenter): 0 nodes, 0 faces (0 BC), 0 cells]')

    def test_fpdtype(self):
        from ..conf import env
        from ..dependency import str_of
        from ..block import Block
        blk = Block()
        self.assertEqual(blk.ndcrd.dtype, env.fpdtype)
        self.assertEqual(blk.fpdtype, env.fpdtype)
        self.assertEqual(blk.fpdtypestr, str_of(env.fpdtype))

    def test_MAX(self):
        from .. import block
        self.assertEqual(block.MAX_FCNND, 4)
        self.assertEqual(block.MAX_CLNND, 8)
        self.assertEqual(block.MAX_CLNFC, 6)
        self.assertEqual(block.Block.FCMND, 4)
        self.assertEqual(block.Block.CLMND, 8)
        self.assertEqual(block.Block.CLMFC, 6)

    def test_metric(self):
        from ..block import Block
        # build a simple 2D triangle with 4 subtriangles.
        blk = Block(ndim=2, nnode=4, nface=6, ncell=3, nbound=3)
        blk.ndcrd[0,:] = (0,0)
        blk.ndcrd[1,:] = (-1,-1)
        blk.ndcrd[2,:] = (1,-1)
        blk.ndcrd[3,:] = (0,1)
        blk.cltpn[:] = 3
        blk.clnds[0,:4] = (3, 0,1,2)
        blk.clnds[1,:4] = (3, 0,2,3)
        blk.clnds[2,:4] = (3, 0,3,1)
        blk.build_interior()
        # test for volume (actually area in 2D).
        self.assertEqual(blk.clvol[0], 1)
        self.assertEqual(blk.clvol[1], .5)
        self.assertEqual(blk.clvol[2], .5)
        self.assertEqual(blk.clvol.sum(), 2)
        # test for ghost information.
        blk.build_boundary()
        blk.build_ghost()
        self.assertEqual(blk.clvol[0], 1)
        self.assertEqual(blk.clvol[1], .5)
        self.assertEqual(blk.clvol[2], .5)
        self.assertEqual(blk.clvol.sum(), 2)

class GroupTest(TestCase):
    __test__ = False
    testblock = None

    def test_groupname(self):
        blk = self.testblock
        self.assertEqual(len(blk.grpnames), 1)
        self.assertEqual(blk.grpnames[0], 'fluid')

    def test_clgrp(self):
        blk = self.testblock
        icl = 0
        while icl < blk.ncell:
            self.assertEqual(blk.clgrp[icl], 0)
            icl += 1

class TestGroup2D(GroupTest):
    __test__ = True
    testblock = get_blk_from_oblique_neu()

class TestGroup3D(GroupTest):
    __test__ = True
    testblock = get_blk_from_sample_neu()

class MetricTest(TestCase):
    __test__ = False
    testblock = None

    def test_fcnml(self):
        blk = self.testblock
        ndim = blk.ndim
        bfcs = blk.bndfcs[:,0]
        bcls = blk.fccls[bfcs,0]    # interior cells next to boundary.
        igcl = 0
        while igcl < blk.ngstcell:
            ibfc = bfcs[igcl]   # current boundary face.
            iicl = bcls[igcl]   # current interior cell.
            dvec = blk.fccnd[ibfc,:] - blk.clcnd[iicl,:]
            leng = (dvec[:]*blk.fcnml[ibfc,:]).sum()
            self.assertTrue(leng > 0)
            # check for nodes not belong to boundary face.
            clnnd = blk.clnds[iicl,0]
            for ind in blk.clnds[iicl,1:clnnd+1]:
                fcnnd = blk.fcnds[ibfc,0]
                if ind not in blk.fcnds[ibfc,1:fcnnd+1]:
                    dvec = blk.fccnd[ibfc,:] - blk.ndcrd[ind,:]
                    leng = (dvec[:]*blk.fcnml[ibfc,:]).sum()
                    self.assertTrue(leng > 0)
            igcl += 1

class TestMetric2D(MetricTest):
    __test__ = True
    testblock = get_blk_from_oblique_neu()

class TestMetric3D(MetricTest):
    __test__ = True
    testblock = get_blk_from_sample_neu()

class GhostTest(TestCase):
    __test__ = False
    testblock = None
    rounding_to = 100

    def test_cltpn(self):
        blk = self.testblock
        bfcs = blk.bndfcs[:,0]
        bcls = blk.fccls[bfcs,0]    # interior cells next to boundary.
        igcl = 0
        while igcl < blk.ngstcell:
            iicl = bcls[igcl]
            self.assertEqual(blk.gstcltpn[igcl], blk.cltpn[iicl])
            igcl += 1

    def test_clgrp(self):
        blk = self.testblock
        bfcs = blk.bndfcs[:,0]
        bcls = blk.fccls[bfcs,0]    # interior cells next to boundary.
        igcl = 0
        while igcl < blk.ngstcell:
            iicl = bcls[igcl]
            self.assertEqual(blk.gstclgrp[igcl], blk.clgrp[iicl])
            igcl += 1

    def test_fctpn(self):
        blk = self.testblock
        bfcs = blk.bndfcs[:,0]
        bcls = blk.fccls[bfcs,0]    # interior cells next to boundary.
        igcl = 0
        while igcl < blk.ngstcell:
            iicl = bcls[igcl]
            for it in range(1,blk.gstclfcs[igcl,0]+1):
                igfc = blk.gstclfcs[igcl,it]
                iifc = blk.clfcs[iicl,it]
                if igfc >= 0:
                    self.assertEqual(igfc, iifc)
                else:
                    igfc = -igfc - 1
                    self.assertEqual(blk.gstfctpn[igfc], blk.fctpn[iifc])
            igcl += 1

    def test_ndcrd(self):
        blk = self.testblock
        ndim = blk.ndim
        bfcs = blk.bndfcs[:,0]
        bcls = blk.fccls[bfcs,0]    # interior cells next to boundary.
        igcl = 0
        while igcl < blk.ngstcell:
            ibfc = bfcs[igcl]   # current boundary face.
            iicl = bcls[igcl]   # current interior cell.
            for i in range(1,blk.gstclnds[igcl,0]+1):
                ignd = blk.gstclnds[igcl,i] # current ghost node.
                iind = blk.clnds[iicl,i]    # current mirrored interior node.
                if ignd >= 0:   # node for ghost cell is real node.
                    self.assertEqual(ignd, iind)
                else:
                    ignd = -ignd - 1    # flip index for ghost node.
                    v1 = blk.gstndcrd[ignd,:] - blk.fccnd[ibfc,:]
                    v2 = blk.ndcrd[iind,:] - blk.fccnd[ibfc,:]
                    # normal component.
                    v1n = (v1[:]*blk.fcnml[ibfc,:]).sum()*blk.fcnml[ibfc,:]
                    v2n = (v2[:]*blk.fcnml[ibfc,:]).sum()*blk.fcnml[ibfc,:]
                    for idim in range(ndim):
                        # opposite direction.
                        self.assertAlmostEqual(v1n[idim], -v2n[idim],
                            self.rounding_to)
                    # tangent component.
                    v1t = v1-v1n
                    v2t = v2-v2n
                    for idim in range(ndim):
                        # same direction.
                        self.assertAlmostEqual(v1t[idim], v2t[idim],
                            self.rounding_to)
            igcl += 1

    def test_fccnd(self):
        blk = self.testblock
        ndim = blk.ndim
        bfcs = blk.bndfcs[:,0]
        bcls = blk.fccls[bfcs,0]    # interior cells next to boundary.
        igcl = 0
        while igcl < blk.ngstcell:
            ibfc = bfcs[igcl]
            iicl = bcls[igcl]
            for it in range(1,blk.gstclfcs[igcl,0]+1):
                igfc = blk.gstclfcs[igcl,it]
                iifc = blk.clfcs[iicl,it]
                if igfc >= 0:
                    self.assertEqual(igfc, iifc)
                else:
                    igfc = -igfc - 1
                    v1 = blk.gstfccnd[igfc,:] - blk.fccnd[ibfc,:]
                    v2 = blk.fccnd[iifc,:] - blk.fccnd[ibfc,:]
                    # normal components.
                    v1n = (v1*blk.fcnml[ibfc,:]).sum()*blk.fcnml[ibfc,:]
                    v2n = (v2*blk.fcnml[ibfc,:]).sum()*blk.fcnml[ibfc,:]
                    for idim in range(ndim):
                        # opposite direction.
                        self.assertAlmostEqual(v1n[idim], -v2n[idim],
                            self.rounding_to)
                    # tangent component.
                    v1t = v1-v1n
                    v2t = v2-v2n
                    for idim in range(ndim):
                        # same direction.
                        self.assertAlmostEqual(v1t[idim], v2t[idim],
                            self.rounding_to)
            igcl += 1

    def test_fcnml(self):
        blk = self.testblock
        ndim = blk.ndim
        bfcs = blk.bndfcs[:,0]
        bcls = blk.fccls[bfcs,0]    # interior cells next to boundary.
        igcl = 0
        while igcl < blk.ngstcell:
            ibfc = bfcs[igcl]
            iicl = bcls[igcl]
            for it in range(1,blk.gstclfcs[igcl,0]+1):
                igfc = blk.gstclfcs[igcl,it]
                iifc = blk.clfcs[iicl,it]
                if igfc >= 0:
                    self.assertEqual(igfc, iifc)
                else:
                    igfc = -igfc - 1
                    v1 = blk.gstfcnml[igfc,:]
                    v2 = blk.fcnml[iifc,:]
                    # flip interior face normal vector to have proper direction.
                    if blk.fccls[iifc,0] != iicl:
                        v2 = -v2
                    # normal components.
                    v1n = (v1*blk.fcnml[ibfc,:]).sum()*blk.fcnml[ibfc,:]
                    v2n = (v2*blk.fcnml[ibfc,:]).sum()*blk.fcnml[ibfc,:]
                    for idim in range(ndim):
                        # opposite direction.
                        self.assertAlmostEqual(v1n[idim], -v2n[idim],
                            self.rounding_to)
                    # tangent component.
                    v1t = v1-v1n
                    v2t = v2-v2n
                    for idim in range(ndim):
                        # same direction.
                        self.assertAlmostEqual(v1t[idim], v2t[idim],
                            self.rounding_to)
            igcl += 1

    def test_fcara(self):
        blk = self.testblock
        bfcs = blk.bndfcs[:,0]
        bcls = blk.fccls[bfcs,0]    # interior cells next to boundary.
        igcl = 0
        while igcl < blk.ngstcell:
            ibfc = bfcs[igcl]
            iicl = bcls[igcl]
            for it in range(1,blk.gstclfcs[igcl,0]+1):
                igfc = blk.gstclfcs[igcl,it]
                iifc = blk.clfcs[iicl,it]
                if igfc >= 0:
                    self.assertEqual(igfc, iifc)
                else:
                    igfc = -igfc - 1
                    self.assertAlmostEqual(
                        blk.gstfcara[igfc], blk.fcara[iifc], self.rounding_to)
            igcl += 1

    def test_clvol(self):
        blk = self.testblock
        bfcs = blk.bndfcs[:,0]
        bcls = blk.fccls[bfcs,0]    # interior cells next to boundary.
        igcl = 0
        while igcl < blk.ngstcell:
            iicl = bcls[igcl]
            self.assertAlmostEqual(blk.gstclvol[igcl], blk.clvol[iicl],
                self.rounding_to)
            igcl += 1

"""class TestGhostSingle2D(GhostTest):
    __test__ = True
    testblock = get_blk_from_oblique_neu(fpdtype='float32')
    rounding_to = 6"""

class TestGhostDouble2D(GhostTest):
    __test__ = True
    testblock = get_blk_from_oblique_neu(fpdtype='float64')
    rounding_to = 15

"""class TestGhostSingle3D(GhostTest):
    __test__ = True
    testblock = get_blk_from_sample_neu(fpdtype='float32')
    rounding_to = 4"""

class TestGhostDouble3D(GhostTest):
    __test__ = True
    testblock = get_blk_from_sample_neu(fpdtype='float64')
    rounding_to = 13

class TestShared(TestCase):
    oblique = get_blk_from_oblique_neu()

    def assertSharedGhost(self, nint, ngst, shared, ghost):
        """
        For every ghost entity, assert it located in the reversed order than
        interior entity, in the shared array.

        @param nint: number of interior entity.
        @type nint: int
        @param ngst: number of ghost entity.
        @type ngst: int
        @param shared: shared array.
        @type shared: numpy.ndarray
        @param ghost: ghost array.
        @type ghost: numpy.ndarray
        @return: nothing.
        """
        from random import randint
        # shape of arrays.
        self.assertEqual(len(shared.shape), 1)
        self.assertEqual(len(ghost.shape), 1)
        self.assertEqual(shared.shape[0], nint+ngst)
        self.assertEqual(ghost.shape[0], ngst)
        # test for each ghost entity.
        for igst in range(ngst):    # 0-indexed.
            oval = ghost[igst]
            tval = oval
            while tval == oval:
                tval = randint(0,10000000)
            self.assertNotEqual(oval, tval)
            ghost[igst] = tval
            self.assertEqual(shared[ngst-1-igst], tval)
            ghost[igst] = oval
            self.assertEqual(shared[ngst-1-igst], oval)

    def test_metrics(self):
        blk = self.oblique
        for idim in range(blk.ndim):
            self.assertSharedGhost(blk.nnode, blk.ngstnode,
                blk.shndcrd[:,idim], blk.gstndcrd[:,idim])
            self.assertSharedGhost(blk.nface, blk.ngstface,
                blk.shfccnd[:,idim], blk.gstfccnd[:,idim])
            self.assertSharedGhost(blk.nface, blk.ngstface,
                blk.shfcnml[:,idim], blk.gstfcnml[:,idim])
            self.assertSharedGhost(blk.ncell, blk.ngstcell,
                blk.shclcnd[:,idim], blk.gstclcnd[:,idim])
        self.assertSharedGhost(blk.nface, blk.ngstface,
            blk.shfcara, blk.gstfcara)
        self.assertSharedGhost(blk.ncell, blk.ngstcell,
            blk.shclvol, blk.gstclvol)

    def test_type(self):
        blk = self.oblique
        self.assertSharedGhost(blk.ncell, blk.ngstcell,
            blk.shcltpn, blk.gstcltpn)
        self.assertSharedGhost(blk.ncell, blk.ngstcell,
            blk.shclgrp, blk.gstclgrp)
        self.assertSharedGhost(blk.nface, blk.ngstface,
            blk.shfctpn, blk.gstfctpn)

    def test_conn(self):
        blk = self.oblique
        for it in range(blk.FCMND+1):
            self.assertSharedGhost(blk.nface, blk.ngstface,
                blk.shfcnds[:,it], blk.gstfcnds[:,it])
        for it in range(4):
            self.assertSharedGhost(blk.nface, blk.ngstface,
                blk.shfccls[:,it], blk.gstfccls[:,it])
        for it in range(blk.CLMND+1):
            self.assertSharedGhost(blk.ncell, blk.ngstcell,
                blk.shclnds[:,it], blk.gstclnds[:,it])
        for it in range(blk.CLMFC+1):
            self.assertSharedGhost(blk.ncell, blk.ngstcell,
                blk.shclfcs[:,it], blk.gstclfcs[:,it])

class TestMeshData(TestCase):
    oblique = get_blk_from_oblique_neu()
    sample = get_blk_from_sample_neu()

    def test_simpex(self):
        self.assertTrue(self.oblique.check_simplex())
        self.assertFalse(self.sample.check_simplex())
