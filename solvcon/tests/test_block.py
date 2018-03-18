# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2012 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.


import unittest
import pickle
from unittest import TestCase

import numpy as np

from .. import dependency
dependency.import_module_may_fail('..march')
from .. import block
from ..testing import get_blk_from_oblique_neu, get_blk_from_sample_neu

class TestCreation(TestCase):
    def test_table_names(self):
        from ..block import Block
        self.assertEqual(
            ('ndcrd', 'fccnd', 'fcnml', 'fcara', 'clcnd', 'clvol'),
            Block.GEOMETRY_TABLE_NAMES)
        self.assertEqual(
            ('fctpn', 'cltpn', 'clgrp'),
            Block.META_TABLE_NAMES)
        self.assertEqual(
            ('fcnds', 'fccls', 'clnds', 'clfcs'),
            Block.CONNECTIVITY_TABLE_NAMES)
        self.assertEqual(
            Block.GEOMETRY_TABLE_NAMES
          + Block.META_TABLE_NAMES
          + Block.CONNECTIVITY_TABLE_NAMES,
            Block.TABLE_NAMES)

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
        self.assertEqual(block.UnstructuredBlock.FCMND, 4)
        self.assertEqual(block.UnstructuredBlock.CLMND, 8)
        self.assertEqual(block.UnstructuredBlock.CLMFC, 6)

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
        self.assertEqual(blk.nface, 6)
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

class PickleTC(TestCase):

    def setUp(self):
        self.blk = get_blk_from_oblique_neu()

    def _check_shape(self, newblk, blk):
        # shape.
        self.assertEqual(newblk.ndim, blk.ndim)
        self.assertEqual(newblk.nnode, blk.nnode)
        self.assertEqual(newblk.nface, blk.nface)
        self.assertEqual(newblk.ncell, blk.ncell)
        self.assertEqual(newblk.nbound, blk.nbound)
        self.assertEqual(newblk.ngstnode, blk.ngstnode)
        self.assertEqual(newblk.ngstface, blk.ngstface)
        self.assertEqual(newblk.ngstcell, blk.ngstcell)
        # serial number.
        self.assertEqual(newblk.blkn, blk.blkn)

    def _check_group(self, newblk, blk):
        # group names.
        self.assertEqual(len(newblk.grpnames), len(blk.grpnames))
        for igrp in range(len(blk.grpnames)):
            self.assertEqual(newblk.grpnames[igrp], blk.grpnames[igrp])

    def _check_bc(self, newblk, blk):
        from ..boundcond import interface
        self.assertTrue((newblk.bndfcs == blk.bndfcs).all())
        self.assertEqual(len(newblk.bclist), len(blk.bclist))
        for ibc in range(len(newblk.bclist)):
            newbc = newblk.bclist[ibc]
            bc = blk.bclist[ibc]
            self.assertFalse(isinstance(newbc, interface))
            self.assertFalse(isinstance(bc, interface))
            # meta data.
            self.assertEqual(newbc.sern, bc.sern)
            self.assertEqual(newbc.name, bc.name)
            self.assertNotEqual(newbc.blk, bc.blk)
            self.assertEqual(newbc.blkn, bc.blkn)
            self.assertTrue(newbc.svr == None)
            # faces.
            self.assertTrue((newbc.facn[:,:2] == bc.facn[:,:2]).all())
            # values.
            self.assertEqual(newbc.value.shape[1], bc.value.shape[1])
            if newbc.value.shape[1] > 0:
                self.assertTrue((newbc.value == bc.value).all())

    def _check_array_shape(self, newblk, blk):
        self.assertEqual(newblk.bndfcs.shape, blk.bndfcs.shape)
        # metrics.
        self.assertEqual(newblk.ndcrd.shape, blk.ndcrd.shape)
        self.assertEqual(newblk.fccnd.shape, blk.fccnd.shape)
        self.assertEqual(newblk.fcnml.shape, blk.fcnml.shape)
        self.assertEqual(newblk.fcara.shape, blk.fcara.shape)
        self.assertEqual(newblk.clcnd.shape, blk.clcnd.shape)
        self.assertEqual(newblk.clvol.shape, blk.clvol.shape)
        # type.
        self.assertEqual(newblk.fctpn.shape, blk.fctpn.shape)
        self.assertEqual(newblk.cltpn.shape, blk.cltpn.shape)
        self.assertEqual(newblk.clgrp.shape, blk.clgrp.shape)
        # connectivity.
        self.assertEqual(newblk.fcnds.shape, blk.fcnds.shape)
        self.assertEqual(newblk.fccls.shape, blk.fccls.shape)
        self.assertEqual(newblk.clnds.shape, blk.clnds.shape)
        self.assertEqual(newblk.clfcs.shape, blk.clfcs.shape)
        # ghost metrics.
        self.assertEqual(newblk.gstndcrd.shape, blk.gstndcrd.shape)
        self.assertEqual(newblk.gstfccnd.shape, blk.gstfccnd.shape)
        self.assertEqual(newblk.gstfcnml.shape, blk.gstfcnml.shape)
        self.assertEqual(newblk.gstfcara.shape, blk.gstfcara.shape)
        self.assertEqual(newblk.gstclcnd.shape, blk.gstclcnd.shape)
        self.assertEqual(newblk.gstclvol.shape, blk.gstclvol.shape)
        # ghost type.
        self.assertEqual(newblk.gstfctpn.shape, blk.gstfctpn.shape)
        self.assertEqual(newblk.gstcltpn.shape, blk.gstcltpn.shape)
        self.assertEqual(newblk.gstclgrp.shape, blk.gstclgrp.shape)
        # ghost connectivity.
        self.assertEqual(newblk.gstfcnds.shape, blk.gstfcnds.shape)
        self.assertEqual(newblk.gstfccls.shape, blk.gstfccls.shape)
        self.assertEqual(newblk.gstclnds.shape, blk.gstclnds.shape)
        self.assertEqual(newblk.gstclfcs.shape, blk.gstclfcs.shape)
        # shared metrics.
        self.assertEqual(newblk.shndcrd.shape, blk.shndcrd.shape)
        self.assertEqual(newblk.shfccnd.shape, blk.shfccnd.shape)
        self.assertEqual(newblk.shfcnml.shape, blk.shfcnml.shape)
        self.assertEqual(newblk.shfcara.shape, blk.shfcara.shape)
        self.assertEqual(newblk.shclcnd.shape, blk.shclcnd.shape)
        self.assertEqual(newblk.shclvol.shape, blk.shclvol.shape)
        # shared type.
        self.assertEqual(newblk.shfctpn.shape, blk.shfctpn.shape)
        self.assertEqual(newblk.shcltpn.shape, blk.shcltpn.shape)
        self.assertEqual(newblk.shclgrp.shape, blk.shclgrp.shape)
        # shared connectivity.
        self.assertEqual(newblk.shfcnds.shape, blk.shfcnds.shape)
        self.assertEqual(newblk.shfccls.shape, blk.shfccls.shape)
        self.assertEqual(newblk.shclnds.shape, blk.shclnds.shape)
        self.assertEqual(newblk.shclfcs.shape, blk.shclfcs.shape)

    def _check_array_content(self, newblk, blk):
        self.assertTrue((newblk.bndfcs == blk.bndfcs).all())
        # metrics.
        self.assertTrue((newblk.ndcrd == blk.ndcrd).all())
        self.assertTrue((newblk.fccnd == blk.fccnd).all())
        self.assertTrue((newblk.fcnml == blk.fcnml).all())
        self.assertTrue((newblk.fcara == blk.fcara).all())
        self.assertTrue((newblk.clcnd == blk.clcnd).all())
        self.assertTrue((newblk.clvol == blk.clvol).all())
        # type.
        self.assertTrue((newblk.fctpn == blk.fctpn).all())
        self.assertTrue((newblk.cltpn == blk.cltpn).all())
        self.assertTrue((newblk.clgrp == blk.clgrp).all())
        # connectivity.
        self.assertTrue((newblk.fcnds == blk.fcnds).all())
        self.assertTrue((newblk.fccls == blk.fccls).all())
        self.assertTrue((newblk.clnds == blk.clnds).all())
        self.assertTrue((newblk.clfcs == blk.clfcs).all())
        # ghost metrics.
        self.assertTrue((newblk.gstndcrd == blk.gstndcrd).all())
        self.assertTrue((newblk.gstfccnd == blk.gstfccnd).all())
        self.assertTrue((newblk.gstfcnml == blk.gstfcnml).all())
        self.assertTrue((newblk.gstfcara == blk.gstfcara).all())
        self.assertTrue((newblk.gstclcnd == blk.gstclcnd).all())
        self.assertTrue((newblk.gstclvol == blk.gstclvol).all())
        # ghost type.
        self.assertTrue((newblk.gstfctpn == blk.gstfctpn).all())
        self.assertTrue((newblk.gstcltpn == blk.gstcltpn).all())
        self.assertTrue((newblk.gstclgrp == blk.gstclgrp).all())
        # ghost connectivity.
        self.assertTrue((newblk.gstfcnds == blk.gstfcnds).all())
        self.assertTrue((newblk.gstfccls == blk.gstfccls).all())
        self.assertTrue((newblk.gstclnds == blk.gstclnds).all())
        self.assertTrue((newblk.gstclfcs == blk.gstclfcs).all())
        # shared metrics.
        self.assertTrue((newblk.shndcrd == blk.shndcrd).all())
        self.assertTrue((newblk.shfccnd == blk.shfccnd).all())
        self.assertTrue((newblk.shfcnml == blk.shfcnml).all())
        self.assertTrue((newblk.shfcara == blk.shfcara).all())
        self.assertTrue((newblk.shclcnd == blk.shclcnd).all())
        self.assertTrue((newblk.shclvol == blk.shclvol).all())
        # shared type.
        self.assertTrue((newblk.shfctpn == blk.shfctpn).all())
        self.assertTrue((newblk.shcltpn == blk.shcltpn).all())
        self.assertTrue((newblk.shclgrp == blk.shclgrp).all())
        # shared connectivity.
        self.assertTrue((newblk.shfcnds == blk.shfcnds).all())
        self.assertTrue((newblk.shfccls == blk.shfccls).all())
        self.assertTrue((newblk.shclnds == blk.shclnds).all())
        self.assertTrue((newblk.shclfcs == blk.shclfcs).all())

    def test_dumps(self):
        pickle.dumps(self.blk, 2)

    def test_loads(self):
        data = pickle.dumps(self.blk, 2)
        lblk = pickle.loads(data)
        self._check_shape(lblk, self.blk)
        self._check_group(lblk, self.blk)
        self._check_bc(lblk, self.blk)
        self._check_array_shape(lblk, self.blk)
        self._check_array_content(lblk, self.blk)

class TestUnstructuredBlock2D(TestCase):

    def test_default_constructor(self):
        march.UnstructuredBlock2D()

class TestUnstructuredBlock3D(TestCase):

    def test_default_constructor(self):
        march.UnstructuredBlock3D()

class TestTriangles(TestCase):

    def setUp(self):
        self.msh = march.UnstructuredBlock2D()
        self.msh = march.UnstructuredBlock2D(4, 6, 3, False)
        self.msh.ndcrd[0] = ( 0,  0)
        self.msh.ndcrd[1] = (-1, -1)
        self.msh.ndcrd[2] = ( 1, -1)
        self.msh.ndcrd[3] = ( 0,  1)
        self.msh.cltpn.fill(3)
        self.msh.clnds[0,:4] = (3, 0, 1, 2)
        self.msh.clnds[1,:4] = (3, 0, 2, 3)
        self.msh.clnds[2,:4] = (3, 0, 3, 1)
        self.msh.build_interior()
        self.msh.build_boundary()
        self.msh.build_ghost()

    def test_shape(self):
        self.assertEqual(4, self.msh.nnode)
        self.assertEqual(6, self.msh.nface)
        self.assertEqual(3, self.msh.ncell)
        self.assertEqual((4, 2), self.msh.ndcrd.shape)
        self.assertEqual((6, 2), self.msh.fccnd.shape)

    def test_ce0(self):
        ce = march.ConservationElement2D(self.msh, 0)
        golden = """ConservationElement2D(
  cnd=Vector2D(-1.66533453693773474901e-17,-6.88888888888888883955e-01),
  vol=1.66666666666666674068e+00,
  bces[0]=BasicCE2D(
    cnd=Vector2D(-3.70370370370370405322e-01,-4.81481481481481510265e-01),
    vol=5.00000000000000000000e-01,
    sfcnd[0]=Vector2D(-1.66666666666666685170e-01,0.00000000000000000000e+00),
    sfcnd[1]=Vector2D(-6.66666666666666740682e-01,-5.00000000000000000000e-01),
    sfcnd[2]=Vector2D(-inf,-inf),
    sfcnd[3]=Vector2D(-inf,-inf),
    sfnml[0]=Vector2D(0.00000000000000000000e+00,3.33333333333333370341e-01),
    sfnml[1]=Vector2D(-1.00000000000000000000e+00,6.66666666666666629659e-01),
    sfnml[2]=Vector2D(-inf,-inf),
    sfnml[3]=Vector2D(-inf,-inf)
  ),
  bces[1]=BasicCE2D(
    cnd=Vector2D(0.00000000000000000000e+00,-1.00000000000000000000e+00),
    vol=6.66666666666666740682e-01,
    sfcnd[0]=Vector2D(-5.00000000000000000000e-01,-1.16666666666666674068e+00),
    sfcnd[1]=Vector2D(5.00000000000000000000e-01,-1.16666666666666674068e+00),
    sfcnd[2]=Vector2D(-inf,-inf),
    sfcnd[3]=Vector2D(-inf,-inf),
    sfnml[0]=Vector2D(-3.33333333333333481363e-01,-1.00000000000000000000e+00),
    sfnml[1]=Vector2D(3.33333333333333481363e-01,-1.00000000000000000000e+00),
    sfnml[2]=Vector2D(-inf,-inf),
    sfnml[3]=Vector2D(-inf,-inf)
  ),
  bces[2]=BasicCE2D(
    cnd=Vector2D(3.70370370370370349811e-01,-4.81481481481481510265e-01),
    vol=5.00000000000000000000e-01,
    sfcnd[0]=Vector2D(6.66666666666666629659e-01,-5.00000000000000000000e-01),
    sfcnd[1]=Vector2D(1.66666666666666657415e-01,0.00000000000000000000e+00),
    sfcnd[2]=Vector2D(-inf,-inf),
    sfcnd[3]=Vector2D(-inf,-inf),
    sfnml[0]=Vector2D(1.00000000000000000000e+00,6.66666666666666740682e-01),
    sfnml[1]=Vector2D(-0.00000000000000000000e+00,3.33333333333333314830e-01),
    sfnml[2]=Vector2D(-inf,-inf),
    sfnml[3]=Vector2D(-inf,-inf)
  ),
  bces[3]=BasicCE2D(
    cnd=Vector2D(-inf,-inf),
    vol=-inf,
    sfcnd[0]=Vector2D(-inf,-inf),
    sfcnd[1]=Vector2D(-inf,-inf),
    sfcnd[2]=Vector2D(-inf,-inf),
    sfcnd[3]=Vector2D(-inf,-inf),
    sfnml[0]=Vector2D(-inf,-inf),
    sfnml[1]=Vector2D(-inf,-inf),
    sfnml[2]=Vector2D(-inf,-inf),
    sfnml[3]=Vector2D(-inf,-inf)
  ),
  bces[4]=BasicCE2D(
    cnd=Vector2D(-inf,-inf),
    vol=-inf,
    sfcnd[0]=Vector2D(-inf,-inf),
    sfcnd[1]=Vector2D(-inf,-inf),
    sfcnd[2]=Vector2D(-inf,-inf),
    sfcnd[3]=Vector2D(-inf,-inf),
    sfnml[0]=Vector2D(-inf,-inf),
    sfnml[1]=Vector2D(-inf,-inf),
    sfnml[2]=Vector2D(-inf,-inf),
    sfnml[3]=Vector2D(-inf,-inf)
  ),
  bces[5]=BasicCE2D(
    cnd=Vector2D(-inf,-inf),
    vol=-inf,
    sfcnd[0]=Vector2D(-inf,-inf),
    sfcnd[1]=Vector2D(-inf,-inf),
    sfcnd[2]=Vector2D(-inf,-inf),
    sfcnd[3]=Vector2D(-inf,-inf),
    sfnml[0]=Vector2D(-inf,-inf),
    sfnml[1]=Vector2D(-inf,-inf),
    sfnml[2]=Vector2D(-inf,-inf),
    sfnml[3]=Vector2D(-inf,-inf)
  )
)"""
        self.assertEqual(golden, ce.repr(indent=2, precision=20))

class TestTetrahedra(TestCase):

    def setUp(self):
        self.msh = march.UnstructuredBlock3D(5, 10, 4, False)
        self.msh.ndcrd[0] = ( 0,  0,  0)
        self.msh.ndcrd[1] = (10,  0,  0)
        self.msh.ndcrd[2] = ( 0, 10,  0)
        self.msh.ndcrd[3] = ( 0,  0, 10)
        self.msh.ndcrd[4] = ( 1,  1,  1)
        self.msh.cltpn.fill(5)
        self.msh.clnds[0,:5] = (4, 0, 1, 2, 4)
        self.msh.clnds[1,:5] = (4, 0, 2, 3, 4)
        self.msh.clnds[2,:5] = (4, 0, 3, 1, 4)
        self.msh.clnds[3,:5] = (4, 1, 2, 3, 4)
        self.msh.build_interior()
        self.msh.build_boundary()
        self.msh.build_ghost()

    def test_shape(self):
        self.assertEqual(5, self.msh.nnode)
        self.assertEqual(10, self.msh.nface)
        self.assertEqual(4, self.msh.ncell)
        self.assertEqual((5, 3), self.msh.ndcrd.shape)
        self.assertEqual((10, 3), self.msh.fccnd.shape)

    def test_ce0(self):
        ce = march.ConservationElement3D(self.msh, 0)
        golden = """ConservationElement3D(
  cnd=Vector3D(2.99999999999999955591e+00,2.99999999999999955591e+00,6.69642857142856984254e-01),
  vol=5.83333333333333428072e+01,
  bces[0]=BasicCE3D(
    cnd=Vector3D(3.18749999999999955591e+00,3.18749999999999955591e+00,0.00000000000000000000e+00),
    vol=8.33333333333333392545e+00,
    sfcnd[0]=Vector3D(9.16666666666666407615e-01,4.25000000000000000000e+00,-8.33333333333333287074e-02),
    sfcnd[1]=Vector3D(4.25000000000000000000e+00,4.25000000000000000000e+00,-8.33333333333333287074e-02),
    sfcnd[2]=Vector3D(4.25000000000000000000e+00,9.16666666666666629659e-01,-8.33333333333333287074e-02),
    sfcnd[3]=Vector3D(-inf,-inf,-inf),
    sfnml[0]=Vector3D(-1.25000000000000000000e+00,0.00000000000000000000e+00,-1.37499999999999946709e+01),
    sfnml[1]=Vector3D(1.25000000000000000000e+00,1.25000000000000000000e+00,-2.25000000000000035527e+01),
    sfnml[2]=Vector3D(0.00000000000000000000e+00,-1.25000000000000000000e+00,-1.37500000000000000000e+01),
    sfnml[3]=Vector3D(-inf,-inf,-inf)
  ),
  bces[1]=BasicCE3D(
    cnd=Vector3D(3.43749999999999911182e+00,6.25000000000000000000e-01,6.25000000000000000000e-01),
    vol=8.33333333333333392545e+00,
    sfcnd[0]=Vector3D(4.24999999999999911182e+00,8.33333333333333287074e-02,9.16666666666666629659e-01),
    sfcnd[1]=Vector3D(4.58333333333333303727e+00,4.16666666666666685170e-01,1.25000000000000000000e+00),
    sfcnd[2]=Vector3D(1.24999999999999955591e+00,4.16666666666666685170e-01,1.25000000000000000000e+00),
    sfcnd[3]=Vector3D(-inf,-inf,-inf),
    sfnml[0]=Vector3D(0.00000000000000000000e+00,-1.37500000000000000000e+01,1.25000000000000000000e+00),
    sfnml[1]=Vector3D(1.25000000000000000000e+00,8.75000000000000000000e+00,2.50000000000000088818e+00),
    sfnml[2]=Vector3D(-1.25000000000000000000e+00,4.44089209850062616169e-16,1.24999999999999933387e+00),
    sfnml[3]=Vector3D(-inf,-inf,-inf)
  ),
  bces[2]=BasicCE3D(
    cnd=Vector3D(6.24999999999999888978e-01,3.43750000000000000000e+00,6.25000000000000111022e-01),
    vol=8.33333333333333214910e+00,
    sfcnd[0]=Vector3D(4.16666666666666685170e-01,1.25000000000000000000e+00,1.25000000000000000000e+00),
    sfcnd[1]=Vector3D(4.16666666666666685170e-01,4.58333333333333303727e+00,1.25000000000000000000e+00),
    sfcnd[2]=Vector3D(8.33333333333333287074e-02,4.25000000000000000000e+00,9.16666666666666629659e-01),
    sfcnd[3]=Vector3D(-inf,-inf,-inf),
    sfnml[0]=Vector3D(0.00000000000000000000e+00,-1.25000000000000000000e+00,1.25000000000000000000e+00),
    sfnml[1]=Vector3D(8.75000000000000000000e+00,1.25000000000000000000e+00,2.50000000000000000000e+00),
    sfnml[2]=Vector3D(-1.37500000000000000000e+01,0.00000000000000000000e+00,1.24999999999999977796e+00),
    sfnml[3]=Vector3D(-inf,-inf,-inf)
  ),
  bces[3]=BasicCE3D(
    cnd=Vector3D(3.43749999999999911182e+00,3.43749999999999911182e+00,8.59374999999999777955e-01),
    vol=3.33333333333333428072e+01,
    sfcnd[0]=Vector3D(4.25000000000000000000e+00,4.25000000000000000000e+00,9.16666666666666629659e-01),
    sfcnd[1]=Vector3D(1.25000000000000000000e+00,4.58333333333333303727e+00,1.25000000000000000000e+00),
    sfcnd[2]=Vector3D(4.58333333333333303727e+00,1.25000000000000000000e+00,1.25000000000000000000e+00),
    sfcnd[3]=Vector3D(-inf,-inf,-inf),
    sfnml[0]=Vector3D(1.37500000000000000000e+01,1.37500000000000000000e+01,2.25000000000000000000e+01),
    sfnml[1]=Vector3D(-8.75000000000000000000e+00,0.00000000000000000000e+00,8.75000000000000000000e+00),
    sfnml[2]=Vector3D(0.00000000000000000000e+00,-8.75000000000000000000e+00,8.75000000000000000000e+00),
    sfnml[3]=Vector3D(-inf,-inf,-inf)
  ),
  bces[4]=BasicCE3D(
    cnd=Vector3D(-inf,-inf,-inf),
    vol=-inf,
    sfcnd[0]=Vector3D(-inf,-inf,-inf),
    sfcnd[1]=Vector3D(-inf,-inf,-inf),
    sfcnd[2]=Vector3D(-inf,-inf,-inf),
    sfcnd[3]=Vector3D(-inf,-inf,-inf),
    sfnml[0]=Vector3D(-inf,-inf,-inf),
    sfnml[1]=Vector3D(-inf,-inf,-inf),
    sfnml[2]=Vector3D(-inf,-inf,-inf),
    sfnml[3]=Vector3D(-inf,-inf,-inf)
  ),
  bces[5]=BasicCE3D(
    cnd=Vector3D(-inf,-inf,-inf),
    vol=-inf,
    sfcnd[0]=Vector3D(-inf,-inf,-inf),
    sfcnd[1]=Vector3D(-inf,-inf,-inf),
    sfcnd[2]=Vector3D(-inf,-inf,-inf),
    sfcnd[3]=Vector3D(-inf,-inf,-inf),
    sfnml[0]=Vector3D(-inf,-inf,-inf),
    sfnml[1]=Vector3D(-inf,-inf,-inf),
    sfnml[2]=Vector3D(-inf,-inf,-inf),
    sfnml[3]=Vector3D(-inf,-inf,-inf)
  )
)"""
        self.assertEqual(golden, ce.repr(indent=2, precision=20))
