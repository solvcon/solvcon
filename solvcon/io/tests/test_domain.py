# -*- coding: UTF-8 -*-

from unittest import TestCase

def get_sample_neu():
    """
    Read data from oblique.neu file and convert it into Block.
    """
    from ...testing import loadfile
    from ..gambit import GambitNeutral
    blk = GambitNeutral(loadfile('sample.neu')).toblock()
    for bc in blk.bclist:
        if bc.name == 'unspecified':
            bc.name = 'unspec_orig'
    return blk

class CheckDomainIO(TestCase):
    def _check_domain_shape(self, don, doo):
        self.assertEqual(don.edgecut, doo.edgecut)
    def _check_domain_array(self, don, doo):
        self.assertTrue((don.part == doo.part).all())
        self.assertTrue((don.shapes == doo.shapes).all())
        self.assertTrue((don.ifparr == doo.ifparr).all())
        self.assertTrue((don.mappers[0] == doo.mappers[0]).all())
        self.assertTrue((don.mappers[1] == doo.mappers[1]).all())
        self.assertTrue((don.mappers[2] == doo.mappers[2]).all())
        for it in range(len(doo)):
            try:
                self.assertTrue((don.idxinfo[it][0] ==
                    doo.idxinfo[it][0]).all())
                self.assertTrue((don.idxinfo[it][1] ==
                    doo.idxinfo[it][1]).all())
                self.assertTrue((don.idxinfo[it][2] ==
                    doo.idxinfo[it][2]).all())
            except StandardError as e:
                msgs = list(e.args)
                msgs.append('%d-th block' % it)
                e.args = tuple(msgs)
                raise

    def _check_block_shape(self, newblk, blk):
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
    def _check_block_group(self, newblk, blk):
        # group names.
        self.assertEqual(len(newblk.grpnames), len(blk.grpnames))
        for igrp in range(len(blk.grpnames)):
            self.assertEqual(newblk.grpnames[igrp], blk.grpnames[igrp])
    def _check_block_bc(self, newblk, blk):
        from ...boundcond import interface
        self.assertTrue((newblk.bndfcs == blk.bndfcs).all())
        self.assertEqual(len(newblk.bclist), len(blk.bclist))
        for ibc in range(len(newblk.bclist)):
            try:
                newbc = newblk.bclist[ibc]
                bc = blk.bclist[ibc]
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
                # for interface.
                if isinstance(bc, interface):
                    self.assertTrue(isinstance(newbc, interface))
                    self.assertEqual(newbc.rblkn, bc.rblkn)
                    self.assertTrue((newbc.rblkinfo == bc.rblkinfo).all())
                    self.assertTrue((newbc.rclp == bc.rclp).all())
            except StandardError as e:
                msgs = list(e.args)
                msgs.append('%d-th BC' % ibc)
                e.args = tuple(msgs)
                raise
    def _check_block_array(self, newblk, blk):
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

class TestReloadTrivial(CheckDomainIO):
    def test_whole_domain(self):
        from tempfile import mkdtemp
        from shutil import rmtree
        from ...domain import Collective
        from ..domain import DomainIO
        npart = 3
        # create original domain.
        blk = get_sample_neu()
        doo = Collective(blk=blk)
        doo.split(npart)
        dio = DomainIO(compressor='gz', fmt='TrivialDomainFormat')
        # save and reload to new domain.
        dirname = mkdtemp()
        dio.save(dom=doo, dirname=dirname)
        don = dio.load(dirname=dirname, with_split=True)
        rmtree(dirname)
        # check domain.
        self._check_domain_shape(don, doo)
        self._check_domain_array(don, doo)
        # check whole block.
        self._check_block_shape(don.blk, doo.blk)
        self._check_block_group(don.blk, doo.blk)
        self._check_block_bc(don.blk, doo.blk)
        self._check_block_array(don.blk, doo.blk)
        # check split blocks.
        for iblk in range(npart):
            try:
                self._check_block_shape(don[iblk], doo[iblk])
                self._check_block_group(don[iblk], doo[iblk])
                self._check_block_bc(don[iblk], doo[iblk])
                self._check_block_array(don[iblk], doo[iblk])
            except StandardError as e:
                msgs = list(e.args)
                msgs.append('%d-th block' % iblk)
                e.args = tuple(msgs)
                raise

    def test_single_block(self):
        from tempfile import mkdtemp
        from shutil import rmtree
        from ...domain import Collective
        from ..domain import DomainIO
        npart = 3
        # create original domain.
        blk = get_sample_neu()
        doo = Collective(blk=blk)
        doo.split(npart)
        dio = DomainIO(compressor='gz', fmt='TrivialDomainFormat')
        # save and reload to new domain.
        dirname = mkdtemp()
        dio.save(dom=doo, dirname=dirname)
        don = dio.load_block(dirname=dirname)
        # check whole block.
        blk = dio.load_block(dirname=dirname, blkid=None, bcmapper=None)
        self._check_block_shape(blk, doo.blk)
        self._check_block_group(blk, doo.blk)
        self._check_block_bc(blk, doo.blk)
        self._check_block_array(blk, doo.blk)
        # check split blocks.
        for iblk in range(npart):
            try:
                blk = dio.load_block(dirname=dirname, blkid=iblk,
                    bcmapper=None)
                self._check_block_shape(blk, doo[iblk])
                self._check_block_group(blk, doo[iblk])
                self._check_block_bc(blk, doo[iblk])
                self._check_block_array(blk, doo[iblk])
            except StandardError as e:
                msgs = list(e.args)
                msgs.append('%d-th block' % iblk)
                e.args = tuple(msgs)
                raise
        # finalize.
        rmtree(dirname)

    def test_limited_domain(self):
        from tempfile import mkdtemp
        from shutil import rmtree
        from ...domain import Collective
        from ..domain import DomainIO
        npart = 3
        # create original domain.
        blk = get_sample_neu()
        doo = Collective(blk=blk)
        doo.split(npart)
        dio = DomainIO(compressor='gz', fmt='TrivialDomainFormat')
        # save and reload to new domain.
        dirname = mkdtemp()
        dio.save(dom=doo, dirname=dirname)
        don = dio.load(dirname=dirname)
        rmtree(dirname)
        # check domain.
        self._check_domain_shape(don, doo)
        self._check_domain_array(don, doo)
        # check whole block.
        self._check_block_shape(don.blk, doo.blk)
        self._check_block_group(don.blk, doo.blk)
        self._check_block_bc(don.blk, doo.blk)
        self._check_block_array(don.blk, doo.blk)
        # check split blocks.
        self.assertEqual(len(don), 0)

class TestReloadIncenter(CheckDomainIO):
    def test_whole_domain(self):
        from tempfile import mkdtemp
        from shutil import rmtree
        from ...domain import Collective
        from ..domain import DomainIO
        npart = 3
        # create original domain.
        blk = get_sample_neu()
        doo = Collective(blk=blk)
        doo.split(npart)
        dio = DomainIO(compressor='gz', fmt='IncenterDomainFormat')
        # save and reload to new domain.
        dirname = mkdtemp()
        dio.save(dom=doo, dirname=dirname)
        don = dio.load(dirname=dirname, with_split=True)
        rmtree(dirname)
        # check domain.
        self._check_domain_shape(don, doo)
        self._check_domain_array(don, doo)
        # check whole block.
        self._check_block_shape(don.blk, doo.blk)
        self._check_block_group(don.blk, doo.blk)
        self._check_block_bc(don.blk, doo.blk)
        self._check_block_array(don.blk, doo.blk)
        # check split blocks.
        for iblk in range(npart):
            try:
                self._check_block_shape(don[iblk], doo[iblk])
                self._check_block_group(don[iblk], doo[iblk])
                self._check_block_bc(don[iblk], doo[iblk])
                self._check_block_array(don[iblk], doo[iblk])
            except StandardError as e:
                msgs = list(e.args)
                msgs.append('%d-th block' % iblk)
                e.args = tuple(msgs)
                raise

    def test_single_block(self):
        from tempfile import mkdtemp
        from shutil import rmtree
        from ...domain import Collective
        from ..domain import DomainIO
        npart = 3
        # create original domain.
        blk = get_sample_neu()
        doo = Collective(blk=blk)
        doo.split(npart)
        dio = DomainIO(compressor='gz', fmt='IncenterDomainFormat')
        # save and reload to new domain.
        dirname = mkdtemp()
        dio.save(dom=doo, dirname=dirname)
        don = dio.load_block(dirname=dirname)
        # check whole block.
        blk = dio.load_block(dirname=dirname, blkid=None, bcmapper=None)
        self._check_block_shape(blk, doo.blk)
        self._check_block_group(blk, doo.blk)
        self._check_block_bc(blk, doo.blk)
        self._check_block_array(blk, doo.blk)
        # check split blocks.
        for iblk in range(npart):
            try:
                blk = dio.load_block(dirname=dirname, blkid=iblk,
                    bcmapper=None)
                self._check_block_shape(blk, doo[iblk])
                self._check_block_group(blk, doo[iblk])
                self._check_block_bc(blk, doo[iblk])
                self._check_block_array(blk, doo[iblk])
            except StandardError as e:
                msgs = list(e.args)
                msgs.append('%d-th block' % iblk)
                e.args = tuple(msgs)
                raise
        # finalize.
        rmtree(dirname)

    def test_limited_domain(self):
        from tempfile import mkdtemp
        from shutil import rmtree
        from ...domain import Collective
        from ..domain import DomainIO
        npart = 3
        # create original domain.
        blk = get_sample_neu()
        doo = Collective(blk=blk)
        doo.split(npart)
        dio = DomainIO(compressor='gz', fmt='IncenterDomainFormat')
        # save and reload to new domain.
        dirname = mkdtemp()
        dio.save(dom=doo, dirname=dirname)
        don = dio.load(dirname=dirname)
        rmtree(dirname)
        # check domain.
        self._check_domain_shape(don, doo)
        self._check_domain_array(don, doo)
        # check whole block.
        self._check_block_shape(don.blk, doo.blk)
        self._check_block_group(don.blk, doo.blk)
        self._check_block_bc(don.blk, doo.blk)
        self._check_block_array(don.blk, doo.blk)
        # check split blocks.
        self.assertEqual(len(don), 0)
