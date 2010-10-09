# -*- coding: UTF-8 -*-

from unittest import TestCase

def get_sample_neu():
    """
    Read data from oblique.neu file and convert it into Block.
    """
    from ..testing import loadfile
    from ..io.gambit import GambitNeutral
    blk = GambitNeutral(loadfile('sample.neu')).toblock()
    for bc in blk.bclist:
        if bc.name == 'unspecified':
            bc.name = 'unspec_orig'
    return blk

class TestDomain(TestCase):
    def test_import(self):
        from ..domain import Domain

    def test_create(self):
        from ..domain import Domain
        dom = Domain(None)
        blk = get_sample_neu()
        dom = Domain(blk)
        self.assertTrue(dom.blk is blk)

class TestCollective(TestCase):
    from ..domain import Collective
    nblk = 2
    blk = get_sample_neu()
    dom = Collective(blk=blk)
    dom.split(nblk)

    def test_group(self):
        for blk in self.dom:
            self.assertEqual(blk.grpnames, self.blk.grpnames)

    def test_partition(self):
        self.assertEqual(len(self.dom.idxinfo), self.nblk)

    def test_splitted_nodes(self):
        iblk = 0
        for blk in self.dom:
            self.assertEqual(blk.clnds[:,1:].max()+1, blk.nnode)
            iblk += 1

    def test_splitted_faces(self):
        iblk = 0
        for blk in self.dom:
            self.assertEqual(blk.clfcs[:,1:].max()+1, blk.nface)
            iblk += 1

    def test_splitted_neiblk(self):
        ncut = 0
        for blk in self.dom:
            ncut += (blk.fccls[:,2]!=-1).sum()
        self.assertEqual(ncut, self.dom.edgecut*2)

    def test_splitted_neibcl(self):
        for blk in self.dom:
            slct = (blk.fccls[:,2]==-1)
            self.assertTrue((blk.fccls[slct,3] == -1).all())

    def test_splitted_bnds(self):
        dom = self.dom
        nbndsp = sum([(blk.fccls[:,1]<0).sum() for blk in dom])
        self.assertEqual(
            nbndsp,
            (dom.blk.fccls[:,1]<0).sum() + dom.edgecut*2,
        )

    def test_splitted_bcs(self):
        dom = self.dom
        for bc in dom.blk.bclist:
            nfc = 0
            bcname = bc.name
            for blk in dom:
                for sbc in blk.bclist:
                    if sbc.name == bcname:
                        nfc += len(sbc)
            self.assertEqual(nfc, len(bc))

    def test_splitted_interfaces(self):
        from ..boundcond import bctregy
        interface = bctregy.interface
        dom = self.dom
        nfc = 0
        for blk in dom:
            for sbc in blk.bclist:
                if isinstance(sbc, interface):
                    nfc += len(sbc)
        self.assertEqual(nfc, dom.edgecut*2)

    def test_bcs(self):
        from ..boundcond import bctregy
        for blk in self.dom:
            for bc in blk.bclist:
                self.assertEqual(blk, bc.blk)

    def test_interface_face(self):
        from ..boundcond import bctregy
        for blk in self.dom:
            has_interface = False
            for bc in blk.bclist:
                if isinstance(bc, bctregy.interface):
                    has_interface = True
                    bfcs = bc.facn[:,0]
                    self.assertTrue((blk.fccls[bfcs,1] < 0).all())
            self.assertTrue(has_interface)

    def test_interfaces_count(self):
        from ..boundcond import bctregy
        dom = self.dom
        # count from blocks.
        cnt_blk = 0
        for blk in dom:
            for sbc in blk.bclist:
                if isinstance(sbc, bctregy.interface):
                    cnt_blk += 1
        # compare.
        self.assertEqual(cnt_blk, dom.ifparr.shape[0]*2)

    def nottest_write(self):
        from ..io.vtk import VtkLegacyUstGridWriter
        dom = self.dom
        VtkLegacyUstGridWriter(
            dom.blk, binary=False,
        ).write('test.vtk')
        writers = list()
        iblk = 0
        for blk in dom:
            writers.append(VtkLegacyUstGridWriter(
                blk, binary=False,
            ))
            writers[-1].write('test%d.vtk'%iblk)
            iblk += 1

class TestInterface(TestCase):
    def test_oblique2(self):
        from ..domain import Collective
        from .. import testing
        blk = testing.get_blk_from_oblique_neu()
        dom = Collective(blk=blk)
        dom.split(2)
        iflists = dom.make_iflist_per_block()
    def test_oblique8(self):
        from ..domain import Collective
        from .. import testing
        blk = testing.get_blk_from_oblique_neu()
        dom = Collective(blk=blk)
        dom.split(8)
        iflists = dom.make_iflist_per_block()
    def test_oblique24(self):
        from ..domain import Collective
        from .. import testing
        blk = testing.get_blk_from_oblique_neu()
        dom = Collective(blk=blk)
        dom.split(24)
        iflists = dom.make_iflist_per_block()

    def test_sample2(self):
        from ..domain import Collective
        from .. import testing
        blk = testing.get_blk_from_sample_neu()
        dom = Collective(blk=blk)
        dom.split(2)
        iflists = dom.make_iflist_per_block()
    def test_sample4(self):
        from ..domain import Collective
        from .. import testing
        blk = testing.get_blk_from_sample_neu()
        dom = Collective(blk=blk)
        dom.split(4)
        iflists = dom.make_iflist_per_block()
