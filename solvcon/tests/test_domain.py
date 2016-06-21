# -*- coding: UTF-8 -*-


from __future__ import absolute_import, division, print_function


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

    def test_splitted_ncell_by_clnds(self):
        """
        The number of cell of the block and sub-block checker.

        clnds: cell nodes (vertices).
            [[total number N of node share this cell,
              node-1, node-2, ......, node-N]......]
            The length of this array is equal to ncell,
            and should be the sum of the ncell length of all sub-blocks.
        """
        iblk = 0
        length_sum = 0
        for blk in self.dom:
            length_sum += len(blk.clnds)
            iblk += 1
        self.assertEqual(length_sum, self.dom.blk.ncell)

    def test_splitted_ncell_by_clfcs(self):
        """
        The number of cell of the block and sub-block checker.

        clfcs: cell faces.
            [[total number N of face share this cell,
              face-1, face-2, ......, face-N]......]
            The length of this array is equal to ncell,
            and should be the sum of the ncell length of all sub-blocks.
        """
        iblk = 0
        length_sum = 0
        for blk in self.dom:
            length_sum += len(blk.clfcs)
            iblk += 1
        self.assertEqual(length_sum, self.dom.blk.ncell)

    def test_splitted_nodes(self):
        """
        Max node number checker.

        Is the max node of each sub-block index - 1 equal to
        the number of node of each sub-block?
        """
        iblk = 0
        for blk in self.dom:
            self.assertEqual(blk.clnds[:,1:].max()+1, blk.nnode)
            # next sub-block
            iblk += 1

    def test_splitted_faces(self):
        iblk = 0
        for blk in self.dom:
            self.assertEqual(blk.clfcs[:,1:].max()+1, blk.nface)
            iblk += 1

    def test_splitted_neiblk(self):
        """
        edgecut should be double of ncut because nblk is 2 in 2D.

        One cut gives 2 sub-blocks. The number of the cells along
        the cut edges/faces are double of ncut when nblk is 2 in 2D.

        Meanings of fccls columns:
        * 0 - belong
        * 1 - neibor
        * 2 - neighboring block. Given -1 for no such block or int index of
              the block.
        * 3 - neighbor block cell - cell index of the sub-block.
        """
        ncut = 0
        for blk in self.dom:
            ncut += (blk.fccls[:,2]!=-1).sum()
        self.assertEqual(ncut, self.dom.edgecut*2)

    def test_splitted_neibcl(self):
        """
        If there is no neighboring block, there should be no cell of neibor
        block.

        Meanings of fccls columns:
        * 0 - belong
        * 1 - neibor
        * 2 - neighboring block. Given -1 for no such block or int index of
              the block.
        * 3 - neighbor block cell - cell index of the sub-block.
        """
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
