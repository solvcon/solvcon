# -*- coding: UTF-8 -*-

import os
from unittest import TestCase
from ...testing import openfile, loadfile
from .. import gambit

class NeutralTest(TestCase):
    __test__ = False
    neu = None
    blk = None
    round_to = 100

    def test_load(self):
        """
        Test reading functionality.
        """
        neu = self.neu
        self.assert_(str(neu) == 
            '[Neutral (Example): 60 nodes, 116 elements, 1 groups, 2 bcs]')
        self.assert_(str(neu.grps[0]) == 
            '[Group #1(fluid): 116 elements]')
        self.assert_(str(neu.bcs[0]) == 
            '[BC "element_side.1": 14 entries with 0 values]')
        self.assert_(str(neu.bcs[1]) == 
            '[BC "node.2": 16 entries with 0 values]')

    def test_blk_volume(self):
        """
        Test the volume calculated in metrics for Block object convert from
        GambitNeutral object.
        """
        blk = self.blk
        self.assertAlmostEqual(blk.clvol.sum()/1000, 1.0, self.round_to)

    def test_blk_bound(self):
        """
        Test for both computed and extracted boundary faces for Block object
        converted from GambitNeutral object.
        """
        from numpy import array
        from ...boundcond import bctregy
        blk = self.blk
        # total number of boundary faces.
        self.assertEqual(blk.nbound, 74)
        # number of boundary faces extracted from file.
        self.assertEqual(blk.bclist[0].__class__, bctregy.BC)
        self.assertEqual(len(blk.bclist[0]), 14)
        self.assertEqual(blk.bclist[1].__class__, bctregy.unspecified)
        self.assertEqual(len(blk.bclist[1]), 60)
        self.assertEqual(blk.bndfcs.shape[0], 74)
        # extracted faces have only one related cell.
        exfcs = blk.bclist[0].facn[:,0]
        self.assert_((blk.fccls[exfcs,1] < 0).all())
        # mutual existance of original cell list and extract cell list.
        ocls = array([3,4,7,8,100,110,115,35,47,16,28,52,34,70], dtype='int32')-1
        ecls = blk.fccls[exfcs,0]
        for icl in ocls:
            self.assert_(icl in ecls)
        for icl in ecls:
            self.assert_(icl in ocls)

"""class TestNeutralSingle(NeutralTest):
    __test__ = True
    neu = gambit.GambitNeutral(loadfile('sample.neu'))
    blk = neu.toblock(fpdtype='float32')
    round_to = 6"""

class TestNeutralDouble(NeutralTest):
    __test__ = True
    neu = gambit.GambitNeutral(loadfile('sample.neu'))
    blk = neu.toblock(fpdtype='float64')
    round_to = 15

"""class TestNeutralReadSingle(NeutralTest):
    __test__ = True
    neu = gambit.GambitNeutral(openfile('sample.neu'))
    blk = neu.toblock(fpdtype='float32')
    round_to = 6"""

class TestNeutralReadDouble(NeutralTest):
    __test__ = True
    neu = gambit.GambitNeutral(openfile('sample.neu'))
    blk = neu.toblock(fpdtype='float64')
    round_to = 15
