# -*- coding: UTF-8 -*-

import os
from unittest import TestCase
from ..testing import loadfile
from ..io import gambit

def test_load():
    from ..boundcond import BC

class TestBc(TestCase):
    #__test__ = False    # temporarily turned off.
    blk = gambit.GambitNeutral(loadfile('sample.neu')).toblock()

    def test_print_vanilla(self):
        from ..boundcond import BC
        obj = BC()
        self.assertEqual(str(obj), '[BC#None "None": 0 faces with 0 values]')

    def test_print_sern(self):
        from ..boundcond import BC
        obj = BC()
        obj.sern = 5
        self.assertEqual(str(obj), '[BC#5 "None": 0 faces with 0 values]')

    def test_certain_bct(self):
        from ..boundcond import bctregy
        # check that the existance of the most generic abstract BC types.
        for key in 'BC', 'unspecified':
            self.assert_(key in bctregy)

    def test_comprehension(self):
        from numpy import concatenate
        # copy data from block.
        allfcs = concatenate([bc.facn[:,0] for bc in self.blk.bclist])
        allfcs.sort()
        bndfcs = self.blk.bndfcs[:,0].copy()
        bndfcs.sort()
        # test for name.
        names = sorted([bc.__class__.__name__ for bc in self.blk.bclist])
        self.assertEqual(names[0], 'BC')
        self.assertEqual(names[1], 'unspecified')
        # loop test.
        nbound = self.blk.nbound
        ibnd = 0
        while ibnd < nbound:
            self.assertEqual(allfcs[ibnd], bndfcs[ibnd])
            ibnd += 1

    def test_bndfcs(self):
        for bc in self.blk.bclist:
            for bfc, idx, ridx in bc.facn:
                self.assertEqual(bfc, bc.blk.bndfcs[idx,0])

    def test_fp(self):
        from ..dependency import str_of
        from ..conf import env
        for bc in self.blk.bclist:
            self.assertEqual(bc.fpdtype, env.fpdtype)
            self.assertEqual(bc.fpdtypestr, str_of(env.fpdtype))
