# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2012 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.


from __future__ import absolute_import, division, print_function


import unittest
from unittest import TestCase

import numpy as np

from .. import dependency
from .. import block_legacy
from ..testing import get_blk_from_oblique_neu, get_blk_from_sample_neu

class TestTableDescriptor(TestCase):
    def test_contents(self):
        blk = block_legacy.LegacyBlock()
        self.assertEqual(list(sorted(blk.TABLE_NAMES)),
                         list(sorted(blk._tables.keys())))
        self.assertEqual(list(sorted(blk.TABLE_NAMES)),
                         list(sorted(blk._shared_arrays.keys())))
        self.assertEqual(list(sorted(blk.TABLE_NAMES)),
                         list(sorted(blk._body_arrays.keys())))
        self.assertEqual(list(sorted(blk.TABLE_NAMES)),
                         list(sorted(blk._ghost_arrays.keys())))

    def test_type(self):
        blk = block_legacy.LegacyBlock()
        for tname in blk.TABLE_NAMES:
            name = tname
            # It shall pass, although causes sanity check failure.
            oldarr = getattr(blk, name)
            setattr(blk, name, np.empty_like(oldarr))
            with self.assertRaisesRegex(
                AttributeError, '%s array mismatch: body'%name):
                blk.check_sanity()
            setattr(blk, name, oldarr) # Put it back for the next test in loop.
            # It shall not be set to a incorrect type.
            for wrong_typed in (None, 'string', 1, 3.5, list(oldarr)):
                with self.assertRaisesRegex(
                    TypeError, 'only Table and ndarray are acceptable'):
                    setattr(blk, name, wrong_typed)

    def test_unacceptable_name(self):
        class MyBlock(block_legacy.LegacyBlock):
            invalid = block_legacy._TableDescriptor('invalid', '', '_invalid_arrays')
        blk = MyBlock()
        # Make sure the collector dict for the "invalid" item is absent.
        self.assertFalse(hasattr(MyBlock, '_invalid_arrays'))
        # The "invalid" descriptor shouldn't work.
        with self.assertRaisesRegex(
            AttributeError, '"invalid" is not in Block.TABLE_NAME'):
            blk.invalid = np.empty(10)
        # No collector is created.
        self.assertFalse(hasattr(MyBlock, '_invalid_arrays'))

class TestCreation(TestCase):
    def test_insanity(self):
        # build a simple 2D triangle with 4 subtriangles.
        blk = block_legacy.LegacyBlock(ndim=2, nnode=4, nface=6, ncell=3, nbound=3)
        blk.ndcrd[0,:] = (0,0)
        blk.ndcrd[1,:] = (-1,-1)
        blk.ndcrd[2,:] = (1,-1)
        blk.ndcrd[3,:] = (0,1)
        blk.cltpn[:] = 3
        blk.clnds[0,:4] = (3, 0,1,2)
        blk.clnds[1,:4] = (3, 0,2,3)
        blk.clnds[2,:4] = (3, 0,3,1)
        blk.build_interior()
        # reset an array
        blk.check_sanity()
        blk.ndcrd = blk.ndcrd.copy()
        with self.assertRaisesRegex(AttributeError,
                                    "ndcrd array mismatch: body"):
            blk.check_sanity()
