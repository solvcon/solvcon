# -*- coding: UTF-8 -*-

from unittest import TestCase

class TestNdarray(TestCase):
    def test_negative_stides(self):
        from numpy import arange
        arr = arange(40, dtype='int32').reshape(10,4)
        pos = arr[5:,:]
        neg = arr[4::-1,:]
        # test for proper inversion.
        i = j = 0
        while i < neg.shape[0]:
            while j < neg.shape[1]:
                self.assertEqual(neg[i,j], arr[4-i,j])
                j += 1
            i += 1
        # test that they share the same memory.
        self.assert_((neg >= 0).all())
        self.assert_((arr >= 0).all())
        tval = -1
        neg[1,2] = tval
        self.assertEqual(arr[3,2], tval)
