# Copyright (c) 2015, Yung-Yu Chen <yyc@solvcon.net>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the SOLVCON nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


from __future__ import absolute_import, division, print_function

import unittest

import numpy as np

from ..mesh import Table


class TestTableCreation(unittest.TestCase):
    def test_nothing(self):
        tbl = Table(0, 1)

    def test_zeros(self):
        tbl = Table(0, 3, creation="zeros")
        self.assertEqual([0, 0, 0], list(tbl.F))

    def test_override(self):
        tbl = Table(1, 2, 4)
        # Make sure "nda" isn't writable.
        with self.assertRaises(AttributeError) as cm:
            tbl._nda = np.arange(12).reshape((3,4))
        self.assertTrue(cm.exception.args[0].startswith(
            "attribute '_nda' of"))
        self.assertTrue(cm.exception.args[0].endswith(
            "objects is not writable"))
        # Make sure the addresses remains the same.
        tbl._nda[...] = np.arange(12).reshape((3,4))
        self.assertEqual(tbl._bodyaddr,
                         tbl._ghostaddr + tbl.itemsize * tbl.offset)


class TestTableParts(unittest.TestCase):

    def test_setter(self):
        tbl = Table(4, 8)
        tbl.F = np.arange(12)
        tbl.G = np.arange(4)
        tbl.B = np.arange(8)

    def test_no_setting_property(self):
        tbl = Table(4, 8)
        with self.assertRaisesRegexp(AttributeError, "can't set attribute"):
            tbl._ghostpart = np.arange(4)
        with self.assertRaisesRegexp(AttributeError, "can't set attribute"):
            tbl._bodypart = np.arange(8)

    def test_1d(self):
        tbl = Table(4, 8)
        tbl.F = np.arange(tbl.size, dtype=tbl.dtype).reshape(tbl.shape)
        self.assertEqual(list(range(12)), list(tbl.F))
        self.assertEqual([3,2,1,0], list(tbl.G))
        self.assertEqual([3,2,1,0], list(tbl._ghostpart))
        self.assertEqual(list(range(4,12)), list(tbl.B))
        self.assertEqual(list(range(4,12)), list(tbl._bodypart))

    def test_2d(self):
        tbl = Table(1, 2, 4)
        tbl.F = np.arange(tbl.size, dtype=tbl.dtype).reshape(tbl.shape)
        # Make sure the above writing doesn't override the memory holder.
        self.assertEqual(tbl._bodyaddr,
                         tbl._ghostaddr + tbl.itemsize * tbl.offset)
        # Check for value.
        self.assertEqual((1,4), tbl.G.shape)
        self.assertEqual((1,4), tbl._ghostpart.shape)
        self.assertEqual(list(range(4)), list(tbl.G.ravel()))
        self.assertEqual(list(range(4)), list(tbl._ghostpart.ravel()))
        self.assertEqual((2,4), tbl.B.shape)
        self.assertEqual((2,4), tbl._bodypart.shape)
        self.assertEqual(list(range(4,12)), list(tbl.B.ravel()))
        self.assertEqual(list(range(4,12)), list(tbl._bodypart.ravel()))

    def test_3d(self):
        tbl = Table(1, 2, 4, 5)
        tbl.F = np.arange(tbl.size, dtype=tbl.dtype).reshape(tbl.shape)
        # Make sure the above writing doesn't override the memory holder.
        self.assertEqual(tbl._bodyaddr,
                         tbl._ghostaddr + tbl.itemsize * tbl.offset)
        # Check for value.
        self.assertEqual((1,4,5), tbl.G.shape)
        self.assertEqual((1,4,5), tbl._ghostpart.shape)
        self.assertEqual(list(range(4*5)), list(tbl.G.ravel()))
        self.assertEqual(list(range(4*5)), list(tbl._ghostpart.ravel()))
        self.assertEqual((2,4,5), tbl.B.shape)
        self.assertEqual((2,4,5), tbl._bodypart.shape)
        self.assertEqual(list(range(4*5,3*4*5)), list(tbl.B.ravel()))
        self.assertEqual(list(range(4*5,3*4*5)), list(tbl._bodypart.ravel()))

# vim: set fenc=utf8 ff=unix nobomb ai et sw=4 ts=4 tw=79:
