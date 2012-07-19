# -*- coding: UTF-8 -*-

import os
from unittest import TestCase
from ...testing import loadfile
from .. import gambit

class WriteTest(TestCase):
    __test__ = False
    blk = None
    str_neu_vtk = None

    def assertLines(self, str1, str2):
        """
        Compare two input strings line by line.
        """
        import sys
        lines1 = str1.splitlines()
        lines2 = str2.splitlines()
        self.assertEqual(len(lines1), len(lines2))
        iswin = sys.platform.startswith('win')
        nlines = len(lines1)
        i = 0
        while i < nlines:
            oline = lines1[i]
            nline = lines2[i]
            # compare only non-float lines.
            if not iswin or 'e' not in oline.lower():
                self.assertEqual(oline, nline)
            i += 1
 
    def test_legacy(self):
        import StringIO
        from .. import vtk
        outf = StringIO.StringIO()
        writer = vtk.VtkLegacyUstGridWriter(self.blk)
        writer.write(outf)
        str_blk_vtk = outf.getvalue()
        # compare new result with old result line by line.
        self.assertLines(self.str_neu_vtk, str_blk_vtk)

"""class TestWriteSingle(WriteTest):
    __test__ = True
    blk = gambit.GambitNeutral(loadfile('sample.neu')).toblock(
        fpdtype='float32')
    str_neu_vtk = loadfile('sample.neu.single.vtk')"""

class TestWriteDouble(WriteTest):
    __test__ = True
    blk = gambit.GambitNeutral(loadfile('sample.neu')).toblock(
        fpdtype='float64')
    str_neu_vtk = loadfile('sample.neu.double.vtk')
