# -*- coding: UTF-8 -*-

from unittest import TestCase

class VtkXmlTest(TestCase):
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
 
    """def test_xml_single(self):
        import StringIO
        from ...testing import loadfile
        from .. import gambit
        from .. import vtkxml
        blk = gambit.GambitNeutral(loadfile('sample.neu')).toblock(
            fpdtype='float64')
        wtr = vtkxml.VtkXmlUstGridWriter(blk, fpdtype='float32')
        outf = StringIO.StringIO()
        wtr.write(outf)
        dat = outf.getvalue()
        self.assertNotEqual(dat.find('Float32'), -1)
        self.assertEqual(dat.find('Float64'), -1)"""

    def test_xml_double(self):
        import StringIO
        from ...testing import loadfile
        from .. import gambit
        from .. import vtkxml
        blk = gambit.GambitNeutral(loadfile('sample.neu')).toblock(
            fpdtype='float64')
        wtr = vtkxml.VtkXmlUstGridWriter(blk, fpdtype='float64')
        outf = StringIO.StringIO()
        wtr.write(outf)
        dat = outf.getvalue()
        self.assertNotEqual(dat.find('Float64'), -1)
        self.assertEqual(dat.find('Float32'), -1)

    def test_xml_appended(self):
        import StringIO
        from ...testing import loadfile
        from .. import gambit
        from .. import vtkxml
        blk = gambit.GambitNeutral(loadfile('sample.neu')).toblock(
            fpdtype='float64')
        wtr = vtkxml.VtkXmlUstGridWriter(blk, appended=True)
        outf = StringIO.StringIO()
        wtr.write(outf)
        dat = outf.getvalue()
        self.assertEqual(dat.find('ascii'), -1)
        self.assertEqual(dat.find('binary'), -1)

    def test_xml_binary(self):
        import StringIO
        from ...testing import loadfile
        from .. import gambit
        from .. import vtkxml
        blk = gambit.GambitNeutral(loadfile('sample.neu')).toblock(
            fpdtype='float64')
        wtr = vtkxml.VtkXmlUstGridWriter(blk, appended=False, binary=True)
        outf = StringIO.StringIO()
        wtr.write(outf)
        dat = outf.getvalue()
        self.assertEqual(dat.find('ascii'), -1)
        self.assertEqual(dat.find('appended'), -1)

    def test_xml_ascii(self):
        import os
        # numpy.tofile() do not work on StringIO.
        from tempfile import mkstemp
        from ...testing import loadfile
        from .. import gambit
        from .. import vtkxml
        blk = gambit.GambitNeutral(loadfile('sample.neu')).toblock(
            fpdtype='float64')
        wtr = vtkxml.VtkXmlUstGridWriter(blk, appended=False, binary=False)
        outfd, ofn = mkstemp(text=True)
        outf = os.fdopen(outfd, 'w')
        wtr.write(outf)
        outf.close()
        outf = open(ofn)
        dat = outf.read()
        outf.close()
        self.assertEqual(dat.find('binary'), -1)
        self.assertEqual(dat.find('appended'), -1)
