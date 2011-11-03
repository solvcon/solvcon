# -*- coding: UTF-8 -*-

import os
from unittest import TestCase
from solvcon.conf import env

class TestHelper(TestCase):
    def test_info(self):
        import sys
        from cStringIO import StringIO
        from ..helper import info
        stdout = sys.stdout
        sys.stdout = StringIO()
        info('test message')
        self.assertEqual(sys.stdout.getvalue(), 'test message')
        sys.stdout = stdout

    def test_platform(self):
        import sys
        from ..helper import iswin
        if sys.platform.startswith('win'):
            self.assertTrue(iswin())
        else:
            self.assertFalse(iswin())

class TestPrinter(TestCase):
    def test_simple(self):
        from cStringIO import StringIO
        from ..helper import Printer
        stream = StringIO()
        p = Printer(stream)
        p('test message')
        self.assertEqual(stream.getvalue(), 'test message')

    def test_prepost(self):
        from cStringIO import StringIO
        from ..helper import Printer
        stream = StringIO()
        p = Printer(stream, prefix='pre', postfix='post')
        p('test message')
        self.assertEqual(stream.getvalue(), 'pretest messagepost')

    def test_multiple(self):
        from cStringIO import StringIO
        from ..helper import Printer
        stream1 = StringIO()
        stream2 = StringIO()
        p = Printer([stream1, stream2])
        p('test message again')
        self.assertEqual(stream1.getvalue(), 'test message again')
        self.assertEqual(stream2.getvalue(), 'test message again')

    def test_stdout(self):
        import sys
        from cStringIO import StringIO
        from ..helper import Printer
        stdout = sys.stdout
        sys.stdout = StringIO()
        p = Printer('sys.stdout')
        p('test message')
        self.assertEqual(sys.stdout.getvalue(), 'test message')
        sys.stdout = stdout

from ..io.gmsh import GmshIO
sblk = GmshIO().load(os.path.join(env.datadir, 'gmsh_square.msh.gz'))
cblk = GmshIO().load(os.path.join(env.datadir, 'gmsh_cube.msh.gz'))
class TestGmsh(TestCase):
    def testSquare(self):
        from ..helper import Gmsh
        cmds = open(os.path.join(env.datadir, 'gmsh_square.geo')).read()
        cmds = [cmd.strip() for cmd in cmds.strip().split('\n')]
        gmh = Gmsh(cmds)()
        blk = gmh.toblock()
        self.assertEqual(len(sblk.bclist), len(blk.bclist))
    def testCube(self):
        from ..helper import Gmsh
        cmds = open(os.path.join(env.datadir, 'gmsh_cube.geo')).read()
        cmds = [cmd.strip() for cmd in cmds.strip().split('\n')]
        gmh = Gmsh(cmds)()
        blk = gmh.toblock()
        self.assertEqual(len(cblk.bclist), len(blk.bclist))
