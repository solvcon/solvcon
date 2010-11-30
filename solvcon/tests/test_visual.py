# -*- coding: UTF-8 -*-

from unittest import TestCase

class TestVtk(TestCase):
    def test_ust_from_blk(self):
        import sys
        from nose.plugins.skip import SkipTest
        try:
            import vtk
        except ImportError:
            raise SkipTest
        from ..testing import get_blk_from_sample_neu
        from ..visual_vtk import make_ust_from_blk
        ust = make_ust_from_blk(get_blk_from_sample_neu())

class TestVtkOperation(TestCase):
    def test_import(self):
        import sys
        from nose.plugins.skip import SkipTest
        try:
            import vtk
        except ImportError:
            raise SkipTest
        from ..testing import get_blk_from_sample_neu
        from ..visual_vtk import make_ust_from_blk
        ust = make_ust_from_blk(get_blk_from_sample_neu())
