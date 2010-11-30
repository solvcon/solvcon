# -*- coding: UTF-8 -*-

from unittest import TestCase

class TestVtk(TestCase):
    def test_ust_from_blk(self):
        from nose.plugins.skip import SkipTest
        try:
            import vtk
        except ImportError:
            raise SkipTest
        from ..testing import get_blk_from_sample_neu
        from ..visual_vtk import make_ust_from_blk
        ust = make_ust_from_blk(get_blk_from_sample_neu())
    def test_set_celldata(self):
        from nose.plugins.skip import SkipTest
        try:
            import vtk
        except ImportError:
            raise SkipTest
        from numpy import empty
        from ..testing import get_blk_from_sample_neu
        from ..visual_vtk import make_ust_from_blk, set_array
        blk = get_blk_from_sample_neu()
        ust = make_ust_from_blk(blk)
        arr = empty(blk.ncell, dtype='float32')
        set_array(arr, 'test', 'float32', ust)

class TestVtkOperation(TestCase):
    def test_import(self):
        from nose.plugins.skip import SkipTest
        try:
            import vtk
        except ImportError:
            raise SkipTest
        # it is the test.
        from ..visual_vtk import VtkOperation, Vop
