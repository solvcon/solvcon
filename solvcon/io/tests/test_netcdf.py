# -*- coding: UTF-8 -*-

from unittest import TestCase

class TestRead(TestCase):
    import os
    from ...conf import env
    testfn = [env.datadir] + ['ref_nctest_classic.nc']
    testfn = os.path.join(*testfn)

    def test_open(self):
        from ..netcdf import NetCDF
        nc = NetCDF()
        nc.open_file(self.testfn)
        nc.close_file()
        nc = NetCDF(self.testfn)
        nc.close_file()

    def test_dim(self):
        from ..netcdf import NetCDF
        nc = NetCDF(self.testfn)
        self.assertEqual(nc.get_dim('ii'), 4)
        self.assertEqual(nc.get_dim('jj'), 3)
        self.assertEqual(nc.get_dim('kk'), 3)
        self.assertEqual(nc.get_dim('i1'), 5)
        self.assertEqual(nc.get_dim('i2'), 3)
        self.assertEqual(nc.get_dim('i3'), 7)
        self.assertEqual(nc.get_dim('rec'), 3)
        self.assertEqual(nc.get_dim('ll'), 3)
        self.assertEqual(nc.get_dim('mm'), 1)
        self.assertEqual(nc.get_dim('nn'), 1)
        self.assertEqual(nc.get_dim('pp'), 7)
        self.assertEqual(nc.get_dim('qq'), 10)
        self.assertEqual(nc.get_dim('d0'), 2)
        self.assertEqual(nc.get_dim('d1'), 3)
        self.assertEqual(nc.get_dim('d2'), 5)
        self.assertEqual(nc.get_dim('d3'), 6)
        self.assertEqual(nc.get_dim('d4'), 4)
        self.assertEqual(nc.get_dim('d5'), 31)
        self.assertEqual(nc.get_dim('w'), 7)
        self.assertEqual(nc.get_dim('x'), 5)
        self.assertEqual(nc.get_dim('y'), 6)
        self.assertEqual(nc.get_dim('z'), 4)
        nc.close_file()

    def test_aa(self):
        from ..netcdf import NetCDF
        nc = NetCDF(self.testfn)
        ii = nc.get_dim('ii')
        arr = nc.get_array('aa', (ii,), 'int32')
        self.assertEqual(arr[0], -2)
        self.assertTrue((arr[1:] == 0).all())
        nc.close_file()
