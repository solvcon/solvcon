# -*- coding: UTF-8 -*-

from unittest import TestCase

class BaseTestRead(TestCase):
    __test__ = False

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

    def test_bb(self):
        from numpy import array
        from ..netcdf import NetCDF
        nc = NetCDF(self.testfn)
        kk = nc.get_dim('kk')
        jj = nc.get_dim('jj')
        arr = nc.get_array('bb', (kk, jj), 'int32')
        self.assertTrue((arr == array([
            [-4, -3, -2], [-1, 0, 1], [2, 3, 0],
        ])).all())
        nc.close_file()

    def test_ce(self):
        from numpy import array
        from ..netcdf import NetCDF
        nc = NetCDF(self.testfn)
        rec = nc.get_dim('rec')
        i2 = nc.get_dim('i2')
        i3 = nc.get_dim('i3')
        arr = nc.get_array('ce', (rec, i2, i3), 'float32')
        self.assertTrue((arr == array([
            [[  1.,   2.,   3.,   4.,   5.,   6.,   7.],
             [  8.,   9.,  10.,  11.,  12.,  13.,  14.],
             [ 15.,  16.,  17.,  18.,  19.,  20.,  21.]],
            [[  1.,   2.,   3.,   4.,   5.,   6.,   7.],
             [  8.,   9.,  10.,  11.,  12.,  13.,  14.],
             [ 15.,  16.,  17.,  18.,  19.,  20.,  21.]],
            [[ 43.,  44.,  45.,  46.,  47.,  48.,  49.],
             [ 50.,  51.,  52.,  53.,  54.,  55.,  56.],
             [ 57.,  58.,  59.,  60.,  61.,  62.,   1.]],
        ], dtype='float32')).all())
        nc.close_file()

    def test_doublevar(self):
        from numpy import arange
        from ..netcdf import NetCDF
        nc = NetCDF(self.testfn)
        w = nc.get_dim('w')
        x = nc.get_dim('x')
        y = nc.get_dim('y')
        z = nc.get_dim('z')
        arr = nc.get_array('doublevar', (w, x, y, z), 'float64')
        arr2 = arange(w*x*y*z, dtype='float64') - 420
        arr2[420-148+1] = 0
        arr2[-1] = 0
        arr2 = arr2.reshape(arr.shape)
        self.assertTrue((arr == arr2).all())
        nc.close_file()

class TestReadClassic(BaseTestRead):
    __test__ = True
    import os
    from ...conf import env
    testfn = [env.datadir] + ['ref_nctest_classic.nc']
    testfn = os.path.join(*testfn)

class TestReadLong(BaseTestRead):
    __test__ = True
    import os
    from ...conf import env
    testfn = [env.datadir] + ['ref_nctest_64bit_offset.nc']
    testfn = os.path.join(*testfn)
