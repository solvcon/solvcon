# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2011 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

from unittest import TestCase

from ctypes import Structure, c_int, c_float, c_double, c_void_p
class Custom(Structure):
    _fields_ = [
        ('nelm', c_int),
        ('dval', c_double),
        ('arr', c_void_p),
    ]

class TestScuda(TestCase):
    def setUp(self):
        import sys, os
        from nose.plugins.skip import SkipTest
        from ..conf import env
        if sys.platform.startswith('win'): raise SkipTest
        if env.scu is None: raise SkipTest
        self.scu = env.scu

class TestScuda20(TestScuda):
    def setUp(self):
        super(TestScuda20, self).setUp()
        import os
        from ctypes import CDLL
        from ..conf import env
        if not self.scu.devprop.has_compute_capability('2.0'): raise SkipTest
        libpath = os.path.join(env.libdir, 'libsc_cuda20test.so')
        self.lib = CDLL(libpath)

    def test_properties(self):
        scu = self.scu
        self.assertTrue(scu.devprop.major >= 2)

    def test_vecadd_float(self):
        from numpy import empty, arange
        scu = self.scu
        lib = self.lib
        nelm = 1024
        # allocate on CPU.
        arra = arange(nelm, dtype='float32')
        arrb = -arra
        arrc = empty(nelm, dtype='float32')
        arrc.fill(2)
        # allocate on GPU.
        gmema = scu.alloc(arra.nbytes)
        gmemb = scu.alloc(arrb.nbytes)
        gmemc = scu.alloc(arrc.nbytes)
        # copy from host to device.
        scu.memcpy(gmema, arra)
        scu.memcpy(gmemb, arrb)
        # invoke kernel.
        lib.vecadd_float(gmema.gptr, gmemb.gptr, gmemc.gptr, nelm)
        # copy from device to host.
        self.assertTrue((arrc == 2.0).all())
        scu.memcpy(arrc, gmemc)
        self.assertTrue((arrc == 0.0).all())
        # deallocate on GPU.
        scu.free(gmemc)
        scu.free(gmemb)
        scu.free(gmema)

    def test_structop(self):
        from ctypes import c_void_p, sizeof, byref
        from numpy import empty, arange
        scu = self.scu
        lib = self.lib
        # CPU data.
        ctm = Custom(nelm=1024, dval=4.0)
        arr = arange(ctm.nelm, dtype='float64')
        ctm.arr = arr.ctypes.data_as(c_void_p)
        # GPU data.
        gtm = Custom(nelm=1024, dval=4.0)
        garr = scu.alloc(arr.nbytes)
        scu.memcpy(garr, arr)
        gtm.arr = garr.gptr
        gp = scu.alloc(sizeof(gtm))
        # operate.
        scu.cudaMemcpy(gp.gptr, byref(gtm), sizeof(gtm),
            scu.cudaMemcpyHostToDevice)
        lib.structop(byref(ctm), gp.gptr)
        # get back and compare.
        rarr = empty(ctm.nelm, dtype='float64')
        rarr.fill(1000)
        scu.memcpy(rarr, garr)
        self.assertTrue((rarr == arr+ctm.dval).all())
        # free.
        scu.free(garr)
        scu.free(gp)
