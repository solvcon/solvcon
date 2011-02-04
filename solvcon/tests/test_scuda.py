# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2011 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

from unittest import TestCase

from ctypes import Structure, c_int, c_float, c_double, c_void_p
class Custom(Structure):
    _fields_ = [
        ('nelm', c_int),
        ('dval', c_double),
        ('arra', c_void_p),
        ('arrb', c_void_p),
        ('arrc', c_void_p),
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
        nelm = 337
        nthread = 32
        # CPU data.
        ctm = Custom(nelm=nelm, dval=4.0)
        arra = arange(ctm.nelm, dtype='float64')
        ctm.arra = arra.ctypes.data_as(c_void_p)
        arrb = arange(ctm.nelm, dtype='float64')
        ctm.arrb = arrb.ctypes.data_as(c_void_p)
        arrc = empty(ctm.nelm, dtype='float64')
        ctm.arrc = arrc.ctypes.data_as(c_void_p)
        # GPU data.
        gtm = Custom(nelm=ctm.nelm, dval=ctm.dval)
        garra = scu.alloc(arra.nbytes)
        scu.memcpy(garra, arra)
        gtm.arra = garra.gptr
        garrb = scu.alloc(arrb.nbytes)
        scu.memcpy(garrb, arrb)
        gtm.arrb = garrb.gptr
        garrc = scu.alloc(arrc.nbytes)
        scu.memcpy(garrc, arrc)
        gtm.arrc = garrc.gptr
        # operate.
        gp = scu.alloc(sizeof(gtm))
        scu.cudaMemcpy(gp.gptr, byref(gtm), sizeof(gtm),
            scu.cudaMemcpyHostToDevice)
        lib.structop(nthread, byref(ctm), gp.gptr)
        # get back and compare.
        rarrc = empty(ctm.nelm, dtype='float64')
        rarrc.fill(1000)
        scu.memcpy(rarrc, garrc)
        self.assertTrue((rarrc == arra+arrb+ctm.dval).all())
        # free.
        scu.free(garra)
        scu.free(garrb)
        scu.free(garrc)
        scu.free(gp)
