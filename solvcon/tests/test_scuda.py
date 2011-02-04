# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2011 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

from unittest import TestCase

class TestScuda(TestCase):
    def setUp(self):
        import sys, os
        from ctypes import CDLL
        from nose.plugins.skip import SkipTest
        from ..conf import env
        if sys.platform.startswith('win'): raise SkipTest
        if env.scu is None: raise SkipTest
        self.scu = env.scu
        if self.scu:
            libpath = os.path.join(env.libdir, 'libsc_cudatest.so')
            self.lib = CDLL(libpath)

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
