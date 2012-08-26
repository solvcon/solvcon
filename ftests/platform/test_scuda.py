# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2012 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

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
        from solvcon.conf import env
        if sys.platform.startswith('win'): raise SkipTest
        if env.scu is None: raise SkipTest
        self.scu = env.scu

class TestScuda20(TestScuda):
    def setUp(self):
        super(TestScuda20, self).setUp()
        import os
        from ctypes import CDLL
        from nose.plugins.skip import SkipTest
        from solvcon.conf import env
        if not self.scu.devprop.has_compute_capability('2.0'): raise SkipTest
        libpath = os.path.join(env.libdir, 'libsc_cuda20test.so')
        self.lib = CDLL(libpath)

    def test_properties(self):
        scu = self.scu
        self.assertTrue(scu.devprop.major >= 2)

    def test_vecadd_float(self):
        from ctypes import (byref, c_size_t, c_void_p, c_int,
            sizeof, create_string_buffer, memmove)
        from numpy import empty, arange
        from solvcon.scuda import CudaDim3
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
        ctm.arra = arra.ctypes._as_parameter_
        arrb = arange(ctm.nelm, dtype='float64')
        ctm.arrb = arrb.ctypes._as_parameter_
        arrc = empty(ctm.nelm, dtype='float64')
        ctm.arrc = arrc.ctypes._as_parameter_
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

class TestScuda13(TestScuda):
    def setUp(self):
        super(TestScuda13, self).setUp()
        import os
        from ctypes import CDLL
        from nose.plugins.skip import SkipTest
        from solvcon.conf import env
        if not self.scu.devprop.has_compute_capability('1.3'): raise SkipTest
        libpath = os.path.join(env.libdir, 'libsc_cuda13test.so')
        self.lib = CDLL(libpath)

    def test_properties(self):
        scu = self.scu
        self.assertTrue(scu.devprop.major >= 1)
        if scu.devprop.major == 1:
            self.assertTrue(scu.devprop.minor >= 3)

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
        ctm.arra = arra.ctypes._as_parameter_
        arrb = arange(ctm.nelm, dtype='float64')
        ctm.arrb = arrb.ctypes._as_parameter_
        arrc = empty(ctm.nelm, dtype='float64')
        ctm.arrc = arrc.ctypes._as_parameter_
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

# vim: set ff=unix fenc=utf8 ft=python ai et sw=4 ts=4 tw=79:
