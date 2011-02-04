#!/usr/bin/env python2.6
# -*- coding: UTF-8 -*-
#
# Copyright (C) 2011 Yung-Yu Chen <yyc@solvcon.net>.
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

def main():
    from ctypes import cdll, CDLL, byref, c_void_p, POINTER, c_float, sizeof
    from numpy import empty, arange
    from solvcon.conf import env
    scu = env.scu
    lib = cdll.LoadLibrary('libsc_cutest3d.so')
    nelm = 1024

    print len(scu), 'CUDA device(s)'
    print 'I\'m using device #%d' % scu.device

    # allocate on CPU.
    arra = arange(nelm, dtype='float32')
    arrb = -arra * 3
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
    lib.invoke_VecAdd(gmema.gptr, gmemb.gptr, gmemc.gptr, nelm)

    # copy from device to host.
    scu.memcpy(arrc, gmemc)
    print arrc.sum()

    # deallocate on GPU.
    scu.free(gmemc)
    scu.free(gmemb)
    scu.free(gmema)

if __name__ == '__main__':
    main()
