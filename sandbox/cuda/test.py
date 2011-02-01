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

# cudaMemcpyKind in driver_types.h
cudaMemcpyHostToHost = 0    # host -> host.
cudaMemcpyHostToDevice = 1  # host -> device.
cudaMemcpyDeviceToHost = 2  # device -> host.
cudaMemcpyDeviceToDevice = 3    # device -> device.

def main():
    from ctypes import cdll, CDLL, byref, c_void_p, POINTER, c_float, sizeof
    from numpy import empty, arange
    cudart = cdll.LoadLibrary('libcudart.so')
    lib = cdll.LoadLibrary('libsc_cutest3d.so')
    nelm = 1024

    # allocate on CPU.
    arra = arange(nelm, dtype='float32')
    arrb = -arra
    arrc = empty(nelm, dtype='float32')
    arrc.fill(2)

    # allocate on GPU.
    pcrra = c_void_p()
    pcrrb = c_void_p()
    pcrrc = c_void_p()
    cudart.cudaMalloc(byref(pcrra), arra.nbytes)
    cudart.cudaMalloc(byref(pcrrb), arrb.nbytes)
    cudart.cudaMalloc(byref(pcrrc), arrc.nbytes)

    # copy from host to device.
    cudart.cudaMemcpy(pcrra, arra.ctypes.data_as(POINTER(c_float)),
        arra.nbytes, cudaMemcpyHostToDevice)
    cudart.cudaMemcpy(pcrrb, arrb.ctypes.data_as(POINTER(c_float)),
        arrb.nbytes, cudaMemcpyHostToDevice)

    # invoke kernel.
    lib.invoke_VecAdd(pcrra, pcrrb, pcrrc, nelm)

    # copy from device to host.
    cudart.cudaMemcpy(arrc.ctypes.data_as(POINTER(c_float)), pcrrc,
        arrc.nbytes, cudaMemcpyDeviceToHost)
    print arrc.sum()

    # deallocate on GPU.
    cudart.cudaFree(byref(pcrra))
    cudart.cudaFree(byref(pcrrb))
    cudart.cudaFree(byref(pcrrc))

if __name__ == '__main__':
    main()
