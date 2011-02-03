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

"""
A wrapper to CUDA shared library.
"""

_libs = dict()
def get_lib(path):
    from ctypes import CDLL
    if path not in _libs:
        _libs[path] = CDLL(path)
    lib = _libs[path]
    return lib

class GpuMemory(object):
    """
    Record-keeper for memory on GPU.
    """
    def __init__(self, gptr, nbytes):
        self.gptr = gptr
        self.nbytes = nbytes

class Scuda(object):
    """
    Wrapper for CUDA library by using ctypes.

    @ivar cudart: CUDA runtime library.
    @itype cudart: ctypes.CDLL
    @ivar cuda: CUDA driver library.
    @itype cuda: ctypes.CDLL
    @ivar device: the ID of device to use.
    @itype device: int
    @ivar _alloc_gpumem: all allocated GpuMemory objects.
    @itype _alloc_gpumem: set
    """

    # cudaMemcpyKind enum in driver_types.h
    cudaMemcpyHostToHost = 0    # host -> host.
    cudaMemcpyHostToDevice = 1  # host -> device.
    cudaMemcpyDeviceToHost = 2  # device -> host.
    cudaMemcpyDeviceToDevice = 3    # device -> device.

    # cudaError enum in driver_types.h
    cudaSuccess = 0
    cudaErrorMissingConfiguration = 1
    cudaErrorMemoryAllocation = 2
    cudaErrorInitializationError = 3
    cudaErrorLaunchFailure = 4
    cudaErrorPriorLaunchFailure = 5
    cudaErrorLaunchTimeout = 6
    cudaErrorLaunchOutOfResources = 7
    cudaErrorInvalidDeviceFunction = 8
    cudaErrorInvalidConfiguration = 9
    cudaErrorInvalidDevice = 10
    cudaErrorInvalidValue = 11
    cudaErrorInvalidPitchValue = 12
    cudaErrorInvalidSymbol = 13
    cudaErrorMapBufferObjectFailed = 14
    cudaErrorUnmapBufferObjectFailed = 15
    cudaErrorInvalidHostPointer = 16
    cudaErrorInvalidDevicePointer = 17
    cudaErrorInvalidTexture = 18
    cudaErrorInvalidTextureBinding = 19
    cudaErrorInvalidChannelDescriptor = 20
    cudaErrorInvalidMemcpyDirection = 21
    cudaErrorAddressOfConstant = 22
    cudaErrorTextureFetchFailed = 23
    cudaErrorTextureNotBound = 24
    cudaErrorSynchronizationError = 25
    cudaErrorInvalidFilterSetting = 26
    cudaErrorInvalidNormSetting = 27
    cudaErrorMixedDeviceExecution = 28
    cudaErrorCudartUnloading = 29
    cudaErrorUnknown = 30
    cudaErrorNotYetImplemented = 31
    cudaErrorMemoryValueTooLarge = 32
    cudaErrorInvalidResourceHandle = 33
    cudaErrorNotReady = 34
    cudaErrorInsufficientDriver = 35
    cudaErrorSetOnActiveProcess = 36
    cudaErrorInvalidSurface = 37
    cudaErrorNoDevice = 38
    cudaErrorECCUncorrectable = 39
    cudaErrorSharedObjectSymbolNotFound = 40
    cudaErrorSharedObjectInitFailed = 41
    cudaErrorUnsupportedLimit = 42
    cudaErrorDuplicateVariableName = 43
    cudaErrorDuplicateTextureName = 44
    cudaErrorDuplicateSurfaceName = 45
    cudaErrorDevicesUnavailable = 46
    cudaErrorInvalidKernelImage = 47
    cudaErrorNoKernelImageForDevice = 48
    cudaErrorIncompatibleDriverContext = 49
    cudaErrorStartupFailure = 0x7f
    cudaErrorApiFailureBase = 10000

    def __init__(self, libname_cudart='libcudart.so',
        libname_cuda='libcuda.so'):
        """
        @keyword libname_cudart: name of the CUDA runtime library.
        @type libname_cudart: str
        @keyword libname_cuda: name of the CUDA driver library.
        @type libname_cuda: str
        """
        self.cudart = get_lib(libname_cudart)
        self.cuda = get_lib(libname_cuda)
        self.device = None
        self._alloc_gpumem = set()
        super(Scuda, self).__init__()
        self.use_first_valid_device()
    def __del__(self):
        for gmem in self._alloc_gpumem:
            self.free(gmem, do_remove=False)
    def __getattr__(self, key):
        if key.startswith('cuda'):
            return getattr(self.cudart, key)
        if key.startswith('cu'):
            return getattr(self.cuda, key)

    def __len__(self):
        from ctypes import byref, c_int
        dcnt = c_int()
        self.cudaGetDeviceCount(byref(dcnt))
        return dcnt.value

    def use_device(self, idx):
        """
        Use the specified device ID.  Set self.device.

        @param idx: device ID to use.
        @type idx: int
        @return: the device ID.
        @rtype: int
        """
        from ctypes import c_int
        assert idx < len(self)
        idx = c_int(idx)
        ret = self.cudaSetDevice(idx)
        if ret != self.cudaSuccess:
            raise ValueError(ret)
        self.device = ret
        return ret
    def use_first_valid_device(self):
        dev = None
        for idx in range(len(self)):
            try:
                dev = self.use_device(idx)
                break
            except ValueError:
                dev = None
                pass
        return dev

    def alloc(self, nbytes):
        from ctypes import byref, c_void_p
        gptr = c_void_p()
        self.cudaMalloc(byref(gptr), nbytes)
        gmem = GpuMemory(gptr, nbytes)
        self._alloc_gpumem.add(gmem)
        return gmem
    def free(self, gmem, do_remove=True):
        from ctypes import byref
        self.cudaFree(byref(gmem.gptr))
        if do_remove: self._alloc_gpumem.remove(gmem)
    def memcpy(self, tgt, src):
        from ctypes import c_void_p
        if isinstance(src, GpuMemory) and isinstance(tgt, GpuMemory):
            dkey = self.cudaMemcpyDeviceToDevice
            psrc = src.gptr
            ptgt = tgt.gptr
            assert ptgt.nbytes > psrc.nbytes
            nbytes = psrc.nbytes
        elif isinstance(src, GpuMemory):
            dkey = self.cudaMemcpyDeviceToHost
            psrc = src.gptr
            ptgt = tgt.ctypes.data_as(c_void_p)
            nbytes = tgt.nbytes
        elif isinstance(tgt, GpuMemory):
            dkey = self.cudaMemcpyHostToDevice
            psrc = src.ctypes.data_as(c_void_p)
            ptgt = tgt.gptr
            nbytes = src.nbytes
        else:
            raise TypeError('don\'t do host to host memcpy')
        self.cudaMemcpy(ptgt, psrc, nbytes, dkey)
