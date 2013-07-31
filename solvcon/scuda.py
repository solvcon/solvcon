# -*- coding: UTF-8 -*-
#
# Copyright (c) 2011, Yung-Yu Chen <yyc@solvcon.net>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the SOLVCON nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""
A wrapper to CUDA shared library by using ctypes.
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

from ctypes import Structure, c_uint, c_char, c_size_t, c_int
class CudaDim3(Structure):
    _fields_ = [
        ('x', c_uint), ('y', c_uint), ('z', c_uint),
    ]
    def __init__(self, *args, **kw):
        super(CudaDim3, self).__init__(*args, **kw)
        for key in ['x', 'y', 'z']:
            if key not in kw: setattr(self, key, 1)
class CudaDeviceProp(Structure):
    _fields_ = [
        ('name', c_char*256),
        ('totalGlobalMem', c_size_t),
        ('sharedMemPerBlock', c_size_t),
        ('regsPerPerBlock', c_int),
        ('warpSize', c_int),
        ('memPitch', c_size_t),
        ('maxThreadsPerBlock', c_int),
        ('maxThreadsDim', c_int*3),
        ('maxGridSize', c_int*3),
        ('totalConstMem', c_size_t),
        ('major', c_int),
        ('minor', c_int),
        ('clockRate', c_int),
        ('textureAlignment', c_size_t),
        ('deviceOverlap', c_int),
        ('multiProcessorCount', c_int),
        ('kernelExecTimeoutEnabled', c_int),
        ('integrated', c_int),
        ('canMapHostMemory', c_int),
        ('computeMode', c_int),
        ('maxTexture1D', c_int),
        ('maxTexture2D', c_int*2),
        ('maxTexture3D', c_int*3),
        ('maxTexture2DArray', c_int*3),
        ('surfaceAlignment', c_size_t),
        ('concurrentKernels', c_int),
        ('ECCEnabled', c_int),
        ('pciBusID', c_int),
        ('pciDeviceID', c_int),
        ('tccDriver', c_int),
        ('__cudaReserved', c_int*21),
    ]
    def __str__(self):
        return self.name
    def get_compute_capability(self):
        return '%d.%d'%(self.major, self.minor)
    def has_compute_capability(self, *args):
        """
        Determine if the device has the compute capability specified by the
        arguments.  Arguments can be in the format of (i) 'x.y' or (ii) x, y.

        @return: has the compute capability or not.
        @rtype: bool
        """
        # parse input.
        if len(args) == 1 and isinstance(args[0], basestring):
            major, minor = [int(val) for val in args[0].split('.')]
        elif len(args) == 2:
            major, minor = args
        else:
            raise ValueError('incompatible arguments.')
        # determine capability.
        if self.major > major:
            return True
        elif self.major == major and self.minor >= minor:
            return True
        else:
            return False
del Structure, c_uint, c_char, c_size_t, c_int

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

    @staticmethod
    def has_cuda():
        ret = False
        try:
            get_lib('libcudart.so')
            get_lib('libcuda.so')
        except OSError:
            ret = False
        else:
            ret = True
        return ret

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
        self.devprop = CudaDeviceProp()
        self._alloc_gpumem = set()
        super(Scuda, self).__init__()
        self._dcnt = None
        self.use_first_valid_device()
    def __del__(self):
        for gmem in self._alloc_gpumem:
            self.free(gmem, do_remove=False)
    def __getattr__(self, key):
        if key.startswith('cuda'):
            return getattr(self.cudart, key)
        elif key.startswith('cu'):
            return getattr(self.cuda, key)
        else:
            raise KeyError

    def __len__(self):
        from ctypes import byref, c_int
        if self._dcnt is None:
            dcnt = c_int()
            self.cudaGetDeviceCount(byref(dcnt))
            self._dcnt = dcnt.value
        return self._dcnt

    def download_device_properties(self):
        """
        Use CUDA runtime API to download device properties to self object.  Set
        self._device_properties.

        @return: nothing
        """
        from ctypes import byref, c_int
        ret = self.cudaGetDeviceProperties(byref(self.devprop),
            c_int(self.device))
        if ret != self.cudaSuccess:
            raise ValueError(ret)
    def use_device(self, idx):
        """
        Use the specified device ID.  Set self.device.

        @param idx: device ID to use.
        @type idx: int
        @return: the device ID.
        @rtype: int
        """
        from ctypes import c_int, byref
        assert idx < len(self)
        idx = c_int(idx)
        ret = self.cudaSetDevice(idx)
        if ret != self.cudaSuccess:
            raise ValueError(ret)
        self.device = idx.value
        self.download_device_properties()
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
        self.cudaFree(byref(gmem.gptr)) # XXX: is byref right?
        if do_remove:
            try:
                self._alloc_gpumem.remove(gmem)
            except KeyError:
                pass
    def memcpy(self, tgt, src):
        if isinstance(src, GpuMemory) and isinstance(tgt, GpuMemory):
            dkey = self.cudaMemcpyDeviceToDevice
            psrc = src.gptr
            ptgt = tgt.gptr
            assert ptgt.nbytes > psrc.nbytes
            nbytes = psrc.nbytes
        elif isinstance(src, GpuMemory):
            dkey = self.cudaMemcpyDeviceToHost
            psrc = src.gptr
            ptgt = tgt.ctypes._as_parameter_
            nbytes = tgt.nbytes
        elif isinstance(tgt, GpuMemory):
            dkey = self.cudaMemcpyHostToDevice
            psrc = src.ctypes._as_parameter_
            ptgt = tgt.gptr
            nbytes = src.nbytes
        else:
            raise TypeError('don\'t do host to host memcpy')
        self.cudaMemcpy(ptgt, psrc, nbytes, dkey)
