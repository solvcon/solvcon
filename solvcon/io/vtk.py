# -*- coding: UTF-8 -*-
#
# Copyright (c) 2008, Yung-Yu Chen <yyc@solvcon.net>
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
VTK legacy file.
"""

class VtkLegacyWriter(object):
    """
    Base class for VTK legacy format data writer.

    @cvar cltpn_map: mapper for cell type number.
    @ctype cltpn_map: numpy.ndarray

    @ivar binary: flag for BINARY (False for ASCII).
    @itype binary: bool
    @ivar fpdtype: floating-point data type (single/double).
    @itype fpdtype: numpy.dtype
    @ivar blk: corresponding block object.
    @itype blk: solvcon.block.Block
    """
    from numpy import array
    cltpn_map = array([1, 3, 9, 5, 12, 10, 13, 14], dtype='int32')
    del array

    def __init__(self, blk, *args, **kw):
        self.binary = kw.pop('binary', False)
        fpdtype = kw.pop('fpdtype', None)
        if fpdtype == None:
            fpdtype = blk.fpdtype
        self.fpdtype = fpdtype
        super(VtkLegacyWriter, self).__init__(*args, **kw)
        self.blk = blk

    @staticmethod
    def _ensure_endian(arr):
        """
        Ensure the endianness for array.  VTK legacy format require BIG ENDIAN
        for binary data set.  WILL CHANGE the byte order of the array passed
        in.  Currently only work for Intel architecture.

        @param arr: array to be modified to be BIG ENDIAN.
        @type arr: numpy.ndarray
        """
        arr.byteswap(True)

    @staticmethod
    def _get_dtypestr(arr):
        """
        Determine VTK type string from the array dtype.

        @param arr: array to be determined.
        @type arr: numpy.ndarray
        @return: VTK type string.
        @rtype: str
        """
        from numpy import float32, float64
        if arr.dtype == float32:
            return 'float'
        elif arr.dtype == float64:
            return 'double'
        else:
            raise TypeError, 'type of arr has to be either float32 or float64'

    def _make_file_header(self):
        """
        Make up VTK legacy file header lines.  Essentially the first three
        lines.

        @return: file headers.
        @rtype: str
        """
        ret = []
        ret.append('# vtk DataFile Version 3.0')
        ret.append('Written with solvcon.io.vtk.legacy.VtkLegacy object.')
        if self.binary:
            ret.append('BINARY')
        else:
            ret.append('ASCII')
        return '\n'.join(ret)

class VtkLegacyUstGridWriter(VtkLegacyWriter):
    """
    VTK legacy unstructured mesh file format.  Capable for ASCII or BINARY.

    @ivar cache_grid: flag to cache the grid data (True) or not (False).
        Default True.
    @itype cache_grid: bool
    @ivar scalars: dictionary holding scalar data.
    @itype scalars: dict
    @ivar vectors: dictionary holding vector data.
    @itype vectors: dict
    @ivar griddata: cached grid data.
    @itype griddata: str
    """
    def __init__(self, blk, *args, **kw):
        self.cache_grid = kw.pop('cache_grid', True)
        self.scalars = kw.pop('scalars', dict())
        self.vectors = kw.pop('vectors', dict())
        super(VtkLegacyUstGridWriter, self).__init__(blk, *args, **kw)
        self.griddata = None

    def write(self, outf, close_on_finish=False):
        """
        Output to VTK file.

        @param outf: output file object or file name.
        @type outf: file str
        @keyword close_on_finish: flag close on finishing (True).  Default
            False.  If outf is file name, the output file will be close no
            matter what is set in this flag.
        @type close_on_finish: bool
        @return: nothing
        """
        if isinstance(outf, str):
            mode = 'wb' if self.binary else 'w'
            outf = open(outf, mode)
            close_on_finish = True
        # generate grid data.
        if not self.griddata:
            self.griddata = '\n'.join([
                self._make_file_header(),
                self._make_data_header(),
                self._make_nodes(),
                self._make_cells(),
            ])
        # write.
        outf.write(self.griddata)
        outf.write('\n')
        outf.write(self._make_value())
        outf.write('\n')
        if close_on_finish:
            outf.close()
        # discard griddata if not cached.
        if not self.cache_grid:
            self.griddata = None

    def _convert_scalar_data(self, arr):
        """
        Helper to convert scalar data array from a block.

        @param arr: array to be converted.  It will be copied and remain
            untouched.
        @type arr: numpy.ndarray
        @return: converted data string.
        @rtype: str
        """
        arr = arr.copy()
        if self.binary:
            self._ensure_endian(arr)
            return arr.tostring()
        else:
            return '\n'.join(['%e'%val for val in arr])

    def _convert_vector_data(self, arr):
        """
        Helper to convert vector data array from a block.

        @param arr: array to be converted.  It will be copied and remain
            untouched.
        @type arr: numpy.ndarray
        @return: converted data string.
        @rtype: str
        """
        from numpy import empty
        arr = arr.copy()
        ndim = self.blk.ndim
        nit = arr.shape[0]
        if ndim == 2:
            arrn = empty((nit, ndim+1), dtype=arr.dtype)
            arrn[:,2] = 0.0
            arrn[:,:2] = arr[:,:]
            arr = arrn
        if self.binary:
            self._ensure_endian(arr)
            return arr.tostring()
        else:
            tmpl = '%e %e %e'
            return '\n'.join([tmpl%tuple(vec) for vec in arr])

    def _make_data_header(self):
        """
        @return: made up unstructred grid data header.
        @rtype: str
        """
        return 'DATASET UNSTRUCTURED_GRID'

    def _make_nodes(self):
        """
        @return: made up node coordinates.
        @rtype: str
        """
        nnode = self.blk.nnode
        ndcrd = self.blk.ndcrd.astype(self.fpdtype)
        ret = []
        ret.append('POINTS %d %s' % (nnode, self._get_dtypestr(ndcrd)))
        ret.append(self._convert_vector_data(ndcrd))
        return '\n'.join(ret)

    def _make_cells(self):
        """
        @return: made up cell definition.
        @rtype: str
        """
        from numpy import empty
        blk = self.blk
        ncell = blk.ncell
        clnds = blk.clnds
        ret = []
        # node definitions.
        size = clnds[:,0].sum() + ncell
        ret.append('CELLS %d %d' % (ncell, size))
        if self.binary:
            # make data.
            arr = empty(size, dtype='int32')
            icl = 0
            it = 0
            while icl < ncell:
                ncl = clnds[icl,0]
                arr[it] = ncl
                arr[it+1:it+1+ncl] = clnds[icl,1:ncl+1]
                it += 1+ncl
                icl += 1
            # write.
            self._ensure_endian(arr)
            ret.append(arr.tostring())
        else:
            for nds in clnds:
                ncl = nds[0]
                cltmpl = ' '.join(['%d']*(ncl+1))
                ret.append(cltmpl%tuple(nds[:ncl+1]))
        # node types.
        ret.append('CELL_TYPES %d' % ncell)
        cltpn = self.cltpn_map[blk.cltpn]
        if self.binary:
            self._ensure_endian(cltpn)
            ret.append(cltpn.tostring())
        else:
            ret.extend(['%d'%val for val in cltpn])
        return '\n'.join(ret)

    def _make_value(self):
        """
        @return: made up field value.
        @rtype: str
        """
        ncell = self.blk.ncell
        ret = []
        ret.append(self._make_value_scalar())
        ret.append(self._make_value_vector())
        ret = '\n'.join(ret).strip()
        if not ret:
            return ret
        else:
            return '\n'.join(['CELL_DATA %d'%ncell, ret])

    def _make_value_scalar(self):
        """
        @return: made up scalar value.
        @rtype: str
        """
        import sys
        blk = self.blk
        sca = dict()
        sca.update(self.scalars)
        for key in sca:
            sca[key] = sca[key].astype(self.fpdtype)
        ret = []
        for key in sorted(sca.keys()):
            ret.append('SCALARS %s %s' % (key, self._get_dtypestr(sca[key])))
            ret.append('LOOKUP_TABLE default')
            try:
                ret.append(self._convert_scalar_data(sca[key]))
            except:
                sys.stderr.write('Error on key = %s\n' % key)
                raise
        return '\n'.join(ret)

    def _make_value_vector(self):
        """
        @return: made up vector value.
        @rtype: str
        """
        from numpy import sqrt
        blk = self.blk
        ncell = blk.ncell
        vec = dict()
        vec.update(self.vectors)
        for key in vec:
            vec[key] = vec[key].astype(self.fpdtype)
        ret = []
        for key in sorted(vec.keys()):
            ret.append('VECTORS %s %s' % (key, self._get_dtypestr(vec[key])))
            ret.append(self._convert_vector_data(vec[key]))
            norm = sqrt((vec[key]**2).sum(axis=1))
            ret.append('SCALARS |%s| %s' % (key, self._get_dtypestr(norm)))
            ret.append('LOOKUP_TABLE default')
            ret.append(self._convert_scalar_data(norm))
        return '\n'.join(ret)
