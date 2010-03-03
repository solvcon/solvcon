# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2009 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""Intrinsic format mesh IO."""

class BlockIO(object):
    """
    Save/load functionality for block/mesh object.

    @cvar READ_BLOCK: length per reading from file.
    @ctype READ_BLOCK: int
    @cvar BINARY_MARKER: a string to mark the starting of binary data.
    @ctype BINARY_MARKER: str
    @ivar blk: attached block object.
    @itype blk: solvcon.block.Block
    @ivar filename: associated in/out filename.
    @itype filename: str
    @ivar fpdtype: specified fpdtype for I/O.
    @itype fpdtyp: numpy.dtype
    @ivar bcmapper: boundary condition object conversion mapper.
    @itype bcmapper: dict
    """

    READ_BLOCK = 1024
    BINARY_MARKER = '-*- start of binary data -*-'
    FORMAT_REV = '0.0.0.1'
    META_GLOBAL = (
        'FORMAT_REV',
        'flag_compress',
    )
    META_DESC = (
        'blkn', 'fpdtypestr',
    )
    META_GEOM = (
        'FCMND', 'CLMND', 'CLMFC',
        'ndim', 'nnode', 'nface', 'ncell', 'nbound',
        'ngstnode', 'ngstface', 'ngstcell',
    )
    META_ATT = (
        'ngroup', 'nbc',
    )

    def __init__(self, **kw):
        self.blk = kw.pop('blk', None)
        self.fpdtype = kw.pop('fpdtype', None)
        self.bcmapper = kw.pop('bcmapper', None)
        self.flag_compress = kw.pop('flag_compress', '')

    @property
    def meta_length(self):
        return sum([len(getattr(self, key)) for key in self.__class__.__dict__
            if key.startswith('META_')])

    def save(self, blk=None, stream=None):
        """
        Save the block object into a file.
        
        @keyword blk: to-be-written block object.
        @type blk: solvcon.block.Block
        @keyword stream: file object or file name to be read.
        @type stream: file or str
        """
        if blk == None:
            blk = self.blk
        if stream == None:
            stream = open(self.filename, 'wb')
        elif isinstance(stream, str):
            stream = open(stream, 'wb')
        # text part.
        stream.write('-*- solvcon blk mesh file -*-\n')
        self._save_meta(blk, stream)
        for ig in range(len(blk.grpnames)):
            stream.write('group%d = %s\n' % (ig, blk.grpnames[ig]))
        for ibc in range(len(blk.bclist)):
            bc = blk.bclist[ibc]
            stream.write('bc%d = %s, %s, %d, %d\n' % (
                bc.sern, bc.name, str(bc.blkn), len(bc), bc.nvalue,
            ))
        # binary part.
        stream.write('-*- start of binary data -*-\n')
        self._save_connectivity(blk, stream)
        self._save_type(blk, stream)
        self._save_metrics(blk, stream)
        self._save_boundcond(blk, stream)

    def _save_meta(self, blk, stream):
        """
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @param stream: output stream.
        @type stream: file
        @return: nothing.
        """
        # global.
        for key in self.META_GLOBAL:
            stream.write('%s = %s\n' % (key, str(getattr(self, key))))
        # special description and geometry.
        for key in self.META_DESC + self.META_GEOM:
            stream.write('%s = %s\n' % (key, str(getattr(blk, key))))
        # attached/META_ATT.
        stream.write('ngroup = %d\n' % len(blk.grpnames))
        stream.write('nbc = %d\n' % len(blk.bclist))

    @staticmethod
    def _write_array(flag_compress, arr, stream):
        """
        @param flag_compress: how to compress data arrays.
        @type flag_compress: str
        @param arr: the array to be written.
        @type arr: numpy.ndarray
        @param stream: output stream.
        @type stream: file
        @return: nothing.
        """
        import bz2
        import zlib
        import struct
        if flag_compress == 'bz2':
            data = bz2.compress(arr.data, 9)
            stream.write(struct.pack('q', len(data)))
        elif flag_compress == 'gz':
            data = zlib.compress(arr.data, 9)
            stream.write(struct.pack('q', len(data)))
        else:
            data = arr.data
        stream.write(data)

    def _save_connectivity(self, blk, stream):
        """
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @param stream: output stream.
        @type stream: file
        @return: nothing.
        """
        from bz2 import compress
        for key in (
            'shfcnds', 'shfccls', 'shclnds', 'shclfcs',
        ):
            self._write_array(self.flag_compress, getattr(blk, key), stream)

    def _save_type(self, blk, stream):
        """
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @param stream: output stream.
        @type stream: file
        @return: nothing.
        """
        for key in (
            'shfctpn', 'shcltpn', 'shclgrp',
        ):
            self._write_array(self.flag_compress, getattr(blk, key), stream)

    def _save_metrics(self, blk, stream):
        """
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @param stream: output stream.
        @type stream: file
        @return: nothing.
        """
        for key in (
            'shndcrd', 'shfccnd', 'shfcnml', 'shfcara',
            'shclcnd', 'shclvol',
        ):
            self._write_array(self.flag_compress, getattr(blk, key), stream)

    def _save_boundcond(self, blk, stream):
        """
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @param stream: output stream.
        @type stream: file
        @return: nothing.
        """
        self._write_array(self.flag_compress, blk.bndfcs, stream)
        for bc in blk.bclist:
            if len(bc) > 0:
                self._write_array(self.flag_compress, bc.facn, stream)
            if bc.value.shape[1] > 0:
                self._write_array(self.flag_compress, bc.value, stream)

    def load(self, stream=None, bcmapper=None):
        """
        Load block from stream with BC mapper applied.
        
        @keyword stream: file object or file name to be read.
        @type stream: file or str
        @keyword bcmapper: BC type mapper.
        @type bcmapper: dict
        @return: the read block object.
        @rtype: solvcon.block.Block
        """
        from ..block import Block
        bcmapper = bcmapper if bcmapper != None else self.bcmapper
        if stream == None:
            stream = open(self.filename, 'rb')
        elif isinstance(stream, str):
            stream = open(stream, 'rb')
        # determine the text part and binary part.
        lines, textlen = self._get_textpart(stream)
        meta = self._parse_meta(lines)
        fpdtype = meta.fpdtype if self.fpdtype == None else self.fpdtype
        blk = Block(fpdtype=fpdtype)
        blk.blkn = meta.blkn
        ## groups.
        meta_length = self.meta_length
        begin = meta_length + 1
        end = meta_length + 1 + meta.ngroup
        blk.grpnames = [line.split('=')[-1].strip() for line in lines[begin:end]]
        ## BC object information.
        begin = meta_length + 1 + meta.ngroup
        end = meta_length + 1 + meta.ngroup + meta.nbc
        bcsinfo = []
        for line in lines[begin:end]:
            key, value = line.split('=')
            sern = int(key.strip()[2:])
            name, blkn, flen, nval = [tok.strip() for tok in value.split(',')]
            try:
                blkn = int(blkn)
            except ValueError:
                blkn = None
            flen = int(flen)
            nval = int(nval)
            bcsinfo.append((sern, name, blkn, flen, nval))
        # load arrays.
        stream.seek(textlen)
        self._load_connectivity(meta, stream, blk)
        self._load_type(meta, stream, blk)
        self._load_metrics(meta, stream, blk)
        self._load_boundcond(meta, bcsinfo, fpdtype, stream, blk)
        # construct subarrays.
        self._construct_subarrays(meta, blk)
        # post-process for BCs.
        if bcmapper != None:
            self._convert_bc(bcmapper, blk)
        return blk

    def _get_textpart(self, stream):
        """
        Retrieve the text part from the stream.
        
        @param stream: input stream.
        @type stream: file
        @return: text lines and length of text part (including separator).
        @rtype: tuple
        """
        buf = ''
        while self.BINARY_MARKER not in buf:
            buf += stream.read(self.READ_BLOCK)
        buf = buf.split(self.BINARY_MARKER)[0]
        lines = buf.split('\n')[:-1]
        textlen = len(buf) + len(self.BINARY_MARKER) + 1
        # assert text format.
        assert lines[0].strip()[:3] == '-*-'
        assert lines[0].strip()[-3:] == '-*-'
        # assert text end location.
        stream.seek(0)
        assert stream.read(textlen)[-1] == '\n'
        stream.seek(0)
        return lines, textlen

    def _parse_meta(self, lines):
        """
        Parse meta information from the file.
        
        @param lines: text part of the file.
        @type lines: list
        @return: meta information dictionary.
        @rtype: solvcon.gendata.AttributeDict
        """
        import numpy as np
        from ..gendata import AttributeDict
        meta = AttributeDict()
        meta_length = self.meta_length
        # parse text.
        ## flags.
        begin = 1
        end = begin + len(self.META_GLOBAL)
        for line in lines[begin:end]:
            key, val = [to.strip() for to in line.split('=')]
            meta[key] = val
        ## type.
        begin = end
        end = begin + len(self.META_DESC)
        for line in lines[begin:end]:
            key, val = [to.strip() for to in line.split('=')]
            meta[key] = val
        ## geometry.
        begin = end
        end = begin + len(self.META_GEOM) + len(self.META_ATT)
        for line in lines[begin:end]:
            key, val = [to.strip() for to in line.split('=')]
            meta[key] = int(val)
        # further process parsed text.
        meta['fpdtype'] = getattr(np, meta.pop('fpdtypestr'))
        try:
            meta.blkn = int(meta.blkn)
        except ValueError:
            meta.blkn = None
        return meta

    @staticmethod
    def _read_array(flag_compress, shape, dtype, stream):
        """
        Read data from the input stream and convert it to ndarray with given
        shape and dtype.

        @param flag_compress: how to compress data arrays.
        @type flag_compress: str
        @param shape: ndarray shape.
        @type shape: tuple
        @param dtype: ndarray dtype.
        @type dtype: numpy.dtype
        @param stream: input stream.
        @type stream: file
        @return: resulted array.
        @rtype: numpy.ndarray
        """
        import bz2
        import zlib
        import numpy as np
        length = shape[0]
        for dim in shape[1:]:
            length *= dim
        dobj = dtype()
        if flag_compress == 'bz2':
            buflen = np.frombuffer(stream.read(8), dtype=np.int64)[0]
            buf = stream.read(buflen)
            buf = bz2.decompress(buf)
        elif flag_compress == 'gz':
            buflen = np.frombuffer(stream.read(8), dtype=np.int64)[0]
            buf = stream.read(buflen)
            buf = zlib.decompress(buf)
        else:
            buf = stream.read(length * dobj.itemsize)
        arr = np.frombuffer(buf, dtype=dtype).reshape(shape)
        return arr

    def _load_connectivity(self, meta, stream, blk):
        """
        @param meta: meta information dictionary.
        @type meta: solvcon.gendata.AttributeDict
        @param stream: input stream.
        @type stream: file
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @return: nothing.
        """
        from numpy import int32
        blk.shfcnds = self._read_array(meta.flag_compress,
            (meta.ngstface+meta.nface, meta.FCMND+1), int32, stream)
        blk.shfccls = self._read_array(meta.flag_compress,
            (meta.ngstface+meta.nface, 4), int32, stream)
        blk.shclnds = self._read_array(meta.flag_compress,
            (meta.ngstcell+meta.ncell, meta.CLMND+1), int32, stream)
        blk.shclfcs = self._read_array(meta.flag_compress,
            (meta.ngstcell+meta.ncell, meta.CLMFC+1), int32, stream)

    def _load_type(self, meta, stream, blk):
        """
        @param meta: meta information dictionary.
        @type meta: solvcon.gendata.AttributeDict
        @param stream: input stream.
        @type stream: file
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @return: nothing.
        """
        from numpy import int32
        blk.shfctpn = self._read_array(meta.flag_compress,
            (meta.ngstface+meta.nface,), int32, stream)
        blk.shcltpn = self._read_array(meta.flag_compress,
            (meta.ngstcell+meta.ncell,), int32, stream)
        blk.shclgrp = self._read_array(meta.flag_compress,
            (meta.ngstcell+meta.ncell,), int32, stream)

    def _load_metrics(self, meta, stream, blk):
        """
        @param meta: meta information dictionary.
        @type meta: solvcon.gendata.AttributeDict
        @param stream: input stream.
        @type stream: file
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @return: nothing.
        """
        fpdtype = blk.fpdtype
        blk.shndcrd = self._read_array(meta.flag_compress,
            (meta.ngstnode+meta.nnode, meta.ndim), fpdtype, stream)
        blk.shfccnd = self._read_array(meta.flag_compress,
            (meta.ngstface+meta.nface, meta.ndim), fpdtype, stream)
        blk.shfcnml = self._read_array(meta.flag_compress,
            (meta.ngstface+meta.nface, meta.ndim), fpdtype, stream)
        blk.shfcara = self._read_array(meta.flag_compress,
            (meta.ngstface+meta.nface,), fpdtype, stream)
        blk.shclcnd = self._read_array(meta.flag_compress,
            (meta.ngstcell+meta.ncell, meta.ndim), fpdtype, stream)
        blk.shclvol = self._read_array(meta.flag_compress,
            (meta.ngstcell+meta.ncell,), fpdtype, stream)

    def _load_boundcond(self, meta, bcsinfo, fpdtype, stream, blk):
        """
        @param meta: meta information dictionary.
        @type meta: solvcon.gendata.AttributeDict
        @param bcsinfo: list for information tuples for BCs.
        @type bcsinfo: list
        @param fpdtype: fpdtype for the to-be-created BC objects.
        @type fpdtype: numpy.dtype
        @param stream: input stream.
        @type stream: file
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @return: nothing.
        """
        import numpy as np
        from ..boundcond import BC
        blk.bndfcs = self._read_array(meta.flag_compress,
            (meta.nbound, 2), np.int32, stream)
        for bcinfo in bcsinfo:
            sern, name, blkn, flen, nval = bcinfo
            bc = BC(fpdtype=fpdtype)
            bc.sern = sern
            bc.name = name
            bc.blk = blk
            bc.blkn = blkn
            if flen > 0:
                bc.facn = self._read_array(meta.flag_compress,
                    (flen, 3), np.int32, stream)
            if nval > 0:
                bc.value = self._read_array(meta.flag_compress,
                    (flen, nval), fpdtype, stream)
            blk.bclist.append(bc)

    def _convert_bc(self, name_mapper, blk):
        """
        Convert boundary condition object into proper types.
        
        @param name_mapper: map name to bc type and value dictionary; the two
            objects are organized in a tuple.
        @type name_mapper: dict
        @param blk: to-be-written Block object.
        @type blk: solvcon.block.Block
        @return: nothing.
        """
        import warnings
        for ibc in range(len(blk.bclist)):
            bc = blk.bclist[ibc]
            # recreate BC according to name mapping.
            mapper = name_mapper.get(bc.name, None)
            if mapper == None:
                warnings.warn('%s does not have a mapper.' % bc.name,
                    UserWarning)
                continue
            bct, vdict = mapper
            if bct is not None:
                newbc = bct(bc)
                newbc.feedValue(vdict)
            # save to block object.
            newbc.sern = bc.sern
            newbc.blk = blk
            blk.bclist[ibc] = newbc

    def _construct_subarrays(self, meta, blk):
        """
        @param meta: meta information dictionary.
        @type meta: solvcon.gendata.AttributeDict
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @return: nothing.
        """
        # metrics.
        blk.ndcrd = blk.shndcrd[meta.ngstnode:,:]
        blk.fccnd = blk.shfccnd[meta.ngstface:,:]
        blk.fcnml = blk.shfcnml[meta.ngstface:,:]
        blk.fcara = blk.shfcara[meta.ngstface:]
        blk.clcnd = blk.shclcnd[meta.ngstcell:,:]
        blk.clvol = blk.shclvol[meta.ngstcell:]
        # type data.
        blk.fctpn = blk.shfctpn[meta.ngstface:]
        blk.cltpn = blk.shcltpn[meta.ngstcell:]
        # ghost connectivities.
        blk.fcnds = blk.shfcnds[meta.ngstface:,:]
        blk.fccls = blk.shfccls[meta.ngstface:,:]
        blk.clnds = blk.shclnds[meta.ngstcell:,:]
        blk.clfcs = blk.shclfcs[meta.ngstcell:,:]
        # descriptive data.
        blk.clgrp = blk.shclgrp[meta.ngstcell:]

        # ghost metrics.
        blk.gstndcrd = blk.shndcrd[meta.ngstnode-1::-1,:]
        blk.gstfccnd = blk.shfccnd[meta.ngstface-1::-1,:]
        blk.gstfcnml = blk.shfcnml[meta.ngstface-1::-1,:]
        blk.gstfcara = blk.shfcara[meta.ngstface-1::-1]
        blk.gstclcnd = blk.shclcnd[meta.ngstcell-1::-1,:]
        blk.gstclvol = blk.shclvol[meta.ngstcell-1::-1]
        # ghost type data.
        blk.gstfctpn = blk.shfctpn[meta.ngstface-1::-1]
        blk.gstcltpn = blk.shcltpn[meta.ngstcell-1::-1]
        # ghost connectivities.
        blk.gstfcnds = blk.shfcnds[meta.ngstface-1::-1,:]
        blk.gstfccls = blk.shfccls[meta.ngstface-1::-1,:]
        blk.gstclnds = blk.shclnds[meta.ngstcell-1::-1,:]
        blk.gstclfcs = blk.shclfcs[meta.ngstcell-1::-1,:]
        # ghost descriptive data.
        blk.gstclgrp = blk.shclgrp[meta.ngstcell-1::-1]
