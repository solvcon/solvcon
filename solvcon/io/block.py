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
Intrinsic format mesh I/O.  Provides:
  - OldTrivialBlockFormat (revision 0.0.0.1).
  - TrivialBlockFormat (revision 0.0.1).
  - IncenterBlockFormat (revision 0.0.7).
"""

from .core import FormatRegistry, FormatMeta, Format, FormatIO, strbool

blfregy = FormatRegistry() # registry singleton.
class BlockFormatMeta(FormatMeta):
    def __new__(cls, name, bases, namespace):
        # recreate the class.
        newcls = super(BlockFormatMeta, cls).__new__(
            cls, name, bases, namespace)
        # register.
        blfregy.register(newcls)
        return newcls

class BlockFormat(Format):
    """
    A class for fundamental facilities for I/O intrinsic mesh format (blk).  A
    blk file is in general composed by (i) meta-data, (ii) group list, (iii)
    boundary condition list, and (iv) binary arrays.  The saved arrays can be
    compressed.  The compression must be applied to all or none of the arrays.
    Each of the derived classes represents a version of format.

    There is only one object method: _save_meta(), since it needs to access
    object for meta-data.  All other methods are either class methods or static
    methods.

    @cvar FILE_HEADER: header of blk file.
    @ctype FILE_HEADER: str

    @cvar meta_length: length of all META_ entries.
    @ctype meta_length: int

    @ivar compressor: the compression to use: '', 'gz', or 'bz2'
    @itype compressor: str
    @ivar fpdtype: specified fpdtype for I/O.
    @itype fpdtype: numpy.dtype
    """

    __metaclass__ = BlockFormatMeta

    FILE_HEADER = '-*- solvcon blk mesh file -*-'

    def __init__(self, **kw):
        self.compressor = kw.pop('compressor', '')
        self.fpdtype = kw.pop('fpdtype', None)
        super(BlockFormat, self).__init__()
    def save(self, blk, stream):
        """
        Save the block object into a file.
        
        @param blk: to-be-written block object.
        @type blk: solvcon.block.Block
        @param stream: file object or file name to be read.
        @type stream: file or str
        """
        # text part.
        stream.write(self.FILE_HEADER + '\n')
        self._save_meta(blk, stream)
        self._save_group(blk, stream)
        self._save_bclist(blk, stream)
        stream.write(self.BINARY_MARKER+'\n')
        # binary part.
        ## connectivity.
        for key in 'shfcnds', 'shfccls', 'shclnds', 'shclfcs':
            self._write_array(self.compressor, getattr(blk, key), stream)
        ## type.
        for key in 'shfctpn', 'shcltpn', 'shclgrp':
            self._write_array(self.compressor, getattr(blk, key), stream)
        ## geometry.
        for key in ('shndcrd', 'shfccnd', 'shfcnml', 'shfcara',
            'shclcnd', 'shclvol'):
            self._write_array(self.compressor, getattr(blk, key), stream)
        ## boundary conditions.
        self._save_boundcond(self.compressor, blk, stream)
    def load(self, stream, bcmapper, only_meta=False):
        """
        Load block from stream with BC mapper applied.
        
        @param stream: file object or file name to be read.
        @type stream: file or str
        @param bcmapper: BC type mapper.
        @type bcmapper: dict
        @keyword only_meta: read only meta data and return.
        @type only_meta: bool
        @return: the read block object.
        @rtype: solvcon.block.Block
        """
        from ..block import Block
        # determine the text part and binary part.
        lines, textlen = self._get_textpart(stream)
        # create meta-data dict and block object.
        meta = self._parse_meta(lines)
        if only_meta:
            return meta
        fpdtype = meta.fpdtype if self.fpdtype == None else self.fpdtype
        kw = {'fpdtype': fpdtype}
        if 'use_incenter' in meta and meta['use_incenter'] is not None:
            kw['use_incenter'] = meta.use_incenter
        blk = Block(**kw)
        blk.blkn = meta.blkn
        # load group and BC list.
        self._load_group(meta, lines, blk)
        bcsinfo = self._load_bclist(meta, lines, blk)
        # load arrays.
        stream.seek(textlen)
        self._load_connectivity(meta, stream, blk)
        self._load_type(meta, stream, blk)
        self._load_geometry(meta, stream, blk)
        self._load_boundcond(meta, bcsinfo, fpdtype, stream, blk)
        # conversion.
        self._construct_subarrays(meta, blk)
        if bcmapper != None:
            self._convert_bc(bcmapper, blk)
        return blk

    ############################################################################
    # Facilities for writing.
    ############################################################################
    @classmethod
    def _save_group(cls, blk, stream):
        """
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @param stream: output stream.
        @type stream: file
        @return: nothing.
        """
        for ig in range(len(blk.grpnames)):
            stream.write('group%d = %s\n' % (ig, blk.grpnames[ig]))
    @classmethod
    def _save_bclist(cls, blk, stream):
        """
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @param stream: output stream.
        @type stream: file
        @return: nothing.
        """
        for ibc in range(len(blk.bclist)):
            bc = blk.bclist[ibc]
            stream.write('bc%d = %s, %s, %d, %d\n' % (
                bc.sern, bc.name, str(bc.blkn), len(bc), bc.nvalue,
            ))
    @classmethod
    def _save_boundcond(cls, compressor, blk, stream):
        """
        @param compressor: the compression to use: '', 'gz', or 'bz2'
        @type compressor: str
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @param stream: file object or file name to be read.
        @type stream: file or str
        @return: nothing.
        """
        cls._write_array(compressor, blk.bndfcs, stream)
        for bc in blk.bclist:
            if len(bc) > 0:
                cls._write_array(compressor, bc.facn, stream)
            if bc.value.shape[1] > 0:
                cls._write_array(compressor, bc.value, stream)

    ############################################################################
    # Facilities for reading.
    ############################################################################
    @classmethod
    def _load_group(cls, meta, lines, blk):
        """
        @param meta: meta information dictionary.
        @type meta: solvcon.gendata.AttributeDict
        @param lines: text data
        @type lines: list
        @param blk: Block object to load to.
        @type blk: solvcon.block.Block
        @return: BC information.
        @rtype: list
        """
        meta_length = cls.meta_length
        begin = meta_length + 1
        end = meta_length + 1 + meta.ngroup
        blk.grpnames = [line.split('=')[-1].strip()
            for line in lines[begin:end]]
    @classmethod
    def _load_bclist(cls, meta, lines, blk):
        """
        @param meta: meta information dictionary.
        @type meta: solvcon.gendata.AttributeDict
        @param lines: text data
        @type lines: list
        @param blk: Block object to load to.
        @type blk: solvcon.block.Block
        @return: BC information.
        @rtype: list
        """
        meta_length = cls.meta_length
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
        return bcsinfo
    @classmethod
    def _load_connectivity(cls, meta, stream, blk):
        """
        @param meta: meta information dictionary.
        @type meta: solvcon.gendata.AttributeDict
        @param stream: input stream.
        @type stream: file
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @return: nothing.
        """
        blk.shfcnds = cls._read_array(meta.compressor,
            (meta.ngstface+meta.nface, meta.FCMND+1), 'int32', stream)
        blk.shfccls = cls._read_array(meta.compressor,
            (meta.ngstface+meta.nface, 4), 'int32', stream)
        blk.shclnds = cls._read_array(meta.compressor,
            (meta.ngstcell+meta.ncell, meta.CLMND+1), 'int32', stream)
        blk.shclfcs = cls._read_array(meta.compressor,
            (meta.ngstcell+meta.ncell, meta.CLMFC+1), 'int32', stream)
    @classmethod
    def _load_type(cls, meta, stream, blk):
        """
        @param meta: meta information dictionary.
        @type meta: solvcon.gendata.AttributeDict
        @param stream: input stream.
        @type stream: file
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @return: nothing.
        """
        blk.shfctpn = cls._read_array(meta.compressor,
            (meta.ngstface+meta.nface,), 'int32', stream)
        blk.shcltpn = cls._read_array(meta.compressor,
            (meta.ngstcell+meta.ncell,), 'int32', stream)
        blk.shclgrp = cls._read_array(meta.compressor,
            (meta.ngstcell+meta.ncell,), 'int32', stream)
    @classmethod
    def _load_geometry(cls, meta, stream, blk, extra=True):
        """
        @param meta: meta information dictionary.
        @type meta: solvcon.gendata.AttributeDict
        @param stream: input stream.
        @type stream: file
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @keyword extra: load extra geometry data.
        @type extra: bool
        @return: nothing.
        """
        fpdtype = blk.fpdtype
        blk.shndcrd = cls._read_array(meta.compressor,
            (meta.ngstnode+meta.nnode, meta.ndim), fpdtype, stream)
        if extra:
            blk.shfccnd = cls._read_array(meta.compressor,
                (meta.ngstface+meta.nface, meta.ndim), fpdtype, stream)
            blk.shfcnml = cls._read_array(meta.compressor,
                (meta.ngstface+meta.nface, meta.ndim), fpdtype, stream)
            blk.shfcara = cls._read_array(meta.compressor,
                (meta.ngstface+meta.nface,), fpdtype, stream)
            blk.shclcnd = cls._read_array(meta.compressor,
                (meta.ngstcell+meta.ncell, meta.ndim), fpdtype, stream)
            blk.shclvol = cls._read_array(meta.compressor,
                (meta.ngstcell+meta.ncell,), fpdtype, stream)
    @classmethod
    def _load_boundcond(cls, meta, bcsinfo, fpdtype, stream, blk):
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
        from ..boundcond import BC
        blk.bndfcs = cls._read_array(meta.compressor,
            (meta.nbound, 2), 'int32', stream)
        for bcinfo in bcsinfo:
            sern, name, blkn, flen, nval = bcinfo
            bc = BC(fpdtype=fpdtype)
            bc.sern = sern
            bc.name = name
            bc.blk = blk
            bc.blkn = blkn
            if flen > 0:
                bc.facn = cls._read_array(meta.compressor,
                    (flen, 3), 'int32', stream)
            if nval > 0:
                bc.value = cls._read_array(meta.compressor,
                    (flen, nval), fpdtype, stream)
            blk.bclist.append(bc)

    ############################################################################
    # Facilities for conversion.
    ############################################################################
    @staticmethod
    def _convert_bc(name_mapper, blk):
        """
        Convert boundary condition object into proper types.  The method can
        process periodic BCs.
        
        @param name_mapper: map name to bc type and value dictionary; the two
            objects are organized in a tuple.
        @type name_mapper: dict
        @param blk: to-be-written Block object.
        @type blk: solvcon.block.Block
        @return: nothing.
        """
        import warnings
        from ..boundcond import periodic
        # match periodic BCs.
        nmidx = dict([(blk.bclist[idx].name, idx) for idx in
            range(len(blk.bclist))])
        npidx = list()
        for key in name_mapper:
            bct, vdict = name_mapper[key]
            if not issubclass(bct, periodic):
                try:
                    if key in nmidx:
                        npidx.append(nmidx[key])
                except Exception as e:
                    args = list(e.args)
                    args.append(str(nmidx))
                    e.args = tuple(args)
                    raise
                continue
            val = vdict['link']
            ibc0 = nmidx[key]
            ibc1 = nmidx[val]
            pbc0 = blk.bclist[ibc0] = bct(blk.bclist[ibc0])
            pbc1 = blk.bclist[ibc1] = bct(blk.bclist[ibc1])
            ref = vdict['ref']
            pbc0.sort(ref)
            pbc1.sort(ref)
            pbc0.couple(pbc1)
            pbc1.couple(pbc0)
        # process non-periodic BCs.
        for ibc in npidx:
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
    @staticmethod
    def _construct_subarrays(meta, blk):
        """
        @param meta: meta information dictionary.
        @type meta: solvcon.gendata.AttributeDict
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @return: nothing.
        """
        # geometry.
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

class OldTrivialBlockFormat(BlockFormat):
    """
    Simplest blk format (old).
    """
    FORMAT_REV = '0.0.0.1'
    SPEC_OF_META = (
        ('GLOBAL', str),
        ('DESC', str),
        ('GEOM', int),
        ('SWITCH', strbool),
        ('ATT', int),
    )
    META_GLOBAL = ('FORMAT_REV', ('flag_compress', 'compressor'),)
    META_DESC = ('blkn', 'fpdtypestr',)
    META_GEOM = (
        'FCMND', 'CLMND', 'CLMFC',
        'ndim', 'nnode', 'nface', 'ncell', 'nbound',
        'ngstnode', 'ngstface', 'ngstcell',
    )
    META_SWITCH = tuple()
    META_ATT = ('ngroup', 'nbc',)
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
            skey = key
            if hasattr(key, '__iter__'):
                key, skey = key
            stream.write('%s = %s\n' % (key, str(getattr(self, skey))))
        # special description and geometry.
        for key in self.META_DESC + self.META_GEOM:
            stream.write('%s = %s\n' % (key, str(getattr(blk, key))))
        # optional switches.
        for key in self.META_SWITCH:
            stream.write('%s = %s\n' % (key, str(getattr(self, key))))
        # attached/META_ATT.
        stream.write('ngroup = %d\n' % len(blk.grpnames))
        stream.write('nbc = %d\n' % len(blk.bclist))
    @classmethod
    def _parse_meta(cls, lines):
        """
        Parse meta information from the file.
        
        @param lines: text part of the file.
        @type lines: list
        @return: meta information dictionary.
        @rtype: solvcon.gendata.AttributeDict
        """
        import numpy as np
        # parse shared text.
        meta = super(BlockFormat, cls)._parse_meta(lines)
        # further process parsed text.
        meta['compressor'] = meta.pop('flag_compress')
        meta['fpdtype'] = getattr(np, meta.pop('fpdtypestr'))
        try:
            meta.blkn = int(meta.blkn)
        except ValueError:
            meta.blkn = None
        return meta

class TrivialBlockFormat(BlockFormat):
    """
    Simplest blk format.
    """
    FORMAT_REV = '0.0.1'
    SPEC_OF_META = (
        ('GLOBAL', str),
        ('SWITCH', strbool),
        ('DESC', str),
        ('GEOM', int),
        ('ATT', int),
    )
    META_GLOBAL = ('FORMAT_REV', 'compressor',)
    META_DESC = ('fpdtypestr',)
    META_GEOM = (
        'blkn', 'FCMND', 'CLMND', 'CLMFC',
        'ndim', 'nnode', 'nface', 'ncell', 'nbound',
        'ngstnode', 'ngstface', 'ngstcell',
    )
    META_SWITCH = tuple()
    META_ATT = ('ngroup', 'nbc',)

    ############################################################################
    # Facilities for writing.
    ############################################################################
    def _save_meta(self, blk, stream):
        """
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @param stream: output stream.
        @type stream: file
        @return: nothing.
        """
        for secname in 'GLOBAL', 'SWITCH':
            for key in getattr(self, 'META_'+secname):
                skey = key
                if hasattr(key, '__iter__'):
                    key, skey = key
                stream.write('%s = %s\n' % (key, str(getattr(self, key))))
        # special description and geometry.
        for key in self.META_DESC + self.META_GEOM:
            stream.write('%s = %s\n' % (key, str(getattr(blk, key))))
        # attached/META_ATT.
        stream.write('ngroup = %d\n' % len(blk.grpnames))
        stream.write('nbc = %d\n' % len(blk.bclist))
    @classmethod
    def _save_bclist(cls, blk, stream):
        """
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @param stream: output stream.
        @type stream: file
        @return: nothing.
        """
        from ..boundcond import interface
        for ibc in range(len(blk.bclist)):
            bc = blk.bclist[ibc]
            dat = '%s, %s, %d, %d' % (bc.name, str(bc.blkn), len(bc),
                bc.nvalue)
            if isinstance(bc, interface):
                dat = ', '.join([dat, str(bc.rblkn)])
            stream.write('bc%d = %s\n' % (bc.sern, dat))
    @classmethod
    def _save_boundcond(cls, compressor, blk, stream):
        """
        @param compressor: the compression to use: '', 'gz', or 'bz2'
        @type compressor: str
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @param stream: file object or file name to be read.
        @type stream: file or str
        @return: nothing.
        """
        from ..boundcond import interface
        cls._write_array(compressor, blk.bndfcs, stream)
        for bc in blk.bclist:
            if len(bc) > 0:
                cls._write_array(compressor, bc.facn, stream)
            if bc.value.shape[1] > 0:
                cls._write_array(compressor, bc.value, stream)
            if isinstance(bc, interface):
                cls._write_array(compressor, bc.rblkinfo, stream)
                cls._write_array(compressor, bc.rclp, stream)

    ############################################################################
    # Facilities for reading.
    ############################################################################
    @classmethod
    def _parse_meta(cls, lines):
        """
        Parse meta information from the file.
        
        @param lines: text part of the file.
        @type lines: list
        @return: meta information dictionary.
        @rtype: solvcon.gendata.AttributeDict
        """
        import numpy as np
        # parse shared text.
        meta = super(BlockFormat, cls)._parse_meta(lines)
        # further process parsed text.
        meta['fpdtype'] = getattr(np, meta.pop('fpdtypestr'))
        return meta
    @classmethod
    def _load_bclist(cls, meta, lines, blk):
        """
        @param meta: meta information dictionary.
        @type meta: solvcon.gendata.AttributeDict
        @param lines: text data
        @type lines: list
        @param blk: Block object to load to.
        @type blk: solvcon.block.Block
        @return: BC information.
        @rtype: list
        """
        meta_length = cls.meta_length
        begin = meta_length + 1 + meta.ngroup
        end = meta_length + 1 + meta.ngroup + meta.nbc
        bcsinfo = []
        for line in lines[begin:end]:
            key, value = line.split('=')
            sern = int(key.strip()[2:])
            tokens = [sern]
            tokens.extend([tok.strip() for tok in value.split(',')])
            try:
                tokens[2] = int(tokens[2])
            except ValueError:
                tokens[2] = None
            for it in range(3,len(tokens)):
                tokens[it] = int(tokens[it])
            bcsinfo.append(tuple(tokens))
        return bcsinfo
    @classmethod
    def _load_boundcond(cls, meta, bcsinfo, fpdtype, stream, blk):
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
        from ..boundcond import BC, interface
        blk.bndfcs = cls._read_array(meta.compressor,
            (meta.nbound, 2), 'int32', stream)
        for bcinfo in bcsinfo:
            if len(bcinfo) == 5:
                bc = BC(fpdtype=fpdtype)
            elif len(bcinfo) == 6:
                bc = interface(fpdtype=fpdtype)
            else:
                raise IndexError('bcinfo has one 5 or 6 elements')
            bc.sern = bcinfo[0]
            bc.name = bcinfo[1]
            bc.blk = blk
            bc.blkn = bcinfo[2]
            if bcinfo[3] > 0:
                bc.facn = cls._read_array(meta.compressor,
                    (bcinfo[3], 3), 'int32', stream)
            if bcinfo[4] > 0:
                bc.value = cls._read_array(meta.compressor,
                    (bcinfo[3], bcinfo[4]), fpdtype, stream)
            if isinstance(bc, interface):
                bc.rblkn = bcinfo[5]
                bc.rblkinfo = cls._read_array(meta.compressor,
                    (6,), 'int32', stream)
                bc.rclp = cls._read_array(meta.compressor,
                    (bcinfo[3], 3), 'int32', stream)
            blk.bclist.append(bc)

class IncenterBlockFormat(TrivialBlockFormat):
    """
    Block format that can indicate incenter.
    """
    FORMAT_REV = '0.0.7'
    SPEC_OF_META = (
        ('GLOBAL', str),
        ('SWITCH', strbool),
        ('DESC', str),
        ('GEOM', int),
        ('FEATURE', strbool),
        ('ATT', int),
    )
    META_FEATURE = ('use_incenter',)

    ############################################################################
    # Facilities for writing.
    ############################################################################
    def _save_meta(self, blk, stream):
        """
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @param stream: output stream.
        @type stream: file
        @return: nothing.
        """
        for secname in 'GLOBAL', 'SWITCH':
            for key in getattr(self, 'META_'+secname):
                skey = key
                if hasattr(key, '__iter__'):
                    key, skey = key
                stream.write('%s = %s\n' % (key, str(getattr(self, key))))
        # special description and geometry.
        for key in self.META_DESC + self.META_GEOM + self.META_FEATURE:
            stream.write('%s = %s\n' % (key, str(getattr(blk, key))))
        # attached/META_ATT.
        stream.write('ngroup = %d\n' % len(blk.grpnames))
        stream.write('nbc = %d\n' % len(blk.bclist))

class BlockIO(FormatIO):
    """
    Proxy to blk file format.

    @ivar blk: attached block object.
    @itype blk: solvcon.block.Block
    @ivar bcmapper: boundary condition object conversion mapper.
    @itype bcmapper: dict
    @ivar filename: filename of the blk file.
    @itype filename: str
    @ivar fmt: name of the format to use; can be either class name or
        revision string.
    @itype fmt: str

    @ivar compressor: the compression to use: '', 'gz', or 'bz2'
    @itype compressor: str
    @ivar fpdtype: specified fpdtype for I/O.
    @itype fpdtype: numpy.dtype
    """
    def __init__(self, **kw):
        self.blk = kw.pop('blk', None)
        self.bcmapper = kw.pop('bcmapper', None)
        self.filename = kw.pop('filename', None)
        fmt = kw.pop('fmt', None)
        fpdtype = kw.pop('fpdtype', None)
        compressor = kw.pop('compressor', '')
        super(BlockIO, self).__init__()
        # create BlockFormat object.
        if fmt == None and self.filename != None:
            fmt = self._peek_revision(self.filename)
        if fmt == None:
            fmt = 'IncenterBlockFormat'
        self.blf = blfregy[fmt](compressor=compressor, fpdtype=fpdtype)
    @staticmethod
    def _peek_revision(filename):
        from .core import Format
        # open file.
        try:
            stream = open(filename, 'rb')
        except:
            return None
        # read text part.
        fmt = Format()
        lines, textlen = fmt._get_textpart(stream)
        stream.close()
        # determine the format revision.
        rev = None
        for line in lines:
            if 'FORMAT_REV' in line:
                rev = line.split('=')[-1].strip()
                break
        return rev
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
        self.blf.save(blk, stream)
    def read_meta(self, stream=None):
        """
        Read meta-data of blk file from stream.
        
        @keyword stream: file object or file name to be read.
        @type stream: file or str
        @return: meta-data, raw text lines of meta-data, and the length of
            meta-data in bytes.
        @rtype: solvcon.gendata.AttributeDict, list, int
        """
        if stream == None:
            stream = open(self.filename, 'rb')
        elif isinstance(stream, str):
            stream = open(stream, 'rb')
        return self.blf.read_meta(stream)
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
        blf = self.blf
        if stream is None:
            stream = open(self.filename, 'rb')
        elif isinstance(stream, basestring):
            # guess for file format.
            fmt = self._peek_revision(stream)
            if fmt == None:
                fmt = 'IncenterBlockFormat'
            blf = blfregy[fmt]()
            stream = open(stream, 'rb')
        return blf.load(stream, bcmapper)
