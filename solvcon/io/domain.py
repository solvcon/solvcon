# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2010 Yung-Yu Chen <yyc@solvcon.net>.
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
Intrinsic format mesh I/O.  Provides:
  - TrivialDomainFormat (revision 0.0.1).
"""

from .core import FormatRegistry, FormatMeta, Format, FormatIO, strbool

dmfregy = FormatRegistry() # registry singleton.
class DomainFormatMeta(FormatMeta):
    def __new__(cls, name, bases, namespace):
        # recreate the class.
        newcls = super(DomainFormatMeta, cls).__new__(
            cls, name, bases, namespace)
        # register.
        dmfregy.register(newcls)
        return newcls

class DomainFormat(Format):
    """
    @cvar META_GLOBAL: global meta entries.
    @ctype META_GLOBAL: tuple
    @cvar META_SWITCH: optional flags.
    @ctype META_SWITCH: tuple

    @ivar compressor: the compression to use: '', 'gz', or 'bz2'
    @itype compressor: str
    @ivar blk_format_rev: the format (revision) of block to be saved.
    @itype blk_format_rev: str
    """

    __metaclass__ = DomainFormatMeta

    FILE_HEADER = '-*- solvcon dom file -*-'
    DOM_FILENAME = 'domain.dom'
    WHOLE_FILENAME = 'whole.blk'
    SPLIT_FILENAME = 'part%d.blk'

    SPEC_OF_META = (
        ('GLOBAL', str),
        ('SWITCH', strbool),
        ('SHAPE', int),
        ('IDXINFO', None),
    )
    META_GLOBAL = ('FORMAT_REV', 'compressor', 'blk_format_rev',)
    META_SWITCH = tuple()
    META_SHAPE = ('edgecut', 'nnode', 'nface', 'ncell',
        'npart', 'nifp', 'ndmblk',)

    def __init__(self, **kw):
        self.compressor = kw.pop('compressor', '')
        self.blk_format_rev = kw.pop('blk_format_rev', self.FORMAT_REV)
        super(DomainFormat, self).__init__()
    def read_meta(self, dirname):
        """
        Read meta-data from directory
        
        @param dirname: directory to be read.
        @type dirname: str
        @return: meta-data, raw text lines of meta-data, and the length of
            meta-data in bytes.
        @rtype: solvcon.gendata.AttributeDict
        """
        import os
        stream = open(os.path.join(dirname, self.DOM_FILENAME), 'rb')
        meta = self._parse_meta(self._get_textpart(stream)[0])
        stream.close()
        return meta
    def save(self, dom, dirname):
        """
        Save the dom object into a file.
        
        @param dom: to-be-written domain object; must be split.
        @type dom: solvcon.domain.Collective
        @param dirname: the directory to save data.
        @type dirname: str
        """
        import os
        from .block import blfregy
        stream = open(os.path.join(dirname, self.DOM_FILENAME), 'wb')
        # text part.
        stream.write(self.FILE_HEADER + '\n')
        self._save_meta(dom, stream)
        self._save_idxinfo_shape(dom, stream)
        self._save_block_filenames(dirname, dom, stream)
        stream.write(self.BINARY_MARKER + '\n')
        # binary part.
        self._write_array(self.compressor, dom.part, stream)
        self._write_array(self.compressor, dom.shapes, stream)
        self._write_array(self.compressor, dom.ifparr, stream)
        for maparr in dom.mappers:
            self._write_array(self.compressor, maparr, stream)
        for mynds, myfcs, mycls in dom.idxinfo:
            self._write_array(self.compressor, mynds, stream)
            self._write_array(self.compressor, myfcs, stream)
            self._write_array(self.compressor, mycls, stream)
        stream.close()
        # blocks.
        blf = blfregy[self.blk_format_rev](compressor=self.compressor)
        stream = open(os.path.join(dirname, self.WHOLE_FILENAME), 'wb')
        blf.save(dom.blk, stream)
        stream.close()
        for iblk in range(len(dom)):
            stream = open(
                os.path.join(dirname, self.SPLIT_FILENAME%iblk), 'wb')
            blf.save(dom[iblk], stream)
            stream.close()
    def load(self, dirname, bcmapper, with_split, domaintype):
        """
        Load domain file in the specified directory with BC mapper applied.
        
        @param dirname: directory name of domain.
        @type dirname: str
        @param bcmapper: BC type mapper.
        @type bcmapper: dict
        @param with_split: load split block as well.
        @type with_split: bool
        @param domaintype: the type used to instantiate domain object.
        @type domaintype: solvcon.domain.Collective
        @return: the read domain object.
        @rtype: solvcon.domain.Collective
        """
        import os
        from .block import blfregy
        from ..domain import Collective
        assert issubclass(domaintype, Collective)
        dom = domaintype(None)
        stream = open(os.path.join(dirname, self.DOM_FILENAME), 'rb')
        # determine the text part and binary part.
        lines, textlen = self._get_textpart(stream)
        # create meta-data dict.
        meta = self._parse_meta(lines)
        dom.edgecut = meta.edgecut
        # load idxinfo shape.
        idxlens = self._load_idxinfo_shape(meta, lines)
        # load filename.
        whole, split = self._load_block_filename(meta, lines)
        # load arrays.
        stream.seek(textlen)
        dom.part = self._read_array(meta.compressor,
            (meta.ncell,), 'int32', stream)
        dom.shapes = self._read_array(meta.compressor,
            (meta.npart, 7), 'int32', stream)
        dom.ifparr = self._read_array(meta.compressor,
            (meta.nifp, 2), 'int32', stream)
        ndmaps = self._read_array(meta.compressor,
            (meta.nnode, 1+2*meta.ndmblk), 'int32', stream)
        fcmaps = self._read_array(meta.compressor,
            (meta.nface, 5), 'int32', stream)
        clmaps = self._read_array(meta.compressor,
            (meta.ncell, 2), 'int32', stream)
        dom.mappers = (ndmaps, fcmaps, clmaps)
        idxinfo = list()
        for nnd, nfc, ncl in idxlens:
            mynds = self._read_array(meta.compressor, (nnd,), 'int32', stream)
            myfcs = self._read_array(meta.compressor, (nfc,), 'int32', stream)
            mycls = self._read_array(meta.compressor, (ncl,), 'int32', stream)
            idxinfo.append((mynds, myfcs, mycls))
        dom.idxinfo = tuple(idxinfo)
        stream.close()
        # load blocks.
        blf = blfregy[meta.blk_format_rev]()
        stream = open(os.path.join(dirname, whole), 'rb')
        dom.blk = blf.load(stream, bcmapper)
        stream.close()
        if with_split:
            for sfn in split:
                stream = open(os.path.join(dirname, sfn), 'rb')
                dom.append(blf.load(stream, bcmapper))
                stream.close()
        return dom
    def load_block(self, dirname, blkid, bcmapper):
        """
        Load block file in the specified directory with BC mapper applied.
        
        @param dirname: directory name of domain.
        @type dirname: str
        @param blkid: the id of the block to be loaded.
        @type blkid: int or None
        @param bcmapper: BC type mapper.
        @type bcmapper: dict
        @return: the read block object.
        @rtype: solvcon.block.Block
        """
        import os
        from .block import blfregy
        # load the text part of DOM file for block filename.
        stream = open(os.path.join(dirname, self.DOM_FILENAME), 'rb')
        lines, textlen = self._get_textpart(stream)
        meta = self._parse_meta(lines)
        whole, split = self._load_block_filename(meta, lines)
        stream.close()
        blkfn = whole if blkid == None else split[blkid]
        # load block.
        stream = open(os.path.join(dirname, blkfn), 'rb')
        blk = blfregy[meta.blk_format_rev]().load(stream, bcmapper)
        stream.close()
        return blk

    ############################################################################
    # Facilities for writing.
    ############################################################################
    def _save_meta(self, dom, stream):
        """
        @param dom: partitioned domain object to store.
        @type dom: solvcon.domain.Collective
        @param stream: output stream.
        @type stream: file
        @return: nothing.
        """
        # auto sections.
        for secname in 'GLOBAL', 'SWITCH':
            for key in getattr(self, 'META_'+secname):
                skey = key
                if hasattr(key, '__iter__'):
                    key, skey = key
                stream.write('%s = %s\n' % (key, str(getattr(self, key))))
        # shape.
        stream.write('%s = %d\n' % ('edgecut', dom.edgecut))
        stream.write('%s = %d\n' % ('nnode', dom.blk.nnode))
        stream.write('%s = %d\n' % ('nface', dom.blk.nface))
        stream.write('%s = %d\n' % ('ncell', dom.blk.ncell))
        stream.write('%s = %d\n' % ('npart', len(dom)))
        stream.write('%s = %d\n' % ('nifp', dom.ifparr.shape[0]))
        ndmaps, fcmaps, clmaps = dom.mappers
        assert ndmaps.shape[1]%2 == 1
        stream.write('%s = %d\n' % ('ndmblk', (ndmaps.shape[1]-1)/2))
    @staticmethod
    def _save_idxinfo_shape(dom, stream):
        nblk = len(dom)
        for iblk in range(nblk):
            key = 'idxinfo%d' % iblk
            spe = ' '.join(['%d'%arr.shape[0] for arr in dom.idxinfo[iblk]])
            stream.write('%s = %s\n' % (key, spe))
    @staticmethod
    def _save_block_filenames(dirname, dom, stream):
        import os
        stream.write('%s = %s\n' % ('whole', 'whole.blk'))
        for iblk in range(len(dom)):
            stream.write('%s = %s\n' % ('part%d'%iblk, 'part%d.blk'%iblk))

    ############################################################################
    # Facilities for reading.
    ############################################################################
    @classmethod
    def _load_idxinfo_shape(cls, meta, lines):
        """
        @param meta: meta information dictionary.
        @type meta: solvcon.gendata.AttributeDict
        @param lines: text data
        @type lines: list
        @return: length of indices.
        @rtype: list
        """
        meta_length = cls.meta_length
        begin = meta_length + 1
        end = meta_length + 1 + meta.npart
        idxlens = list()
        for line in lines[begin:end]:
            try:
                toks = line.split('=')[-1].strip().split()
                if len(toks) != 3:
                    raise IndexError('must have 3 tokens')
                idxlens.append([int(tok) for tok in toks])
            except StandardError as e:
                e.value += '; wrong format in the %d-th index length' % len(
                    idxlens)
                raise e
        return idxlens
    @classmethod
    def _load_block_filename(cls, meta, lines):
        """
        @param meta: meta information dictionary.
        @type meta: solvcon.gendata.AttributeDict
        @param lines: text data
        @type lines: list
        @return: filenames for whole and split blocks.
        @rtype: str, list of str
        """
        meta_length = cls.meta_length
        # filename for whole block.
        end = meta_length + 1 + meta.npart
        whole = lines[end].split('=')[-1].strip()
        # filenames for split blocks.
        begin = end + 1
        end = begin + meta.npart
        split = [line.split('=')[-1].strip() for line in lines[begin:end]]
        return whole, split

class TrivialDomainFormat(DomainFormat):
    """
    Simplest dom format.
    """
    FORMAT_REV = '0.0.1'

class DomainIO(FormatIO):
    """
    Proxy to dom directory format.

    @ivar dom: attached block object.
    @itype dom: solvcon.block.Block
    @ivar dirname: directory name of domain.
    @itype dirname: str
    @ivar fmt: name of the format to use; can be either class name or
        revision string.
    @itype fmt: str

    @ivar compressor: the compression to use: '', 'gz', or 'bz2'
    @itype compressor: str
    @ivar dmf: the format class for the domain to be read.
    @itype dmf: DomainFormat
    """
    def __init__(self, **kw):
        self.dom = kw.pop('dom', None)
        self.dirname = kw.pop('dirname', None)
        fmt = kw.pop('fmt', 'TrivialDomainFormat')
        compressor = kw.pop('compressor', '')
        super(DomainIO, self).__init__()
        # create DomainFormat object.
        self.dmf = dmfregy[fmt](compressor=compressor)
    def save(self, dom=None, dirname=None):
        """
        Save the block object into a file.
        
        @keyword dom: to-be-written domain object.
        @type dom: solvcon.domain.Domain
        @keyword dirname: directory name to be read.
        @type dirname: str
        """
        dom = self.dom if dom == None else dom
        dirname = self.dirname if dirname == None else dirname
        self.dmf.save(dom, dirname)
    def read_meta(self, dirname=None):
        """
        Read meta-data of dom file from stream.
        
        @keyword dirname: the directory of domain data.
        @type dirname: str
        @return: meta-data, raw text lines of meta-data, and the length of
            meta-data in bytes.
        @rtype: solvcon.gendata.AttributeDict, list, int
        """
        dirname = self.dirname if dirname == None else dirname
        return self.dmf.read_meta(dirname)
    def load(self, dirname=None, bcmapper=None, with_split=False,
        domaintype=None):
        """
        Load domain from stream with BC mapper applied.
        
        @keyword dirname: directory name to be read.
        @type dirname: str
        @keyword bcmapper: BC type mapper.
        @type bcmapper: dict
        @keyword with_split: load split block as well.
        @type with_split: bool
        @keyword domaintype: the type used to instantiate domain object.
        @type domaintype: type
        @return: the read domain object.
        @rtype: solvcon.domain.Collective
        """
        from .. import domain
        dirname = self.dirname if dirname == None else dirname
        if domaintype is None:
            domaintype = domain.Collective
        elif isinstance(domaintype, basestring):
            domiantype = getattr(domain, domaintype)
        return self.dmf.load(dirname, bcmapper, with_split, domaintype)
    def load_block(self, dirname=None, blkid=None, bcmapper=None):
        """
        Load block from stream with BC mapper applied.
        
        @keyword dirname: directory name to be read.
        @type dirname: str
        @keyword blkid: the id of block to be read.
        @type blkid: int
        @keyword bcmapper: BC type mapper.
        @type bcmapper: dict
        @return: the read block object.
        @rtype: solvcon.block.Block
        """
        dirname = self.dirname if dirname == None else dirname
        return self.dmf.load_block(dirname, blkid, bcmapper)
