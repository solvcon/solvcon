# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2010 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
Intrinsic format mesh I/O.  Provides:
  - TrivialDomainFormat (revision 0.0.1).
"""

from .core import FormatRegistry, FormatMeta, Format, strbool

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

    SPEC_OF_META = (
        ('GLOBAL', str),
        ('CUT', int),
        ('BLOCK', str),
        ('SWITCH', strbool),
    )
    META_GLOBAL = ('FORMAT_REV', 'compressor',)
    META_CUT = ('edgecut', 'npart',)
    META_BLOCK = ('blk_format_rev',)
    META_SWITCH = tuple()

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
    def save(self, dom, stream):
        """
        Save the dom object into a file.
        
        @param dom: to-be-written domain object; must be split.
        @type dom: solvcon.domain.Collective
        @param stream: file object or file name to be read.
        @type stream: file or str
        """
        raise NotImplementedError
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
        raise NotImplementedError

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
        # global.
        for key in self.META_GLOBAL:
            skey = key
            if hasattr(key, '__iter__'):
                key, skey = key
            stream.write('%s = %s\n' % (key, str(getattr(self, skey))))
        # cut.
        stream.write('%s = %d\n' % ('edgecut', dom.edgecut))
        stream.write('%s = %d\n' % ('npart', len(dom)))
        # optional switches.
        for key in self.META_SWITCH:
            stream.write('%s = %s\n' % (key, str(getattr(self, key))))
    @staticmethod
    def _save_block_filenames(dirname, dom, stream):
        import os
        stream.write('%s = %s\n' % ('whole',
            os.path.join(dirname, 'whole.blk')))
        for iblk in range(len(dom)):
            stream.write('%s = %s\n' % ('part%d'%iblk,
                os.path.join(dirname, 'part%d.blk'%iblk)))
    def _save_blocks(self, dom, dirname):
        import os
        from .block import blfregy
        # block format.
        blf = blfregy[self.blk_format_rev](compressor=self.compressor)
        # save whole.
        stream = open(os.path.join(dirname, 'whole.blk'), 'wb')
        blf.save(dom.blk, stream)
        stream.close()
        # save split blocks.
        for iblk in range(len(dom)):
            stream = open(os.path.join(dirname, 'part%d.blk'%iblk), 'wb')
            blf.save(dom[iblk], stream)
            stream.close()

    ############################################################################
    # Facilities for reading.
    ############################################################################
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
        end = meta_length + 1
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
    def save(self, dom, dirname):
        import os
        stream = open(os.path.join(dirname, self.DOM_FILENAME), 'wb')
        # text part.
        stream.write(self.FILE_HEADER + '\n')
        self._save_meta(dom, stream)
        self._save_block_filenames(dirname, dom, stream)
        stream.write(self.BINARY_MARKER + '\n')
        # binary part.
        # blocks.
        self._save_blocks(dom, dirname)
    def load_block(self, dirname, blkid, bcmapper):
        import os
        from .block import blfregy
        stream = open(os.path.join(dirname, self.DOM_FILENAME), 'rb')
        # determine the text part and binary part.
        lines, textlen = self._get_textpart(stream)
        # create meta-data dict.
        meta = self._parse_meta(lines)
        # load filename.
        whole, split = self._load_block_filename(meta, lines)
        blkfn = whole if blkid == None else split[blkid]
        # load block.
        return blfregy[meta.blk_format_rev]().load(
            open(os.path.join(dirname, blkfn), 'rb'), bcmapper)

class DomainIO(object):
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
        self.dmf = dmfregy[fmt](compressor=compressor, fpdtype=fpdtype)
    def save(self, dom=None, dirname=None):
        """
        Save the block object into a file.
        
        @keyword dom: to-be-written domain object.
        @type dom: solvcon.domain.Domain
        @keyword stream: file object or file name to be read.
        @type stream: file or str
        """
        if blk == None:
            blk = self.blk
        if stream == None:
            stream = open(self.filename, 'wb')
        elif isinstance(stream, str):
            stream = open(stream, 'wb')
        self.dmf.save(dom, stream)
    def read_meta(self, stream=None):
        """
        Read meta-data of dom file from stream.
        
        @keyword stream: file object or file name to be read.
        @type stream: file or str
        @return: meta-data, raw text lines of meta-data, and the length of
            meta-data in bytes.
        @rtype: solvcon.gendata.AttributeDict, list, int
        """
        return self.dmf.read_meta(stream)
    def load_block(self, dirname, blkid, bcmapper=None):
        """
        Load block from stream with BC mapper applied.
        
        @return: the read block object.
        @rtype: solvcon.block.Block
        """
        if stream == None:
            stream = open(self.filename, 'rb')
        elif isinstance(stream, str):
            stream = open(stream, 'rb')
        return self.dmf.load(stream)
