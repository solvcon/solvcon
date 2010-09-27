# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2010 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
Intrinsic format mesh I/O.  Provides:
  - TrivialBlockFormat (revision 0.0.0.1).
"""

from ..gendata import SingleAssignDict, AttributeDict

class BlockFormatRegistry(SingleAssignDict, AttributeDict):
    def register(self, blftype):
        name = blftype.__name__
        rev = blftype.FORMAT_REV
        self[name] = self[rev] = blftype
        return blftype
blfregy = BlockFormatRegistry() # registry singleton.

class BlockFormatMeta(type):
    """
    Sum the length of all META_ entries of bases and derived classes; only 
    collect from 1 level of parents.
    """
    def __new__(cls, name, bases, namespace):
        mldict = dict()
        # collect length from base classes.  later overrides former.
        # NOTE: only 1 level of parents is considered.
        for base in bases:
            for key in base.__dict__:
                if key.startswith('META_'):
                    mldict[key] = len(getattr(base, key))
        # collect length from derived class, and override bases.
        for key in namespace:
            if key.startswith('META_'):
                mldict[key] = len(namespace[key])
        # sum them all.
        namespace['meta_length'] = sum([mldict[k] for k in mldict])
        # recreate the class.
        newcls = super(BlockFormatMeta, cls).__new__(cls, name, bases, namespace)
        # register.
        blfregy.register(newcls)
        return newcls

class BlockFormat(object):
    """
    Abstract class for fundamental facilities for I/O intrinsic mesh format
    (blk).  A blk file is in general composed by (i) meta-data, (ii) group
    list, (iii) boundary condition list, and (iv) binary arrays.  The saved
    arrays can be compressed.  The compression must be applied to all or none
    of the arrays.  Each of the concrete derived classes represents a version
    of format, and should override (i) save() and (ii) load() methods.

    There is only one object method: _save_meta(), since it needs to access
    object for meta-data.  All other methods are either class methods or static
    methods.

    @cvar READ_BLOCK: length per reading from file.
    @ctype READ_BLOCK: int
    @cvar BINARY_MARKER: a string to mark the starting of binary data.
    @ctype BINARY_MARKER: str
    @cvar FILE_HEADER: header of blk file.
    @ctype FILE_HEADER: str
    @cvar FORMAT_REV: revision of format; must be overridden.
    @ctype FORMAT_REV: str

    @cvar META_GLOBAL: global meta entries.
    @ctype META_GLOBAL: tuple
    @cvar META_DESC: descriptive meta entries.
    @ctype META_DESC: tuple
    @cvar META_GEOM: geometric meta entries.
    @ctype META_GEOM: tuple
    @cvar META_SWITCH: optional flags.
    @ctype META_SWITCH: tuple
    @cvar META_ATT: attached meta entries.
    @ctype META_ATT: tuple

    @cvar meta_length: length of all META_ entries.
    @ctype meta_length: int

    @ivar flag_compress: the compression to use: '', 'gz', or 'bz2'
    @itype flag_compress: str
    @ivar fpdtype: specified fpdtype for I/O.
    @itype fpdtype: numpy.dtype
    """

    __metaclass__ = BlockFormatMeta

    READ_BLOCK = 1024
    FILE_HEADER = '-*- solvcon blk mesh file -*-'
    BINARY_MARKER = '-*- start of binary data -*-'
    FORMAT_REV = None

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
    META_SWITCH = tuple()
    META_ATT = (
        'ngroup', 'nbc',
    )

    def __init__(self, **kw):
        self.flag_compress = kw.pop('flag_compress', '')
        self.fpdtype = kw.pop('fpdtype', None)
        super(BlockFormat, self).__init__()
    def read_meta(self, stream):
        """
        Read meta-data of blk file from stream.
        
        @param stream: file object or file name to be read.
        @type stream: file or str
        @return: meta-data, raw text lines of meta-data, and the length of
            meta-data in bytes.
        @rtype: solvcon.gendata.AttributeDict, list, int
        """
        return self._parse_meta(self._get_textpart(stream)[0])
    def save(self, blk, stream):
        """
        Save the block object into a file.
        
        @param blk: to-be-written block object.
        @type blk: solvcon.block.Block
        @param stream: file object or file name to be read.
        @type stream: file or str
        """
        raise NotImplementedError
    def load(self, stream, bcmapper):
        """
        Load block from stream with BC mapper applied.
        
        @param stream: file object or file name to be read.
        @type stream: file or str
        @param bcmapper: BC type mapper.
        @type bcmapper: dict
        @return: the read block object.
        @rtype: solvcon.block.Block
        """
        raise NotImplementedError

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
        # global.
        for key in self.META_GLOBAL:
            stream.write('%s = %s\n' % (key, str(getattr(self, key))))
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
    def _save_group_and_bclist(cls, blk, stream):
        """
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @param stream: output stream.
        @type stream: file
        @return: nothing.
        """
        ## group.
        for ig in range(len(blk.grpnames)):
            stream.write('group%d = %s\n' % (ig, blk.grpnames[ig]))
        ## boundary conditions.
        for ibc in range(len(blk.bclist)):
            bc = blk.bclist[ibc]
            stream.write('bc%d = %s, %s, %d, %d\n' % (
                bc.sern, bc.name, str(bc.blkn), len(bc), bc.nvalue,
            ))
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
        import bz2, zlib, struct
        if flag_compress == 'bz2':
            data = bz2.compress(arr.data, 9)
            stream.write(struct.pack('q', len(data)))
        elif flag_compress == 'gz':
            data = zlib.compress(arr.data, 9)
            stream.write(struct.pack('q', len(data)))
        else:
            data = arr.data
        stream.write(data)
    @classmethod
    def _save_boundcond(cls, flag_compress, blk, stream):
        """
        @param blk: block object to alter.
        @type blk: solvcon.block.Block
        @param stream: output stream.
        @type stream: file
        @return: nothing.
        """
        cls._write_array(flag_compress, blk.bndfcs, stream)
        for bc in blk.bclist:
            if len(bc) > 0:
                cls._write_array(flag_compress, bc.facn, stream)
            if bc.value.shape[1] > 0:
                cls._write_array(flag_compress, bc.value, stream)

    ############################################################################
    # Facilities for reading.
    ############################################################################
    @classmethod
    def _get_textpart(cls, stream):
        """
        Retrieve the text part from the stream.
        
        @param stream: input stream.
        @type stream: file
        @return: text lines and length of text part (including separator).
        @rtype: tuple
        """
        buf = ''
        while cls.BINARY_MARKER not in buf:
            buf += stream.read(cls.READ_BLOCK)
        buf = buf.split(cls.BINARY_MARKER)[0]
        lines = buf.split('\n')[:-1]
        textlen = len(buf) + len(cls.BINARY_MARKER) + 1
        # assert text format.
        assert lines[0].strip()[:3] == '-*-'
        assert lines[0].strip()[-3:] == '-*-'
        # assert text end location.
        stream.seek(0)
        assert stream.read(textlen)[-1] == '\n'
        stream.seek(0)
        return lines, textlen
    @staticmethod
    def _parse_bool(val):
        try:
            return bool(int(val))
        except:
            val = val.lower()
            if val == 'true' or val == 'on':
                return True
            elif val == 'false' or val == 'off':
                return False
        finally:
            return None
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
        from ..gendata import AttributeDict
        meta = AttributeDict()
        # parse text.
        ## global.
        begin = 1
        end = begin + len(cls.META_GLOBAL)
        for line in lines[begin:end]:
            key, val = [to.strip() for to in line.split('=')]
            meta[key] = val
        ## type.
        begin = end
        end = begin + len(cls.META_DESC)
        for line in lines[begin:end]:
            key, val = [to.strip() for to in line.split('=')]
            meta[key] = val
        ## geometry.
        begin = end
        end = begin + len(cls.META_GEOM)
        for line in lines[begin:end]:
            key, val = [to.strip() for to in line.split('=')]
            meta[key] = int(val)
        ## switches.
        begin = end
        end = begin + len(cls.META_SWITCH)
        for line in lines[begin:end]:
            key, val = [to.strip() for to in line.split('=')]
            meta[key] = cls._parse_bool(val)
        ## attached.
        begin = end
        end = begin + len(cls.META_ATT)
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
    @classmethod
    def _load_group_and_bclist(cls, meta, lines, blk):
        """
        @return: meta information dictionary.
        @rtype: solvcon.gendata.AttributeDict
        @param lines: text data
        @type lines: list
        @param blk: Block object to load to.
        @type blk: solvcon.block.Block
        @return: BC information.
        @rtype: list
        """
        # groups.
        meta_length = cls.meta_length
        begin = meta_length + 1
        end = meta_length + 1 + meta.ngroup
        blk.grpnames = [line.split('=')[-1].strip()
            for line in lines[begin:end]]
        # BC object information.
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
        arr = np.frombuffer(buf, dtype=dtype).reshape(shape).copy()
        return arr
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
        from numpy import int32
        blk.shfcnds = cls._read_array(meta.flag_compress,
            (meta.ngstface+meta.nface, meta.FCMND+1), int32, stream)
        blk.shfccls = cls._read_array(meta.flag_compress,
            (meta.ngstface+meta.nface, 4), int32, stream)
        blk.shclnds = cls._read_array(meta.flag_compress,
            (meta.ngstcell+meta.ncell, meta.CLMND+1), int32, stream)
        blk.shclfcs = cls._read_array(meta.flag_compress,
            (meta.ngstcell+meta.ncell, meta.CLMFC+1), int32, stream)
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
        from numpy import int32
        blk.shfctpn = cls._read_array(meta.flag_compress,
            (meta.ngstface+meta.nface,), int32, stream)
        blk.shcltpn = cls._read_array(meta.flag_compress,
            (meta.ngstcell+meta.ncell,), int32, stream)
        blk.shclgrp = cls._read_array(meta.flag_compress,
            (meta.ngstcell+meta.ncell,), int32, stream)
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
        blk.shndcrd = cls._read_array(meta.flag_compress,
            (meta.ngstnode+meta.nnode, meta.ndim), fpdtype, stream)
        if extra:
            blk.shfccnd = cls._read_array(meta.flag_compress,
                (meta.ngstface+meta.nface, meta.ndim), fpdtype, stream)
            blk.shfcnml = cls._read_array(meta.flag_compress,
                (meta.ngstface+meta.nface, meta.ndim), fpdtype, stream)
            blk.shfcara = cls._read_array(meta.flag_compress,
                (meta.ngstface+meta.nface,), fpdtype, stream)
            blk.shclcnd = cls._read_array(meta.flag_compress,
                (meta.ngstcell+meta.ncell, meta.ndim), fpdtype, stream)
            blk.shclvol = cls._read_array(meta.flag_compress,
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
        import numpy as np
        from ..boundcond import BC
        blk.bndfcs = cls._read_array(meta.flag_compress,
            (meta.nbound, 2), np.int32, stream)
        for bcinfo in bcsinfo:
            sern, name, blkn, flen, nval = bcinfo
            bc = BC(fpdtype=fpdtype)
            bc.sern = sern
            bc.name = name
            bc.blk = blk
            bc.blkn = blkn
            if flen > 0:
                bc.facn = cls._read_array(meta.flag_compress,
                    (flen, 3), np.int32, stream)
            if nval > 0:
                bc.value = cls._read_array(meta.flag_compress,
                    (flen, nval), fpdtype, stream)
            blk.bclist.append(bc)

    ############################################################################
    # Facilities for conversion.
    ############################################################################
    @staticmethod
    def _convert_bc(name_mapper, blk):
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
        from ..boundcond import periodic
        # match periodic BCs.
        nmidx = dict([(blk.bclist[idx].name, idx) for idx in
            range(len(blk.bclist))])
        npidx = list()
        for key in name_mapper:
            bct, vdict = name_mapper[key]
            if not issubclass(bct, periodic):
                npidx.append(nmidx[key])
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

class TrivialBlockFormat(BlockFormat):
    """
    Simplest blk format.
    """
    FORMAT_REV = '0.0.0.1'
    def save(self, blk, stream):
        # text part.
        stream.write(self.FILE_HEADER + '\n')
        self._save_meta(blk, stream)
        self._save_group_and_bclist(blk, stream)
        # binary part.
        stream.write(self.BINARY_MARKER+'\n')
        ## connectivity.
        for key in 'shfcnds', 'shfccls', 'shclnds', 'shclfcs':
            self._write_array(self.flag_compress, getattr(blk, key), stream)
        ## type.
        for key in 'shfctpn', 'shcltpn', 'shclgrp':
            self._write_array(self.flag_compress, getattr(blk, key), stream)
        ## geometry.
        for key in ('shndcrd', 'shfccnd', 'shfcnml', 'shfcara',
            'shclcnd', 'shclvol'):
            self._write_array(self.flag_compress, getattr(blk, key), stream)
        ## boundary conditions.
        self._save_boundcond(self.flag_compress, blk, stream)
    def load(self, stream, bcmapper):
        from ..block import Block
        # determine the text part and binary part.
        lines, textlen = self._get_textpart(stream)
        # create meta-data dict and block object.
        meta = self._parse_meta(lines)
        fpdtype = meta.fpdtype if self.fpdtype == None else self.fpdtype
        blk = Block(fpdtype=fpdtype)
        blk.blkn = meta.blkn
        # load group and BC list.
        bcsinfo = self._load_group_and_bclist(meta, lines, blk)
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

class BlockIO(object):
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

    @ivar flag_compress: the compression to use: '', 'gz', or 'bz2'
    @itype flag_compress: str
    @ivar fpdtype: specified fpdtype for I/O.
    @itype fpdtype: numpy.dtype
    """
    def __init__(self, **kw):
        self.blk = kw.pop('blk', None)
        self.bcmapper = kw.pop('bcmapper', None)
        self.filename = kw.pop('filename', None)
        fmt = kw.pop('fmt', 'TrivialBlockFormat')
        fpdtype = kw.pop('fpdtype', None)
        flag_compress = kw.pop('flag_compress', '')
        super(BlockIO, self).__init__()
        # create BlockFormat object.
        self.blf = blfregy[fmt](flag_compress=flag_compress,
            fpdtype=fpdtype)
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
        self.blf.save(blk=blk, stream=stream)
    def read_meta(self, stream=None):
        """
        Read meta-data of blk file from stream.
        
        @keyword stream: file object or file name to be read.
        @type stream: file or str
        @return: meta-data, raw text lines of meta-data, and the length of
            meta-data in bytes.
        @rtype: solvcon.gendata.AttributeDict, list, int
        """
        return self.blf.read_meta(stream=stream)
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
        return self.blf.load(stream=stream, bcmapper=bcmapper)
