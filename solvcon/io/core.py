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
Core I/O facilities for SOLVCON intrinsic constructs.
"""

from ..gendata import TypeNameRegistry

class FormatRegistry(TypeNameRegistry):
    """
    Registry for a certain class of formats.
    """
    def register(self, ftype):
        ftype = super(FormatRegistry, self).register(ftype)
        self[ftype.FORMAT_REV] = ftype
        return ftype

class FormatMeta(type):
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
        # recreate and return the class.
        return super(FormatMeta, cls).__new__(cls, name, bases, namespace)

class Format(object):
    """
    Abstract class for SOLVCON intrinsic I/O format Each of the concrete
    derived classes represents a version of format.  Public interface method is
    read_meta().  read_meta() method uses _parse_meta() private method to
    report the meta-data of the format, which is defined in META_* class
    attributes (as tuples).  The default _parse_meta() and _save_meta() use
    SPEC_OF_META to determine the order and converter the each field of the
    META section.

    The supported intrisic formats must consist of two parts: (i) text part and
    (ii) binary part.  Compression is OK for the binary part.  The text part
    starts at the beginneg of the file and ends after BINARY_MARKER, which the
    binary part starts after BINARY_MARKER and ends with the file.

    @cvar READ_BLOCK: length per reading from file.
    @ctype READ_BLOCK: int
    @cvar FILE_HEADER: header of the format; must be overridden.
    @ctype FILE_HEADER: str
    @cvar BINARY_MARKER: a string to mark the starting of binary data.
    @ctype BINARY_MARKER: str
    @cvar FORMAT_REV: revision of format; must be overridden.
    @ctype FORMAT_REV: str
    @cvar SPEC_OF_META: the order and converter of each meta-data section
        occured in the text part; must be overridden.
    @ctype SPEC_OF_META: tuple of (str, callable)

    @cvar meta_length: length of all META_ entries; calculated by FormatMeta.
    @ctype meta_length: int
    """

    __metaclass__ = FormatMeta

    READ_BLOCK = 1024
    FILE_HEADER = None
    BINARY_MARKER = '-*- start of binary data -*-'
    FORMAT_REV = None
    SPEC_OF_META = None

    def read_meta(self, stream):
        """
        Read meta-data from stream.
        
        @param stream: file object or file name to be read.
        @type stream: file or str
        @return: meta-data, raw text lines of meta-data, and the length of
            meta-data in bytes.
        @rtype: solvcon.gendata.AttributeDict
        """
        return self._parse_meta(self._get_textpart(stream)[0])

    ############################################################################
    # Facilities for writing.
    ############################################################################
    @staticmethod
    def _write_array(compressor, arr, stream):
        """
        @param compressor: how to compress data arrays.
        @type compressor: str
        @param arr: the array to be written.
        @type arr: numpy.ndarray
        @param stream: output stream.
        @type stream: file
        @return: nothing.
        """
        import bz2, zlib, struct
        if compressor == 'bz2':
            data = bz2.compress(arr.data, 9)
            stream.write(struct.pack('q', len(data)))
        elif compressor == 'gz':
            data = zlib.compress(arr.data, 9)
            stream.write(struct.pack('q', len(data)))
        else:
            data = arr.data
        stream.write(data)
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
        end = 1
        for mname, mfunc in cls.SPEC_OF_META:
            # skip everything after the first section with None converter.
            if mfunc == None: break
            # process.
            msec = getattr(cls, 'META_'+mname)
            begin = end
            end = begin + len(msec)
            for line in lines[begin:end]:
                key, val = [to.strip() for to in line.split('=')]
                try:
                    meta[key] = mfunc(val)
                except ValueError:
                    meta[key] = None
        return meta
    @staticmethod
    def _read_array(compressor, shape, dtype, stream, seek_only=False):
        """
        Read data from the input stream and convert it to ndarray with given
        shape and dtype.

        @param compressor: how to compress data arrays.
        @type compressor: str
        @param shape: ndarray shape.
        @type shape: tuple
        @param dtype: ndarray dtype.
        @type dtype: numpy.dtype or str
        @param stream: input stream.
        @type stream: file
        @keyword seek_only: do not really read, only seek; default False.
        @type seek_only: bool
        @return: resulted array.
        @rtype: numpy.ndarray
        """
        import bz2
        import zlib
        import numpy as np
        length = shape[0]
        for dim in shape[1:]:
            length *= dim
        if isinstance(dtype, basestring):
            dtype = getattr(np, dtype)
        dobj = dtype()
        if compressor == 'bz2':
            buflen = np.frombuffer(stream.read(8), dtype=np.int64)[0]
            if seek_only:
                stream.seek(stream.tell() + buflen)
            else:
                buf = stream.read(buflen)
                buf = bz2.decompress(buf)
        elif compressor == 'gz':
            buflen = np.frombuffer(stream.read(8), dtype=np.int64)[0]
            if seek_only:
                stream.seek(stream.tell() + buflen)
            else:
                buf = stream.read(buflen)
                buf = zlib.decompress(buf)
        else:
            buflen = length * dobj.itemsize
            if seek_only:
                stream.seek(stream.tell() + buflen)
            else:
                buf = stream.read(buflen)
        if seek_only:
            arr = None
        else:
            arr = np.frombuffer(buf, dtype=dtype).reshape(shape).copy()
        return arr

fioregy = TypeNameRegistry()    # registry singleton.
class FormatIOMeta(type):
    """
    Metaclass for FormatIO.
    """
    def __new__(cls, name, bases, namespace):
        # recreate the class.
        newcls = super(FormatIOMeta, cls).__new__(
            cls, name, bases, namespace)
        # register.
        fioregy.register(newcls)
        return newcls
class FormatIO(object):
    """
    Proxy to mesh format object.
    """
    __metaclass__ = FormatIOMeta
    def save(self):
        """
        Save an object for mesh.
        """
        raise NotImplementedError
    def load(self, bcmapper=None):
        """
        Load and return an object for mesh.

        @keyword bcmapper: BC type mapper.
        @type bcmapper: dict
        @return: the loaded object for mesh.
        """
        raise NotImplementedError

################################################################################
# Utility
################################################################################
def strbool(val):
    """
    Create bool object from string.  None is returned if input is not 
    derterminable.

    @param val: the string to check.
    @type val: str
    @return: True/False/None.
    @rtype: bool
    """
    try:
        return bool(int(val))
    except:
        val = val.lower()
        if val == 'true' or val == 'on':
            return True
        elif val == 'false' or val == 'off':
            return False
    else:
        return None
