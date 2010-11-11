# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2010 Yung-Yu Chen.
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
Helping functionalities.
"""

class Printer(object):
    """
    Print message to a stream.

    @ivar _streams: list of (stream, filename) tuples to be used..
    @itype _streams: list
    """

    def __init__(self, streams, **kw):
        import sys, os
        import warnings
        self.prefix = kw.pop('prefix', '')
        self.postfix = kw.pop('postfix', '')
        self.override = kw.pop('override', False)
        self.force_flush = kw.pop('force_flush', True)
        # build (stream, filename) tuples.
        if not isinstance(streams, list) and not isinstance(streams, tuple):
            streams = [streams]
        self._streams = []
        for stream in streams:
            if isinstance(stream, str):
                if stream == 'sys.stdout':
                    self._streams.append((sys.stdout, None))
                else:
                    if not self.override and os.path.exists(stream):
                        warnings.warn('file %s overriden.' % stream,
                            UserWarning)
                    self._streams.append((open(stream, 'w'), stream))
            else:
                self._streams.append((stream, None))

    @property
    def streams(self):
        return (s[0] for s in self._streams)

    def __call__(self, msg):
        msg = ''.join([self.prefix, msg, self.postfix])
        for stream in self.streams:
            stream.write(msg)
            if self.force_flush:
                stream.flush()

class Information(object):
    def __init__(self, **kw):
        self.prefix = kw.pop('prefix', '*')
        self.nchar = kw.pop('nchar', 4)
        self.width = kw.pop('width', 80)
        self.level = kw.pop('level', 0)
        self.muted = kw.pop('muted', False)
    @property
    def streams(self):
        import sys
        from .conf import env
        if self.muted: return []
        streams = [sys.stdout]
        if env.logfile != None: streams.append(env.logfile)
        return streams
    def __call__(self, data, travel=0, level=None, has_gap=True):
        self.level += travel
        if level == None:
            level = self.level
        width = self.nchar*level
        lines = data.split('\n')
        prefix = self.prefix*(self.nchar-1)
        if width > 0:
            if has_gap:
                prefix += ' '
            else:
                prefix += '*'
        prefix = '\n' + prefix*self.level
        data = prefix.join(lines)
        for stream in self.streams:
            stream.write(data)
            stream.flush()
info = Information()

def generate_apidoc(outputdir='doc/api'):
    """
    Use epydoc to generate API doc.
    """
    import os
    from epydoc.docbuilder import build_doc_index
    from epydoc.docwriter.html import HTMLWriter
    docindex = build_doc_index(['solvcon'], introspect=True, parse=True,
                               add_submodules=True)
    html_writer = HTMLWriter(docindex)
    # write.
    outputdir = os.path.join(*outputdir.split('/'))
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    html_writer.write(outputdir)

def iswin():
    """
    @return: flag under windows or not.
    @rtype: bool
    """
    import sys
    if sys.platform.startswith('win'):
        return True
    else:
        return False

def get_username():
    import os
    try:
        username = os.getlogin()
    except:
        username = None
    if not username:
        username = os.environ['LOGNAME']
    return username
