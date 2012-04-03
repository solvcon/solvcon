# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2011 Yung-Yu Chen <yyc@solvcon.net>.
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

def search_in_parents(loc, name):
    """
    Search for something in the file system all the way up from the specified
    location to the root.

    @param loc: the location to start searching.
    @type loc: str
    @param name: the searching target.
    @type name: str
    @return: the absolute path to the FS item.
    @rtype: str
    """
    import os
    item = ''
    loc = os.path.abspath(loc)
    while True:
        if os.path.exists(os.path.join(loc, name)):
            item = os.path.join(loc, name)
            break
        parent = os.path.dirname(loc)
        if loc == parent:
            break
        else:
            loc = parent
    return os.path.abspath(item) if item else item

class Cubit(object):
    """
    Delegate Cubit command through journaling file and load the generated mesh.

    @ivar cmds: commands to be sent to Cubit.
    @itype cmds: list
    @ivar ndim: number of dimensions.
    @itype ndim: int
    @ivar large: use large file of Genesis/ExodusII or not.
    @itype large: bool
    """
    def __init__(self, cmds, ndim, large=False):
        self.cmds = cmds
        self.ndim = ndim
        self.large = large
        self.stdout = None
    def __call__(self):
        """
        Launch Cubit for generating mesh and then load the generated
        Genesis/ExodusII file.

        @return: the loaded Genesis object.
        @rtype: solvcon.io.genesis.Genesis
        """
        from tempfile import mkdtemp
        import os, shutil
        from subprocess import Popen, PIPE, STDOUT
        from .io.genesis import Genesis
        # prepare working directory.
        wdir = mkdtemp()
        joup = os.path.join(wdir, 'jou.jou')
        gnp = os.path.join(wdir, 'gn.g')
        # prepare journaling file.
        cmds = self.cmds[:]
        cmds.insert(0, 'cd "%s"' % wdir)
        cmds.append('set large exodus file %s' % 'on' if self.large else 'off')
        cmds.append('export genesis "%s" dimension %d overwrite' % (
            gnp, self.ndim))
        jouf = open(joup, 'w')
        jouf.write('\n'.join(cmds))
        jouf.close()
        # call Cubit and get the data.
        cli = 'cubit -nographics -batch -nojournal -input %s' % joup
        try:
            pobj = Popen(cli, shell=True, stdout=PIPE, stderr=STDOUT)
            self.stdout = pobj.stdout.read()
            gn = Genesis(gnp)
            gn.load()
            gn.close_file()
        except:
            gn = None
        finally:
            shutil.rmtree(wdir)
        return gn

class Gmsh(object):
    """
    Delegate Gmsh command through journaling file and load the generated mesh.

    @ivar cmds: commands to be sent to gmsh.
    @itype cmds: list
    """
    def __init__(self, cmds):
        self.cmds = cmds
        self.stdout = None
    def __call__(self):
        """
        Launch Gmsh for generating mesh and then load the generated file.

        @return: the loaded Gmsh object.
        @rtype: solvcon.io.gmsh.Gmsh
        """
        from tempfile import mkdtemp
        import os, shutil
        from subprocess import Popen, PIPE, STDOUT
        from .io.gmsh import Gmsh
        # prepare working directory.
        wdir = mkdtemp()
        cmdp = os.path.join(wdir, 'gmsh.geo')
        mshp = os.path.join(wdir, 'gmsh.msh')
        # prepare journaling file.
        cmds = self.cmds[:]
        cmdf = open(cmdp, 'w')
        cmdf.write('\n'.join(cmds))
        cmdf.write('\n')
        cmdf.close()
        # call Gmsh and get the data.
        cli = 'gmsh %s -3 -o %s' % (cmdp, mshp)
        pobj = Popen(cli, shell=True, stdout=PIPE, stderr=STDOUT)
        self.stdout = pobj.stdout.read()
        if not os.path.exists(mshp):
            raise OSError('%s not produced by gmsh command line'%mshp)
        # get the data.
        try:
            mshf = open(mshp)
            gmh = Gmsh(mshf)
            gmh.load()
            mshf.close()
        except:
            gmh = None
            raise
        finally:
            shutil.rmtree(wdir)
        return gmh
