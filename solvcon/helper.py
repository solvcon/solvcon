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
Helping functionalities.
"""

class Printer(object):
    """
    Print message to a stream.

    >>> import StringIO
    >>> output = StringIO.StringIO()
    >>> mesg = Printer(output)
    >>> mesg('mesg')
    >>> mesg(int, float) # object
    >>> output.getvalue()
    "mesg<type 'int'> <type 'float'>"
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

    def __call__(self, *args):
        msg = ' '.join([str(it) for it in args])
        msg = ''.join([self.prefix, msg, self.postfix])
        for stream in self.streams:
            stream.write(msg)
            if self.force_flush:
                stream.flush()

class Information(object):
    """
    Information logger.
    """

    def __init__(self, prefix='*', nchar=4, width=80, level=0, muted=False):
        self.prefix = prefix
        self.nchar = nchar
        self.width = width
        self.level = level
        self.muted = muted

    @property
    def streams(self):
        """
        :type: list

        List of output streams.
        """
        import sys
        from .conf import env
        if self.muted: return []
        streams = [sys.stdout]
        if env.logfile != None: streams.append(env.logfile)
        return streams

    def __call__(self, data, travel=0, level=None, has_gap=True):
        """
        :param data: String data to be output.
        :type data: str

        Output.
        """
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
    :return: Flag under windows or not.
    :rtype: bool
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
    :param loc: The location to start searching.
    :type loc: str
    :param name: The searching target.
    :type name: str
    :return: The absolute path to the FS item.
    :rtype: str

    Search for something in the file system all the way up from the specified
    location to the root.
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
    :ivar cmds: Commands to be sent to Cubit.
    :type cmds: list
    :ivar ndim: Number of dimensions.
    :type ndim: int
    :ivar large: Flag to use large file of Genesis/ExodusII or not.
    :type large: bool

    Delegate Cubit command through journaling file and load the generated mesh.
    """

    def __init__(self, cmds, ndim, large=False):
        self.cmds = cmds
        self.ndim = ndim
        self.large = large
        self.stdout = None

    def __call__(self):
        """
        :return: The loaded Genesis object.
        :rtype: solvcon.io.genesis.Genesis

        Launch Cubit for generating mesh and then load the generated
        Genesis/ExodusII file.
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
    :ivar cmds: Commands to be sent to gmsh.
    :type cmds: list

    Delegate Gmsh command through journaling file and load the generated mesh.
    """
    def __init__(self, cmds, preserve=False):
        """
        >>> gmh = Gmsh(["lc = 0.1;"])
        >>> gmh = Gmsh(["lc = 0.1;"], preserve=True)
        """
        self.cmds = cmds
        self.stdout = None
        self.preserve = preserve
    def __call__(self, options=None):
        """
        :return: The loaded Gmsh object.
        :rtype: solvcon.io.gmsh.Gmsh

        Launch Gmsh for generating mesh and then load the generated file.
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
        if None is not options:
            cli += ' %s' % options
        pobj = Popen(cli, shell=True, stdout=PIPE, stderr=STDOUT)
        self.stdout = pobj.stdout.read()
        if not os.path.exists(mshp):
            raise OSError(
                '%s not produced by gmsh command line, stdout:\n%s' % (mshp,
                    self.stdout))
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
            if self.preserve:
                shutil.copyfile(cmdp, os.path.basename(cmdp))
                shutil.copyfile(mshp, os.path.basename(mshp))
            shutil.rmtree(wdir)
        return gmh
