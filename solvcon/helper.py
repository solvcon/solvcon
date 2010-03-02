# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2009 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
Helping functionalities.
"""

class Printer(object):
    """
    Print message to a stream.

    @ivar _streams: list of (stream, filename) tuples to be used..
    @itype _streams: list
    """

    def __init__(self, streams, prefix='', postfix='', override=False):
        import sys, os
        import warnings
        self.prefix = prefix
        self.postfix = postfix
        self.override = override
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
            stream.flush()

def info(data):
    """
    Helper for command modules/functions to output information.
    """
    import sys
    from .conf import env
    sys.stdout.write(data)
    sys.stdout.flush()
    if env.logfile != None:
        env.logfile.write(data)
        env.logfile.flush()

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
