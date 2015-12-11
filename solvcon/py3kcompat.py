# -*- coding: UTF-8 -*-
#
# Copyright (c) 2015, Yung-Yu Chen <yyc@solvcon.net>
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
# - Neither the name of the copyright holder nor the names of its contributors
#   may be used to endorse or promote products derived from this software
#   without specific prior written permission.
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


from __future__ import absolute_import, division, print_function


__all__ = [
    'with_metaclass',
    'assertRaisesRegex',
    'basestring', # class
    'StringIO', # class
    'ConfigParser', # class
    'pickle', # module
    'TemporaryDirectory', # class
]


from six import (
    with_metaclass,
    assertRaisesRegex,
)


try: # py3k compat.
    basestring = basestring
except NameError:
    class _BaseStringType(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, (bytes, str))
    class basestring(with_metaclass(_BaseStringType)): pass

try: # py3k compat.
    from cStringIO import StringIO
except ImportError:
    from io import StringIO

try: # py3k compat.
    from ConfigParser import ConfigParser
except ImportError:
    from configparser import ConfigParser

try: # py3k compat.
    import cPickle as pickle
except:
    import pickle


import os as _os
import warnings as _warnings
import tempfile as _tempfile
import shutil as _shutil

class _TemporaryDirectory(object):
    """Create and return a temporary directory.  This has the same
    behavior as mkdtemp but can be used as a context manager.  For
    example:

        with TemporaryDirectory() as tmpdir:
            ...

    Upon exiting the context, the directory and everything contained
    in it are removed.

    This class is taken from Python 3.3 for Python 2.7 compatibility.
    """

    # Handle mkdtemp raising an exception
    name = None
    _closed = False

    def __init__(self, suffix="", prefix="tmp", dir=None):
        self.name = _tempfile.mkdtemp(suffix, prefix, dir)

    def __repr__(self):
        return "<{} {!r}>".format(self.__class__.__name__, self.name)

    def __enter__(self):
        return self.name

    def cleanup(self, _warn=False, _warnings=_warnings):
        if self.name and not self._closed:
            try:
                _shutil.rmtree(self.name)
            except (TypeError, AttributeError) as ex:
                if "None" not in '%s' % (ex,):
                    raise
                self._rmtree(self.name)
            self._closed = True
            if _warn and _warnings.warn:
                try:
                    _warnings.warn("Implicitly cleaning up {!r}".format(self),
                                   ResourceWarning)
                except:
                    if _is_running:
                        raise
                    # Don't raise an exception if modules needed for emitting
                    # a warning are already cleaned in shutdown process.

    def __exit__(self, exc, value, tb):
        self.cleanup()

    def __del__(self):
        # Issue a ResourceWarning if implicit cleanup needed
        self.cleanup(_warn=True)

    def _rmtree(self, path, _OSError=OSError, _sep=_os.path.sep,
                _listdir=_os.listdir, _remove=_os.remove, _rmdir=_os.rmdir):
        # Essentially a stripped down version of shutil.rmtree.  We can't
        # use globals because they may be None'ed out at shutdown.
        if not isinstance(path, str):
            _sep = _sep.encode()
        try:
            for name in _listdir(path):
                fullname = path + _sep + name
                try:
                    _remove(fullname)
                except _OSError:
                    self._rmtree(fullname)
            _rmdir(path)
        except _OSError:
            pass

try: # py3k compat.
    from tempfile import TemporaryDirectory
except ImportError:
    TemporaryDirectory = _TemporaryDirectory
