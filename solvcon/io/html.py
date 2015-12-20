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


"""
Mesh I/O for HTML files based on THREE.js.
"""


from __future__ import absolute_import, division, print_function


from ..py3kcompat import NotADirectoryError
import warnings
import os
import shutil
import string
import json

import numpy as np

try:
    import jinja2
except ImportError: # not requiring jinja2 for now.
    pass

from .. import exception
from .. import block
from .. import visual

from . import core as iocore


class HtmlIO(iocore.FormatIO):
    """
    Experimental HTML/THREE.js writer.
    """

    HTMLDIR = os.path.join(
        os.path.abspath(os.path.dirname(__file__)),
        'html'
    )

    JSDIR = os.path.join(
        os.path.abspath(os.path.dirname(visual.__file__)),
        'js'
    )

    EXTERNAL_SCRIPTS = [
        "https://cdnjs.cloudflare.com/ajax/libs/"
            "three.js/r73/three.js",
        "https://cdnjs.cloudflare.com/ajax/libs/"
            "react/0.14.3/react.js",
        "https://cdnjs.cloudflare.com/ajax/libs/"
            "react/0.14.3/react-dom.js",
        "https://cdnjs.cloudflare.com/ajax/libs/"
            "jquery/2.1.1/jquery.js",
    ]

    @property
    def internal_scripts(self):
        files = list()
        for root, directory, fnames in os.walk(self.JSDIR):
            common = os.path.commonprefix([root, self.JSDIR])
            lead = root[len(common):].strip(os.path.sep)
            files.extend(os.path.join(lead, fname) for fname in fnames)
        return files

    def __init__(self, **kw):
        self.blk = kw.pop('blk', None)
        super(HtmlIO, self).__init__()

    @staticmethod
    def convert_js(input_data):
        return "".join([
            "var SOLVCON_input_data = ",
            json.dumps(input_data, cls=block.BlockJSONEncoder),
            "\n",
        ])

    def get_input_data(self):
        blk = self.blk
        surfaces = list()
        for bc in blk.bclist:
            fcnds = blk.fcnds[bc.facn[:,0]]
            nds = fcnds[:,1:].flatten()
            nds.sort()
            nds = np.unique(nds)
            it = 0
            while nds[it] < 0:
                it += 1
            nds = nds[it:].copy()
            ndcrd = blk.ndcrd[nds]
            ndmap = dict((val, it) for it, val in enumerate(nds))
            it = 0
            while it < fcnds.shape[0]:
                jt = 1
                while fcnds[it,jt] >= 0:
                    fcnds[it,jt] = ndmap[fcnds[it,jt]]
                    jt += 1
                it += 1
            surfaces.append(dict(
                name=bc.name,
                ndcrd=ndcrd.tolist(),
                fcnds=fcnds.tolist(),
            ))
        return dict(
            block=self.blk,
            boundary_surfaces=surfaces,
        )

    def _save_directory(self, dirname):
        for fname in self.internal_scripts:
            shutil.copy(os.path.join(self.JSDIR, fname),
                        os.path.join(dirname, fname))
        mesh_blk_str = self.convert_js(self.get_input_data())
        with open(os.path.join(dirname, 'input_data.js'), 'w') as fobj:
            fobj.write(mesh_blk_str)

        with open(os.path.join(self.HTMLDIR, 'index.html')) as fobj:
            template = jinja2.Template(fobj.read())
        with open(os.path.join(dirname, 'index.html'), 'w') as fobj:
            scripts = self.EXTERNAL_SCRIPTS + self.internal_scripts
            scripts += ['input_data.js']
            htmldata = template.render(
                title=os.path.split(dirname.strip('/'))[-1],
                scripts=scripts,
            )
            fobj.write(htmldata)

    def _save_file(self, filename):
        scripttexts = list()
        for fname in self.internal_scripts:
            with open(os.path.join(self.JSDIR, fname)) as fobj:
                scripttexts.append(fobj.read())
        scripttexts.append(self.convert_js(self.get_input_data()))

        with open(os.path.join(self.HTMLDIR, 'index.html')) as fobj:
            template = jinja2.Template(fobj.read())
        with open(os.path.join(filename), 'w') as fobj:
            htmldata = template.render(
                title=os.path.split(filename)[-1],
                scripts=self.EXTERNAL_SCRIPTS,
                scripttexts=scripttexts,
            )
            fobj.write(htmldata)

    def save(self, stream):
        if not isinstance(stream, str):
            raise AssertionError("stream must be a string")
        if os.path.exists(stream):
            otype = "directory" if os.path.isdir(stream) else "file"
            warnings.warn("output %s (%s) already exists" % (otype, stream),
                          exception.IOWarning)
        else:
            otype = "directory" if stream.endswith(os.path.sep) else "file"
            if "directory" == otype:
                os.makedirs(stream)
        if "file" == otype:
            self._save_file(stream)
        else:
            self._save_directory(stream)
