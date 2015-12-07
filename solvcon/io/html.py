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
# - Neither the name of the software nor the names of its contributors may be
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
Mesh I/O for HTML files based on THREE.js.
"""


import os

from . import core as iocore


class HtmlIO(iocore.FormatIO):
    """
    Experimental HTML/THREE.js writer.
    """

    HTML_HEADER = b"""<html>
    <head>
        <title>Mesh</title>
        <style>
            body { margin: 0; }
            canvas { width: 100%; height: 100% }
        </style>
    </head>
    <body>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r73/three.min.js"></script>
    """

    HTML_FOOTER = b"""    </body>
</html>

<!-- vim: set ff=unix fenc=utf8 nobomb et sw=4 ts=4: -->
    """

    def __init__(self, **kw):
        self.blk = kw.pop('blk', None)
        super(HtmlIO, self).__init__()

    def _get_vertex_bytes(self):
        if self.blk.ndim == 2:
            tmpl = "new THREE.Vector3(%g,%g,0)"
        else:
            tmpl = "new THREE.Vector3(%g,%g,%g)"
        return ",\n".join(tmpl % tuple(pnt) for pnt in self.blk.ndcrd)

    def _get_face_bytes(self):
        tmpl = "new THREE.Face3(%d,%d,%d)"
        if self.blk.ndim == 2:
            return ",\n".join(
                tmpl % tuple(nds[1:nds[0]+1]) for nds in self.blk.clnds)
        else:
            return ",\n".join(
                tmpl % tuple(nds[1:nds[0]+1]) for nds in self.blk.fcnds)

    def save(self, stream, ball_radius=0.0):
        if stream == None:
            stream = open(self.filename, 'wb')
        elif isinstance(stream, str):
            stream = open(stream, 'wb')

        basedir = os.path.abspath(os.path.dirname(__file__))
        basedir = os.path.join(basedir, 'three')

        stream.write(self.HTML_HEADER)

        stream.write(b"<!-- Overall section -->\n")
        stream.write(b"<script>\n")
        rfn = os.path.join(basedir, 'TrackballControls.js')
        with open(rfn, 'rb') as fobj:
            stream.write(fobj.read())
        stream.write(b"\n</script>\n")

        stream.write(b"<!-- Mesh geometry section -->\n")
        stream.write(b"<script>\n")
        stream.write(b"""function get_geometry() {
    var geom = new THREE.Geometry();
    geom.vertices.push(
""")
        stream.write(self._get_vertex_bytes().encode())
        stream.write(b"""
    );
    geom.faces.push(
""")
        stream.write(self._get_face_bytes().encode())
        stream.write(("""
    );
    return geom;
}
var ball_radius = %g;
""" % ball_radius).encode())
        stream.write(b"\n</script>\n")

        stream.write(b"<!-- Main program section -->\n")
        stream.write(b"<script>\n")
        rfn = os.path.join(basedir, 'main.js')
        with open(rfn, 'rb') as fobj:
            stream.write(fobj.read())
        stream.write(b"\n</script>\n")

        stream.write(self.HTML_FOOTER)

        stream.close()
