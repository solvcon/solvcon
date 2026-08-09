# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Turn a per-cell scalar field into the flat colored triangles that
``RDomainWidget.updateColorField`` draws.

A run draws through one :class:`FieldPainter`: building it triangulates the
mesh cells once with :func:`cell_triangulation`, and each frame then maps
the scalar field to vertex colors with :func:`field_colors` over that fixed
triangulation, so a running solver redraws without rebuilding the geometry.
Everything here is numpy in and numpy out; no Qt enters the module.
"""

import numpy as np

from .... import core


def colormap(t):
    """Map ``t`` in [0, 1] to a jet-like RGB array (``..., 3``) in [0, 1].

    A four-stop blue-cyan-yellow-red ramp; the triangle-wave channels are the
    standard compact "jet" approximation, enough to read a scalar field
    without pulling in a plotting dependency.
    """
    t = np.clip(np.asarray(t, dtype='float64'), 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
    return np.stack([r, g, b], axis=-1)


def cell_triangulation(mh):
    """Fan every cell into unshared triangles for a flat color field.

    Each cell fans into ``nnd - 2`` triangles whose corners are emitted
    unshared, so a cell can take one flat color.  Returns the triangles as a
    ``TrianglePadFp32`` and the per-cell vertex count that :func:`field_colors`
    repeats each cell's color over.
    """
    nodes = core.PointPadFp32(ndim=3)
    for ind in range(mh.nnode):
        nodes.append(mh.ndcrd[ind, 0], mh.ndcrd[ind, 1], 0.0)
    fan = core.TrianglePadFp32(ndim=3)
    counts = core.SimpleCollectorInt32(0)
    for icl in range(mh.ncell):
        nnd = int(mh.clnds[icl, 0])
        apex = nodes.get_at(int(mh.clnds[icl, 1]))
        for it in range(1, nnd - 1):
            fan.append(apex,
                       nodes.get_at(int(mh.clnds[icl, 1 + it])),
                       nodes.get_at(int(mh.clnds[icl, 1 + it + 1])))
        counts.push_back(3 * (nnd - 2))
    return fan, counts.as_array().ndarray


def field_colors(field, counts, vmin, vmax):
    """Map each cell's scalar to a color and repeat it over the cell's
    vertices, matching the layout of :func:`cell_triangulation`.
    """
    span = vmax - vmin
    t = (field - vmin) / span if span > 0 else np.zeros_like(field)
    return np.repeat(colormap(t), counts, axis=0).astype('float32')


class FieldPainter(object):
    """Hold the fixed triangulation of one run and color its fields.

    ``updateColorField`` wants an indexed vertex soup; the cell fan already
    is one, so the vertices are packed once when the painter is built, the
    geometry being fixed for the run, and indexed sequentially.  A frame
    then costs only the color mapping of :meth:`colors`.

    :ivar verts: The packed triangle vertices of the whole mesh.
    :ivar indices: The sequential vertex indices, one row per triangle.
    """

    def __init__(self, mh):
        fan, self._counts = cell_triangulation(mh)
        verts = fan.pack_array().ndarray.reshape(-1, 3)
        indices = np.arange(verts.shape[0], dtype='uint32').reshape(-1, 3)
        self.verts = core.SimpleArrayFloat32(array=verts)
        self.indices = core.SimpleArrayUint32(array=indices)

    def colors(self, field, vmin, vmax):
        """The vertex colors of one frame of ``field`` over the range."""
        colors = field_colors(field, self._counts, vmin, vmax)
        return core.SimpleArrayFloat32(array=colors)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
