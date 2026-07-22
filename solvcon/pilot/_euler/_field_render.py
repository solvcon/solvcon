# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Turn a per-cell scalar field into the flat colored triangles that
``RDomainWidget.updateColorField`` draws.

The mesh cells are triangulated once with :func:`cell_triangulation`; each
frame then maps the scalar field to vertex colors with :func:`field_colors`
over that fixed triangulation.  Shared by the oblique-shock sample and the
solution panel so both read a 2D field the same way.
"""

import numpy as np

from ... import core


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

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
