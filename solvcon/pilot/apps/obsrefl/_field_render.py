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


def thick_segments(starts, ends, half):
    """Turn line segments into quads, as an unshared triangle soup.

    The graphics backend rasterizes every line one pixel wide, ignoring any
    width asked of it, so a line thick enough to read as a mark has to be
    geometry.  Each segment becomes a rectangle ``2 * half`` across, run
    past both ends by ``half`` so the rectangles of a closed ring meet at
    its corners instead of leaving notches.

    ``starts`` and ``ends`` are ``[n, 2]``; the result is ``[m, 2]`` with
    six vertices per surviving segment.
    """
    starts = np.asarray(starts, dtype='float64').reshape(-1, 2)
    ends = np.asarray(ends, dtype='float64').reshape(-1, 2)
    delta = ends - starts
    length = np.hypot(delta[:, 0], delta[:, 1])
    # A segment of no length has no direction to be thick across.
    keep = length > 0.0
    starts, ends = starts[keep], ends[keep]
    unit = delta[keep] / length[keep][:, None]
    along = unit * half
    across = np.column_stack([-unit[:, 1], unit[:, 0]]) * half
    back, front = starts - along, ends + along
    return np.stack([back + across, back - across, front - across,
                     back + across, front - across, front + across],
                    axis=1).reshape(-1, 2)


def cell_rings(mh, mark):
    """The edges of the marked cells, as ``(starts, ends)`` of ``[n, 2]``.

    Each cell contributes its whole rim, so a thickened ring drawn over it
    outlines the cell without hiding the field color inside.
    """
    clnds = mh.clnds.ndarray[mh.clnds.nghost:]
    ndcrd = mh.ndcrd.ndarray[mh.ndcrd.nghost:][:, :2]
    ring = clnds[np.asarray(mark)]
    counts = ring[:, 0]
    wide = int(counts.max()) if len(counts) else 0
    column = np.arange(wide, dtype='int64')[None, :]
    held = column < counts[:, None]
    ring = ring[:, 1:1 + wide]
    # Pair every node with the next one around the rim, which comes back to
    # the first at the cell's own count and not at the widest cell's.
    following = np.take_along_axis(ring, (column + 1) % counts[:, None],
                                   axis=1)
    return ndcrd[ring[held]], ndcrd[following[held]]


class FieldPainter(object):
    """Hold the fixed triangulation of one run and color its fields.

    ``updateColorField`` wants an indexed vertex soup; the cell fan already
    is one, so the vertices are packed once when the painter is built, the
    geometry being fixed for the run, and indexed sequentially.  A frame
    then costs only the color mapping of :meth:`colors`.

    The gauge of a profile cut rides in the same soup, as thickened rings
    around the cells the cut crosses and a bar along the cut, either of
    which may be left out.  A scene object of its own would refit the
    camera; installing a field refits it only when the field's bounding
    box changes, and this one never does.

    :ivar verts: The packed triangle vertices, the gauge included.
    :ivar indices: The sequential vertex indices, one row per triangle.
    """

    #: Lift each layer of the gauge off the z = 0 field plane and off each
    #: other.  The depth test passes only what is strictly nearer, so
    #: coplanar layers fall to draw order and the bar is lost under the
    #: rings it crosses.
    OUTLINE_LIFT = 0.01
    BAR_LIFT = 0.02
    #: Half-widths of the cell outline and of the cut bar, as fractions of
    #: the average cell.  Sized against the cell and not the domain, or a
    #: width that reads on a coarse mesh swallows a fine one.  The outline
    #: beats the one-pixel wireframe it thickens, and the bar the outline.
    OUTLINE_HALF = 0.05
    BAR_HALF = 0.09
    #: Room the anchors reserve around the domain for the gauge to spill
    #: into, as a fraction of the domain height.  A fraction of the domain
    #: and not of the mesh, so a remesh does not move the box and refit the
    #: camera with it.
    ANCHOR_PAD = 0.05
    #: Violet is clear of everything else on screen: the blue-to-red
    #: field ramp, the white background, the black wireframe, and the
    #: magenta shock ruler.  The bar takes the lighter shade, standing
    #: over the rings it crosses.
    OUTLINE_COLOR = (0.42, 0.05, 0.75)
    BAR_COLOR = (0.72, 0.42, 1.0)

    def __init__(self, mh):
        fan, self._counts = cell_triangulation(mh)
        self._mesh = mh
        self._field = fan.pack_array().ndarray.reshape(-1, 3)
        node = mh.ndcrd.ndarray[mh.ndcrd.nghost:]
        self._lo = node.min(axis=0)[:2]
        self._hi = node.max(axis=0)[:2]
        volume = mh.clvol.ndarray[mh.clvol.nghost:]
        # The side of the average cell, as the length the gauge is sized
        # against.
        self._cell = float(np.sqrt(volume.mean()))
        self.set_gauge(None, None)

    def set_gauge(self, mark, height):
        """Outline the cells of ``mark`` and bar the cut at ``height``.

        The two layers stand on their own: passing None for either leaves
        that one out, and None for both leaves the field bare.
        """
        parts = [(self._anchors(), (0.0, 0.0, 0.0), self.BAR_LIFT)]
        if mark is not None:
            parts.append((thick_segments(*cell_rings(self._mesh, mark),
                                         self._span(self.OUTLINE_HALF)),
                          self.OUTLINE_COLOR, self.OUTLINE_LIFT))
        if height is not None:
            parts.append((thick_segments([(self._lo[0], height)],
                                         [(self._hi[0], height)],
                                         self._span(self.BAR_HALF)),
                          self.BAR_COLOR, self.BAR_LIFT))
        self._gauge = np.concatenate(
            [np.column_stack([flat, np.full(len(flat), lift, dtype='float64')])
             for flat, _, lift in parts]).astype('float32')
        self._gauge_colors = np.concatenate(
            [np.tile(np.array(color, dtype='float32'), (len(flat), 1))
             for flat, color, _ in parts])
        self._pack()

    def colors(self, field, vmin, vmax):
        """The vertex colors of one frame of ``field`` over the range."""
        return core.SimpleArrayFloat32(array=np.concatenate(
            [field_colors(field, self._counts, vmin, vmax),
             self._gauge_colors]))

    def _span(self, fraction):
        """A world length as a fraction of the average cell, held inside
        the room the anchors reserve.

        The clamp applies only where a cell is comparable to the domain,
        and the gauge would otherwise exceed the anchored box.
        """
        return min(fraction * self._cell, self._pad())

    def _pad(self):
        return self.ANCHOR_PAD * (self._hi[1] - self._lo[1])

    def _anchors(self):
        """Two zero-area triangles pinning the gauge's bounding box.

        The camera refits whenever the field's box changes, so a gauge
        that came and went would jerk the view.  These sit at the corners
        of the room the gauge may use and hold the box still.
        """
        pad = self._pad()
        return np.concatenate([np.tile(self._lo - pad, (3, 1)),
                               np.tile(self._hi + pad, (3, 1))])

    def _pack(self):
        verts = np.concatenate([self._field, self._gauge])
        self.verts = core.SimpleArrayFloat32(array=verts)
        self.indices = core.SimpleArrayUint32(array=np.arange(
            len(verts), dtype='uint32').reshape(-1, 3))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
