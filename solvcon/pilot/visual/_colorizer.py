# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""Color a field for drawing on its mesh."""

import numpy as np

from PySide6.QtCore import Qt, QRect
from PySide6.QtGui import QPainter, QLinearGradient, QColor, QPen, QPalette
from PySide6.QtWidgets import QWidget, QSizePolicy

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


def thick_segments(starts, ends, half, extend=True):
    """Turn line segments into quads, as an unshared triangle soup.

    The graphics backend rasterizes every line one pixel wide, ignoring any
    width asked of it, so a line thick enough to read as a mark has to be
    geometry.  Each segment becomes a rectangle ``2 * half`` across.  By
    default it runs past both ends by ``half``, so the rectangles of a
    closed ring meet at its corners instead of leaving notches; the square
    ends then poke out of a sharp corner.  With ``extend`` off the
    rectangles stop at the ends, for a ring that joins its corners with
    :func:`discs` instead.

    :param starts: The start points of the segments, ``[n, 2]``.
    :param ends: The end points of the segments, ``[n, 2]``.
    :param half: The half width of a quad, in world length.
    :param extend: Whether a quad runs past both ends by ``half``.
    :return: The triangle vertices, ``[m, 2]``, six per segment of nonzero
        length.
    """
    starts = np.asarray(starts, dtype='float64').reshape(-1, 2)
    ends = np.asarray(ends, dtype='float64').reshape(-1, 2)
    delta = ends - starts
    length = np.hypot(delta[:, 0], delta[:, 1])
    # A segment of no length has no direction to be thick across.
    keep = length > 0.0
    starts, ends = starts[keep], ends[keep]
    unit = delta[keep] / length[keep][:, None]
    along = unit * half if extend else np.zeros_like(unit)
    across = np.column_stack([-unit[:, 1], unit[:, 0]]) * half
    back, front = starts - along, ends + along
    return np.stack([back + across, back - across, front - across,
                     back + across, front - across, front + across],
                    axis=1).reshape(-1, 2)


def discs(centers, radius, sides=12):
    """Turn points into filled polygons of ``sides`` corners, as an
    unshared triangle soup.

    The graphics backend draws a point one pixel across, so a dot that
    reads as a mark has to be geometry, as a thick line does.

    :param centers: The points, ``[n, 2]``.
    :param radius: The radius of a disc, in world length.
    :param sides: The corners of each polygon.
    :return: The triangle vertices, ``[n * 3 * sides, 2]``.
    """
    centers = np.asarray(centers, dtype='float64').reshape(-1, 2)
    angle = np.linspace(0.0, 2.0 * np.pi, sides + 1)
    rim = radius * np.column_stack([np.cos(angle), np.sin(angle)])
    fan = np.stack([np.zeros((sides, 2)), rim[:-1], rim[1:]], axis=1)
    return (centers[:, None, None, :] + fan[None]).reshape(-1, 2)


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


class FieldColorizer(object):
    """Hold the fixed triangulation of one run and color its fields.

    ``updateColorField`` wants an indexed vertex soup; the cell fan already
    is one, so the vertices are packed once when the colorizer is built, the
    geometry being fixed for the run, and indexed sequentially.  A frame
    then costs only the color mapping of :meth:`colors`.

    The gauge of a profile cut rides in the same soup, as thickened rings
    around the cells the cut crosses and a bar along the cut, either of
    which may be left out, and so does the highlight of one cell: a cased
    ring on its edges, which a pin thickens and dots on each node.  A scene
    object of its own would refit the camera; installing a field refits it
    only when the field's bounding box changes, and this one never does.

    :ivar verts: The packed triangle vertices, the gauge included.
    :ivar indices: The sequential vertex indices, one row per triangle.
    """

    #: Lift each layer of the gauge off the z = 0 field plane and off each
    #: other.  The depth test passes only what is strictly nearer, so
    #: coplanar layers fall to draw order and the bar is lost under the
    #: rings it crosses.
    OUTLINE_LIFT = 0.01
    BAR_LIFT = 0.02
    HALO_LIFT = 0.025
    CORE_LIFT = 0.03
    #: Half-widths of the cell outline and of the cut bar, as fractions of
    #: the average cell.  Sized against the cell and not the domain, or a
    #: width that reads on a coarse mesh swallows a fine one.  The outline
    #: beats the one-pixel wireframe it thickens, and the bar the outline.
    OUTLINE_HALF = 0.05
    BAR_HALF = 0.09
    #: The highlight leaves the cell its field color and marks it with a
    #: cased line, a dark core inside a light halo, which reads on every
    #: part of the ramp.  Half-widths of the core on a hovered and on a
    #: pinned cell, the radius of a pinned cell's node dots, and the width
    #: of the halo around them, as fractions of the average cell.
    HOVER_HALF = 0.05
    PIN_HALF = 0.07
    PIN_DOT = 0.18
    HALO = 0.04
    #: Corners of each round join and node dot.  A polygon of n sides
    #: falls short of its circle by 1 - cos(pi / n) of the radius, 0.2% at
    #: 48, under a pixel on a dot zoomed to fill a quarter of the view, for
    #: a few hundred vertices against the thousands of the field.
    ROUND_SIDES = 48
    #: Room the anchors reserve around the domain for the gauge to spill
    #: into, as a fraction of the domain height.  A fraction of the domain
    #: and not of the mesh, so a remesh does not move the box and refit the
    #: camera with it.
    ANCHOR_PAD = 0.05
    #: Violet is clear of everything else on screen: the blue-to-red
    #: field ramp, the white background, the black wireframe, and the
    #: magenta an app keeps for its own marks.  The bar takes the lighter
    #: shade, standing over the rings it crosses.
    OUTLINE_COLOR = (0.42, 0.05, 0.75)
    BAR_COLOR = (0.72, 0.42, 1.0)
    #: A shade off black keeps the core apart from the anchors.
    CORE_COLOR = (0.05, 0.05, 0.05)
    HALO_COLOR = (1.0, 1.0, 1.0)

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
        self._mark = self._height = self._highlighted = None
        self._pinned = False
        self._field_colors = None
        self._build()

    def set_gauge(self, mark, height):
        """Outline the cells of ``mark`` and bar the cut at ``height``.

        The two layers stand on their own: passing None for either leaves
        that one out, and None for both leaves the field bare.

        :param mark: The indices of the body cells to outline, or None.
        :param height: The y coordinate of the cut, or None.
        """
        self._mark, self._height = mark, height
        self._build()

    def set_highlight(self, cell, pinned=False):
        """Ring ``cell``, thicker and with its nodes dotted when pinned, or
        None to drop the highlight.

        :param cell: The index of a body cell, or None.
        :param pinned: Whether the cell is pinned.
        :return: Whether the highlight changed, and the colors with it.
        """
        pinned = cell is not None and bool(pinned)
        if cell == self._highlighted and pinned == self._pinned:
            return False
        self._highlighted, self._pinned = cell, pinned
        self._build()
        return True

    def recolor(self):
        """The vertex colors of the last frame over the gauge as it stands
        now, or None before any frame, so a gauge change shows without a
        new frame."""
        if self._field_colors is None:
            return None
        return core.SimpleArrayFloat32(array=np.concatenate(
            [self._field_colors, self._gauge_colors]))

    def _build(self):
        # The anchors take the top lift, so no layer above them grows the
        # box.
        parts = [(self._anchors(), (0.0, 0.0, 0.0), self.CORE_LIFT)]
        if self._mark is not None:
            parts.append((thick_segments(*cell_rings(self._mesh, self._mark),
                                         self._span(self.OUTLINE_HALF)),
                          self.OUTLINE_COLOR, self.OUTLINE_LIFT))
        if self._height is not None:
            parts.append((thick_segments([(self._lo[0], self._height)],
                                         [(self._hi[0], self._height)],
                                         self._span(self.BAR_HALF)),
                          self.BAR_COLOR, self.BAR_LIFT))
        if self._highlighted is not None:
            parts.extend(self._casing())
        self._gauge = np.concatenate(
            [np.column_stack([flat, np.full(len(flat), lift, dtype='float64')])
             for flat, _, lift in parts]).astype('float32')
        self._gauge_colors = np.concatenate(
            [np.tile(np.array(color, dtype='float32'), (len(flat), 1))
             for flat, color, _ in parts])
        self._pack()

    def _casing(self):
        """The halo and the core of the highlight, as gauge parts.

        The ring joins its corners round, with a disc on each node as wide
        as the ring, or as wide as the dot on a pinned cell: square ends
        run past a corner would stick out of the halo as bulges.
        """
        starts, ends = cell_rings(self._mesh, [self._highlighted])
        half = self.PIN_HALF if self._pinned else self.HOVER_HALF
        dot = self.PIN_DOT if self._pinned else half
        parts = []
        for grow, color, lift in ((self.HALO, self.HALO_COLOR,
                                   self.HALO_LIFT),
                                  (0.0, self.CORE_COLOR, self.CORE_LIFT)):
            flat = np.concatenate([
                thick_segments(starts, ends, self._span(half + grow),
                               extend=False),
                discs(starts, self._span(dot + grow),
                      sides=self.ROUND_SIDES)])
            parts.append((flat, color, lift))
        return parts

    def colors(self, field, vmin, vmax):
        """The vertex colors of one frame of ``field`` over the range.

        :param field: The scalar of each body cell, ``[ncell]``.
        :param vmin: The value at the low end of the color ramp.
        :param vmax: The value at the high end of the color ramp.
        :return: The colors of the field and of the gauge after it, as
            ``core.SimpleArrayFloat32``.
        """
        self._field_colors = field_colors(field, self._counts, vmin, vmax)
        return self.recolor()

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


class ColorBar(QWidget):
    """A slim colormap ramp with its two ends labelled.

    :ivar lo: Low end of the pinned range, or None with nothing to show.
    :ivar hi: High end of the pinned range.
    :ivar vertical: Whether the ramp runs up the widget instead of across.
    :ivar themed: Whether to draw the frame and the labels in the text color
        of the palette, with no ground of its own, instead of dark on white.
    :ivar label: The name written along a vertical ramp, or None to keep no
        room for a name.
    """

    #: Colors sampled across the ramp to build the gradient.  The map is
    #: piecewise linear in four stops, so this is far more than it takes to
    #: draw it smoothly and still cheap to rebuild every frame.
    STOPS = 32
    #: Thickness of the ramp itself, in pixels, across its run.
    BAR_THICKNESS = 12
    #: Room kept at each end of the run for half an end label, so the text
    #: of a label centered on the extreme does not run off the widget.
    END_PAD = 24
    #: Width the end-label column takes beside a vertical ramp.
    LABEL_WIDTH = 52
    #: Room kept around the ramp, across its run.
    PAD = 4
    #: The ground under the bar, which is what the domain viewer clears its
    #: frame to, so the strip does not read as a panel over the view.
    BACKGROUND = (1.0, 1.0, 1.0)
    #: Ink for the frame and the labels.  Fixed rather than
    #: taken from the palette, because the ground it is written on is: a
    #: dark theme's near-white text would vanish on it.
    INK = (0.1, 0.1, 0.1)

    def __init__(self, vertical=False, parent=None, themed=False,
                 label=None):
        super().__init__(parent)
        self.lo = None
        self.hi = None
        self.vertical = vertical
        self.themed = themed
        self.label = label
        if vertical:
            self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
            self.setMinimumWidth(self.thickness())
        else:
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            self.setMinimumHeight(self.thickness())

    def thickness(self):
        """Room the bar needs across its run, labels included.

        One label line, not two: the ends are labelled on one side of the
        ramp and nothing is written on the other.
        """
        room = self.BAR_THICKNESS + 2 * self.PAD
        if self.vertical:
            if self.label is not None:
                room += self.fontMetrics().height()
            return room + self.LABEL_WIDTH
        return room + self.fontMetrics().height()

    def show_scale(self, lo, hi, label=None):
        """Draw the ramp over ``lo`` to ``hi``, and name it ``label`` if
        given.

        A missing or degenerate range blanks the widget rather than
        dividing by a zero span.
        """
        if None is lo or None is hi or not hi > lo:
            self.lo = self.hi = None
        else:
            self.lo, self.hi = float(lo), float(hi)
        if label is not None:
            self.label = label
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        if None is self.lo:
            return
        if self.themed:
            ink = self.palette().color(QPalette.Text)
        else:
            painter.fillRect(self.rect(), QColor.fromRgbF(*self.BACKGROUND))
            ink = QColor.fromRgbF(*self.INK)
        bar = self._bar_rect()
        painter.fillRect(bar, self._ramp(bar))
        painter.setPen(QPen(ink))
        painter.drawRect(bar)
        self._draw_ends(painter, bar)
        if self.vertical and self.label:
            self._draw_label(painter, bar)

    def _bar_rect(self):
        """Where the ramp is drawn, with room beside it for the labels."""
        if self.vertical:
            return QRect(self.LABEL_WIDTH + self.PAD, self.END_PAD,
                         self.BAR_THICKNESS,
                         max(1, self.height() - 2 * self.END_PAD))
        return QRect(self.END_PAD, self.PAD,
                     max(1, self.width() - 2 * self.END_PAD),
                     self.BAR_THICKNESS)

    def _ramp(self, bar):
        """The colormap as a gradient along ``bar``.

        A vertical ramp runs low at the bottom, the way an axis does, so
        the gradient is laid out bottom to top.
        """
        if self.vertical:
            gradient = QLinearGradient(0, bar.bottom(), 0, bar.top())
        else:
            gradient = QLinearGradient(bar.left(), 0, bar.right(), 0)
        for it in range(self.STOPS + 1):
            t = it / self.STOPS
            red, green, blue = colormap(t)
            gradient.setColorAt(t, QColor.fromRgbF(red, green, blue))
        return gradient

    def _at(self, value, bar):
        """Where ``value`` falls along the ramp, in widget pixels."""
        t = (value - self.lo) / (self.hi - self.lo)
        if self.vertical:
            return bar.bottom() - t * (bar.height() - 1)
        return bar.left() + t * (bar.width() - 1)

    def _label_rect(self, at, bar, height):
        """Room for a label beside the ramp at ``at`` along its run."""
        if self.vertical:
            return QRect(0, round(at) - height // 2, self.LABEL_WIDTH, height)
        return QRect(round(at) - self.END_PAD, bar.bottom() + 1,
                     2 * self.END_PAD, height)

    def _draw_ends(self, painter, bar):
        """Label the two ends of the range against the ramp."""
        height = self.fontMetrics().height()
        align = (Qt.AlignRight if self.vertical else Qt.AlignHCenter)
        for value in (self.lo, self.hi):
            painter.drawText(self._label_rect(self._at(value, bar), bar,
                                              height),
                             align | Qt.AlignVCenter, _number(value))

    def _draw_label(self, painter, bar):
        """Write the label up the far side of the ramp, as an axis title
        runs."""
        metrics = self.fontMetrics()
        height, span = metrics.height(), bar.height()
        painter.translate(bar.right() + self.PAD + height, bar.center().y())
        painter.rotate(-90)
        painter.drawText(QRect(-span // 2, -height, span, height),
                         Qt.AlignCenter,
                         metrics.elidedText(self.label, Qt.ElideRight, span))


def _number(value):
    """Format an end label to three significant digits."""
    return f"{value:#.3g}"

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
