# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Mesh generation for a rectangular domain.
"""

import math

from .. import core

__all__ = [
    'RectDomainMesher',
]


class RectDomainMesher(object):
    """Generate a mesh of a rectangular domain.

    The rectangular domain runs from the lower-left corner ``ll`` to the
    upper-right corner ``ur`` and is divided into ``nx`` by ``ny`` grid
    boxes.
    """

    def __init__(self, nx=64, ny=16, ll=(0.0, 0.0), ur=(4.0, 1.0)):
        self.nx = nx
        self.ny = ny
        (self.x0, self.y0), (self.x1, self.y1) = ll, ur

    @property
    def cell_extent(self):
        """The ``(width, height)`` of one grid box.

        Each element flavor fills a box with one or two cells, so the box is
        the resolution of the mesh whatever the flavor, and it is what a
        measurement over the mesh has to be cut to.
        """
        return ((self.x1 - self.x0) / self.nx,
                (self.y1 - self.y0) / self.ny)

    def _node(self, it, jt):
        return (self.x0 + it * (self.x1 - self.x0) / self.nx,
                self.y0 + jt * (self.y1 - self.y0) / self.ny)

    def _nid(self, it, jt):
        return jt * (self.nx + 1) + it

    def _box(self, it, jt):
        return (self._nid(it, jt), self._nid(it + 1, jt),
                self._nid(it + 1, jt + 1), self._nid(it, jt + 1))

    def make_mesh(self, cell_type='unstructured'):
        """Build a :class:`~solvcon.core.StaticMesh` of the selected flavor.

        The unstructured flavor is the default because it is the one the
        CESE scheme is meant for; the two structured flavors line their cells
        up with the domain, which flatters a solver on a rectangle.

        ``cell_type`` selects the element shape:

        - ``'quad'`` keeps one quadrilateral per grid box,
        - ``'triangle'`` cuts each box along its lower-left-to-upper-right
          diagonal into two triangles (flipping the diagonal at two corners
          so no triangle carries two boundary faces), and
        - ``'unstructured'`` Delaunay-triangulates the same boundary nodes
          plus jittered interior points into an irregular (but deterministic)
          triangulation, refined to the same one-boundary-face-per-cell rule.

        Both triangular flavors keep at most one boundary face per cell,
        which the corner ``'quad'`` cells cannot.  All flavors share the
        boundary layout (nx segments on the bottom and top, ny on the left
        and right) and produce counter-clockwise cells.  The returned mesh
        has ``ndcrd``/``cltpn``/``clnds`` filled, carries one named
        boundary-condition group per domain edge (:attr:`BOUNDARY_NAMES`),
        and has ``build_interior`` / ``build_boundary`` / ``build_ghost``
        run.
        """
        nx, ny = self.nx, self.ny
        if cell_type in ('quad', 'triangle'):
            nodes = core.PointPadFp64(ndim=2)
            for jt in range(ny + 1):
                for it in range(nx + 1):
                    nodes.append(*self._node(it, jt))
            if cell_type == 'quad':
                tpn = core.StaticMesh.QUADRILATERAL
                cells = [(4,) + self._box(it, jt)
                         for jt in range(ny) for it in range(nx)]
            else:
                tpn = core.StaticMesh.TRIANGLE
                cells = []
                for jt in range(ny):
                    for it in range(nx):
                        ll, lr, ur, ul = self._box(it, jt)
                        # Split along the lower-left-to-upper-right diagonal,
                        # except at the upper-left and lower-right domain
                        # corners, whose corner cell would otherwise carry
                        # the two boundary edges meeting there.
                        if (it, jt) in ((0, ny - 1), (nx - 1, 0)):
                            cells += [(3, ll, lr, ul), (3, lr, ur, ul)]
                        else:
                            cells += [(3, ll, lr, ur), (3, ll, ur, ul)]
        elif cell_type == 'unstructured':
            tpn = core.StaticMesh.TRIANGLE
            nodes = self._jitter_points()
            cells = [(3, *tri)
                     for tri in self._split_double_boundary(nodes)]
        else:
            raise ValueError(f"unknown cell_type '{cell_type}'")
        mh = core.StaticMesh(ndim=2, nnode=len(nodes), nface=0,
                             ncell=len(cells))
        mh.ndcrd[:, :] = nodes.pack_array().ndarray
        mh.cltpn.fill(tpn)
        mh.clnds[:, :len(cells[0])] = cells
        mh.build_interior(do_metric=True)
        for name, faces in zip(self.BOUNDARY_NAMES,
                               self.classify_boundary(mh)):
            mh.add_bc(name, faces)
        mh.build_boundary()
        mh.build_ghost()
        return mh

    #: Boundary-condition group name per domain edge, in the order
    #: :meth:`classify_boundary` returns the edges.
    BOUNDARY_NAMES = ('left', 'top', 'bottom', 'right')

    def _jitter_points(self):
        """Collect the unstructured point cloud in a :class:`PointPad`.

        The boundary keeps the structured node layout (a counter-clockwise
        outline walk), so classification is identical across flavors.
        Interior nodes are displaced in logical (it, jt) space by an
        RNG-free phase -- deterministic, yet irregular -- bounded by 0.3
        logical units per axis to stay well inside the domain.
        """
        nx, ny = self.nx, self.ny
        pad = core.PointPadFp64(ndim=2)
        for it in range(nx + 1):
            pad.append(*self._node(it, 0))
        for jt in range(1, ny + 1):
            pad.append(*self._node(nx, jt))
        for it in range(nx - 1, -1, -1):
            pad.append(*self._node(it, ny))
        for jt in range(ny - 1, 0, -1):
            pad.append(*self._node(0, jt))
        for jt in range(1, ny):
            for it in range(1, nx):
                pad.append(*self._node(
                    it + 0.3 * math.sin(12.9898 * it + 78.233 * jt),
                    jt + 0.3 * math.cos(26.651 * it + 41.347 * jt)))
        return pad

    @classmethod
    def _triangulate(cls, pad):
        """Bowyer-Watson Delaunay triangulation of the :class:`PointPad`
        ``pad``; returns the CCW node indices as an n-by-3 integer
        :class:`~solvcon.core.SimpleArray`.

        Seed a far-away super-triangle, insert one point at a time (carve
        the triangles whose circumcircle strictly contains it, fan the
        cavity rim to the point), then drop the triangles touching a super
        vertex.  On-circle points count as outside, keeping the cavity well
        defined under the cocircular ties of the regular boundary layout.
        Naive O(n^2).

        References:
        - A. Bowyer, "Computing Dirichlet tessellations", The Computer
          Journal 24(2):162-166, 1981.
          https://doi.org/10.1093/comjnl/24.2.162
        - D. F. Watson, "Computing the n-dimensional Delaunay tessellation
          with application to Voronoi polytopes", The Computer Journal
          24(2):167-172, 1981.  https://doi.org/10.1093/comjnl/24.2.167
        - https://en.wikipedia.org/wiki/Bowyer%E2%80%93Watson_algorithm
        """
        npt = len(pad)
        xmin, xmax = pad.x.min(), pad.x.max()
        ymin, ymax = pad.y.min(), pad.y.max()
        xmid, ymid = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        span = max(xmax - xmin, ymax - ymin, 1.0)
        # Work on a copy so that adding the super-triangle does not touch
        # `pad`.
        work = pad[:]
        # Add a super triangle.
        work.append(xmid - 64.0 * span, ymid - 32.0 * span)
        work.append(xmid + 64.0 * span, ymid - 32.0 * span)
        work.append(xmid, ymid + 64.0 * span)
        # Create the Delaunay triangulation builder and add the super triangle
        # to it.
        builder = core.DelaunayBuilder()
        rr, ux, uy, uz = work.circumcircle(npt, npt + 1, npt + 2)
        builder.add(ia=npt, ib=npt + 1, ic=npt + 2, rr=rr, ux=ux, uy=uy, uz=uz)
        # The default relative tolerance of the builder counts a point on a
        # circle as outside, so cocircular points do not make the result
        # depend on rounding.
        builder.populate_from_points(pad=work, first=0, last=npt)
        trinodes = builder.compute_triangle_nodes(pad=work, nnode=npt)
        return trinodes

    @classmethod
    def _split_double_boundary(cls, pad):
        """Triangulate the :class:`PointPad` ``pad``, appending Steiner
        points into it until no cell touches the boundary with more than
        one face; returns the triangles.

        A cell with two single-shared (boundary) edges is a corner ear with
        a single interior neighbour, which the CESE solver dislikes.  A
        Steiner point at the ear's centroid keeps the next Delaunay pass
        from rebuilding it; the boundary nodes never move, so the
        classification is unchanged.  One pass suffices here; the cap only
        bounds pathological input.
        """
        for _ in range(16):
            tris = cls._triangulate(pad).ndarray.tolist()
            shared = {}
            for tri in tris:
                for it in range(3):
                    edge = frozenset((tri[it], tri[(it + 1) % 3]))
                    shared[edge] = shared.get(edge, 0) + 1
            extra = []
            for tri in tris:
                on_boundary = sum(
                    1 for it in range(3)
                    if shared[frozenset((tri[it], tri[(it + 1) % 3]))] == 1)
                if on_boundary >= 2:
                    extra.append((sum(pad.x_at(iv) for iv in tri) / 3.0,
                                  sum(pad.y_at(iv) for iv in tri) / 3.0))
            if not extra:
                return tris
            for xc, yc in extra:
                pad.append(xc, yc)
        raise RuntimeError("boundary-cell refinement did not converge")

    @staticmethod
    def classify_boundary(mh, tol=1e-9):
        """Bucket the boundary faces of ``mh`` by domain edge.

        A boundary face has no neighbour cell, i.e. ``fccls(ifc, 1) < 0``.
        Each is classified by its face-centre ``fccnd`` position and
        outward ``fcnml`` direction into the left (``x == xmin``, normal
        in -x), top (``y == ymax``, normal in +y), bottom, and right
        (``x == xmax``, normal in +x) edges.  The edge
        extrema come from the boundary face centres because ``ndcrd`` also
        carries extrapolated ghost-node coordinates that overshoot the real
        edges.

        Returns ``(left, top, bottom, right)`` as sorted face-index lists
        ready to feed :meth:`~solvcon.core.StaticMesh.add_bc`.
        """
        bfaces = [ifc for ifc in range(mh.nface) if mh.fccls[ifc, 1] < 0]
        xcs = [mh.fccnd[ifc, 0] for ifc in bfaces]
        ycs = [mh.fccnd[ifc, 1] for ifc in bfaces]
        xmin, xmax, ymax = min(xcs), max(xcs), max(ycs)
        left, top, bottom, right = [], [], [], []
        for ifc in bfaces:
            xc, yc = mh.fccnd[ifc, 0], mh.fccnd[ifc, 1]
            nx, ny = mh.fcnml[ifc, 0], mh.fcnml[ifc, 1]
            if abs(xc - xmin) <= tol and nx < 0.0:
                left.append(ifc)
            elif abs(xc - xmax) <= tol and nx > 0.0:
                right.append(ifc)
            elif abs(yc - ymax) <= tol and ny > 0.0:
                top.append(ifc)
            else:
                bottom.append(ifc)
        return sorted(left), sorted(top), sorted(bottom), sorted(right)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
