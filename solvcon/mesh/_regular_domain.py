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
        self.bbox = core.BoundBox3dFp64(
            min_x=ll[0], min_y=ll[1], min_z=0.0,
            max_x=ur[0], max_y=ur[1], max_z=0.0)

    @property
    def x0(self):
        return self.bbox.min_x

    @property
    def y0(self):
        return self.bbox.min_y

    @property
    def x1(self):
        return self.bbox.max_x

    @property
    def y1(self):
        return self.bbox.max_y

    @property
    def cell_extent(self):
        """The ``(width, height)`` of one grid box.

        Each element flavor fills a box with one or two cells, so the box is
        the resolution of the mesh whatever the flavor, and it is what a
        measurement over the mesh has to be cut to.
        """
        bbox = self.bbox
        return ((bbox.max_x - bbox.min_x) / self.nx,
                (bbox.max_y - bbox.min_y) / self.ny)

    def _node(self, it, jt):
        bbox = self.bbox
        return (bbox.min_x + it * (bbox.max_x - bbox.min_x) / self.nx,
                bbox.min_y + jt * (bbox.max_y - bbox.min_y) / self.ny)

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
            builder = core.DelaunayBuilder(pad=nodes)
            # An ear is a cell with two boundary edges, and so a single
            # interior neighbor.
            max_iter = 16
            for _ in range(max_iter):
                tris = builder.triangulate()
                ears = builder.calc_ear_centroids(tris=tris)
                if ears.shape[0] == 0:
                    break
                nodes.append(x=ears[:, 0], y=ears[:, 1])
            else:
                raise RuntimeError(
                    "boundary-cell refinement did not converge in "
                    f"{max_iter} iterations")
            cells = [(3, *tri) for tri in tris.ndarray.tolist()]
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
