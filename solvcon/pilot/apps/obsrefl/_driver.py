# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Mesh, boundary-condition groups, and solver driver for the oblique-shock
reflection.
"""

import math

from .... import core

__all__ = [
    'ObliqueShock',
    'ObliqueShockMesher',
    'ObliqueShockRelation',
]


class ObliqueShockMesher(object):
    """Generate the mesh for the oblique-shock reflection.

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
            cells = [(3,) + tri
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

    @staticmethod
    def _circumcircle(pts, ia, ib, ic):
        """Circumcenter and squared circumradius of triangle (ia, ib, ic)."""
        ax, ay = pts[ia]
        bx, by = pts[ib]
        cx, cy = pts[ic]
        dd = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
        a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
        ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / dd
        uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / dd
        return ux, uy, (ax - ux) ** 2 + (ay - uy) ** 2

    @classmethod
    def _triangulate(cls, pad):
        """Bowyer-Watson Delaunay triangulation of the :class:`PointPad`
        ``pad``; returns CCW index triples.

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
        pts = [(pad.x_at(ip), pad.y_at(ip)) for ip in range(npt)]
        xs = [px for px, py in pts]
        ys = [py for px, py in pts]
        xmid = (min(xs) + max(xs)) / 2.0
        ymid = (min(ys) + max(ys)) / 2.0
        span = max(max(xs) - min(xs), max(ys) - min(ys), 1.0)
        pts += [(xmid - 64.0 * span, ymid - 32.0 * span),
                (xmid + 64.0 * span, ymid - 32.0 * span),
                (xmid, ymid + 64.0 * span)]
        # Triangles keyed by CCW vertex triple, valued by circumcircle.
        cc = {(npt, npt + 1, npt + 2):
              cls._circumcircle(pts, npt, npt + 1, npt + 2)}
        for ip in range(npt):
            px, py = pts[ip]
            bad = [tri for tri, (ux, uy, rr) in cc.items()
                   if (px - ux) ** 2 + (py - uy) ** 2 < rr * (1.0 - 1e-12)]
            dedges = set()
            for tri in bad:
                ta, tb, tc = tri
                dedges.update(((ta, tb), (tb, tc), (tc, ta)))
                del cc[tri]
            # The cavity rim is the directed edges whose reverse was not
            # carved; they run CCW, so fanning them to the point keeps the
            # triangles CCW.
            for ea, eb in dedges:
                if (eb, ea) not in dedges:
                    cc[(ea, eb, ip)] = cls._circumcircle(pts, ea, eb, ip)
        tris = []
        for tri in cc:
            if max(tri) >= npt:
                continue
            (ax, ay), (bx, by), (cx, cy) = (pts[iv] for iv in tri)
            if (bx - ax) * (cy - ay) - (cx - ax) * (by - ay) < 0.0:
                tri = (tri[0], tri[2], tri[1])
            tris.append(tri)
        return tris

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
            tris = cls._triangulate(pad)
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
        (``x == xmax``, normal in +x) edges.  The driver imposes the free
        stream at the left, the post-shock state at the top, the slip wall
        at the bottom, and the non-reflective outflow at the right.  The
        edge extrema come from the boundary face centres because ``ndcrd``
        also carries extrapolated ghost-node coordinates that overshoot
        the real edges.

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


class ObliqueShockRelation(object):
    """Calculate the flow-property jumps across an oblique shock.

    (Not yet validated.)

    ``beta`` is the shock angle and ``theta`` the flow-deflection angle,
    both in radians and measured from the upstream flow direction; the
    formulas follow chapter 4 of Anderson, Modern Compressible Flow (3rd
    ed.).  Ported from the gas parcel of the legacy solvcon code base.

    :ivar gamma: Ratio of specific heats.
    """

    def __init__(self, gamma):
        self.gamma = gamma

    def calc_density_ratio(self, mach1, beta):
        """Return the density ratio rho2/rho1 across a shock of angle
        ``beta`` at upstream Mach number ``mach1``."""
        gamma = self.gamma
        mn1sq = (mach1 * math.sin(beta)) ** 2
        return (gamma + 1) * mn1sq / ((gamma - 1) * mn1sq + 2)

    def calc_pressure_ratio(self, mach1, beta):
        """Return the pressure ratio p2/p1 across a shock of angle ``beta``
        at upstream Mach number ``mach1``."""
        gamma = self.gamma
        mn1sq = (mach1 * math.sin(beta)) ** 2
        return 1 + 2 * gamma / (gamma + 1) * (mn1sq - 1)

    def calc_temperature_ratio(self, mach1, beta):
        """Return the temperature ratio T2/T1 across a shock of angle
        ``beta`` at upstream Mach number ``mach1``."""
        return (self.calc_pressure_ratio(mach1, beta)
                / self.calc_density_ratio(mach1, beta))

    def calc_normal_dmach(self, mach_n1):
        """Return the downstream Mach number normal to the shock from the
        normal upstream Mach number ``mach_n1``."""
        gamma = self.gamma
        mn1sq = mach_n1 * mach_n1
        return math.sqrt(((gamma - 1) * mn1sq + 2)
                         / (2 * gamma * mn1sq - (gamma - 1)))

    def calc_dmach(self, mach1, beta=None, theta=None, delta=1):
        """Return the downstream Mach number from the upstream Mach number
        ``mach1`` and either the shock angle ``beta`` or the deflection
        angle ``theta`` (with ``delta`` selecting the weak, 1, or strong,
        0, shock branch)."""
        if (beta is None) == (theta is None):
            raise ValueError(
                f"got (beta={beta}, theta={theta}), "
                f"but I need to take either beta or theta")
        if beta is None:
            beta = self.calc_shock_angle(mach1, theta, delta=delta)
        if theta is None:
            theta = self.calc_flow_angle(mach1, beta)
        mach_n1 = mach1 * math.sin(beta)
        return self.calc_normal_dmach(mach_n1) / math.sin(beta - theta)

    def calc_flow_angle(self, mach1, beta):
        """Return the deflection angle theta from the upstream Mach number
        ``mach1`` and the shock angle ``beta``."""
        return math.atan(self.calc_flow_tangent(mach1, beta))

    def calc_flow_tangent(self, mach1, beta):
        """Return tan(theta) through the theta-beta-M relation."""
        gamma = self.gamma
        m1sq = mach1 * mach1
        return (2 / math.tan(beta) * (m1sq * math.sin(beta) ** 2 - 1)
                / (m1sq * (gamma + math.cos(2 * beta)) + 2))

    def calc_shock_angle(self, mach1, theta, delta=1):
        """Return the shock angle beta from the upstream Mach number
        ``mach1`` and the deflection angle ``theta`` (with ``delta``
        selecting the weak, 1, or strong, 0, shock branch)."""
        return math.atan(self.calc_shock_tangent(mach1, theta, delta))

    def calc_shock_tangent(self, mach1, theta, delta):
        """Return tan(beta) through the closed-form inversion of the
        theta-beta-M relation."""
        gamma = self.gamma
        m1sq = mach1 * mach1
        lmbd, chi = self.calc_shock_tangent_aux(mach1, theta)
        num = (m1sq - 1 + 2 * lmbd
               * math.cos((4 * math.pi * delta + math.acos(chi)) / 3))
        den = 3 * (1 + (gamma - 1) / 2 * m1sq) * math.tan(theta)
        return num / den

    def calc_shock_tangent_aux(self, mach1, theta):
        """Return the (lambda, chi) auxiliary pair of the closed-form
        theta-beta-M inversion used by :meth:`calc_shock_tangent`."""
        gamma = self.gamma
        m1sq = mach1 * mach1
        tansq = math.tan(theta) ** 2
        disc = ((m1sq - 1) ** 2
                - 3 * (1 + (gamma - 1) / 2 * m1sq)
                * (1 + (gamma + 1) / 2 * m1sq) * tansq)
        if disc <= 0.0:
            raise ValueError(
                f"no attached shock for mach1={mach1:g} and "
                f"theta={math.degrees(theta):g} deg")
        lmbd = math.sqrt(disc)
        chi = ((m1sq - 1) ** 3
               - 9 * (1 + (gamma - 1) / 2 * m1sq)
               * (1 + (gamma - 1) / 2 * m1sq + (gamma + 1) / 4 * m1sq * m1sq)
               * tansq) / lmbd ** 3
        return lmbd, chi


class ObliqueShock(object):
    """Drive the CESE Euler solver over the oblique-shock reflection.
    """

    def __init__(self):
        self.gamma = None
        self.density = None
        self.pressure = None
        self.mach = None
        self.speedofsound = None
        self.velocity = None
        self.relation = None
        self.theta = None
        self.shock_angle = None
        self.density2 = None
        self.pressure2 = None
        self.mach2 = None
        self.velocity2 = None
        self.shock_angle2 = None
        self.density3 = None
        self.pressure3 = None
        self.mach3 = None
        self.velocity3 = None
        self.mesher = None
        self.mesh = None
        # Numerical solver core (EulerCore).
        self.svr = None

    def build_constant(self, gamma=1.4, density=1.0, pressure=1.0, mach=3.0,
                       angle=10.0):
        """Fix the flow states on the two sides of the incident shock.

        The free stream (zone 1) enters horizontally at the given Mach
        number; ``angle`` is the flow deflection across the incident shock
        in degrees.  The oblique-shock relations give the post-shock state
        (zone 2), whose velocity points ``angle`` below horizontal; the
        driver imposes it at the top boundary to anchor the incident
        shock.  The reflected shock turns zone 2 back to horizontal, and
        the same relations give the state behind it (zone 3), which the
        steady solution has to reach; it is kept for validation and
        display, not imposed anywhere.
        """
        self.gamma = gamma
        self.density = density
        self.pressure = pressure
        self.mach = mach
        self.speedofsound = math.sqrt(gamma * pressure / density)
        self.velocity = mach * self.speedofsound
        self.relation = ObliqueShockRelation(gamma=gamma)
        self.theta = theta = math.radians(angle)
        self.shock_angle = beta = self.relation.calc_shock_angle(mach, theta)
        self.density2 = density * self.relation.calc_density_ratio(mach, beta)
        self.pressure2 = (
            pressure * self.relation.calc_pressure_ratio(mach, beta))
        self.mach2 = self.relation.calc_dmach(mach, beta=beta)
        speed2 = self.mach2 * math.sqrt(gamma * self.pressure2 / self.density2)
        self.velocity2 = (speed2 * math.cos(theta),
                          -speed2 * math.sin(theta))
        mach2 = self.mach2
        self.shock_angle2 = beta2 = self.relation.calc_shock_angle(
            mach2, theta)
        self.density3 = (
            self.density2 * self.relation.calc_density_ratio(mach2, beta2))
        self.pressure3 = (
            self.pressure2 * self.relation.calc_pressure_ratio(mach2, beta2))
        self.mach3 = self.relation.calc_dmach(mach2, beta=beta2)
        speed3 = self.mach3 * math.sqrt(gamma * self.pressure3 / self.density3)
        self.velocity3 = (speed3, 0.0)

    def build_numerical(self, cell_type='unstructured', time_increment=2.e-3,
                        sigma0=3.0, taumin=0.0, tauscale=1.0, **mesher_kw):
        """After :meth:`build_constant` is done, build the numerical solver
        :attr:`svr` over the selected mesh flavor.
        """
        if None is self.gamma:
            raise ValueError("constants are not set; call build_constant()")
        self.mesher = ObliqueShockMesher(**mesher_kw)
        self.mesh = self.mesher.make_mesh(cell_type=cell_type)
        # The core prepares the CE geometry (prepare_ce) on construction.
        svr = core.EulerCore(mesh=self.mesh, time_increment=time_increment)
        svr.sigma0 = sigma0
        svr.taumin = taumin
        svr.tauscale = tauscale
        svr.init_solution(gamma=self.gamma, rho=self.density,
                          v=[self.velocity, 0.0], p=self.pressure)
        # The mesher attached one named group per domain edge; read the
        # face lists back from the mesh instead of re-classifying.
        left, top, bottom, right = (
            self.mesh.bc(name).facn.ndarray[:, 0].tolist()
            for name in self.mesher.BOUNDARY_NAMES)
        svr.add_inlet(left, value=[self.density, self.velocity, 0.0,
                                   self.pressure, self.gamma])
        svr.add_inlet(top, value=[self.density2, self.velocity2[0],
                                  self.velocity2[1], self.pressure2,
                                  self.gamma])
        svr.add_slipwall(bottom)
        svr.add_nonrefl(right)
        # Prime the ghost rows from the initial interior state so the first
        # substep does not read zero-filled ghosts.
        svr.bc_soln()
        svr.bc_dsoln()
        self.svr = svr

    def march(self, steps):
        """March the solver the requested number of full CESE steps."""
        self.svr.march(steps=steps)

    def zone_states(self):
        """Return the analytic ``(rho, vx, vy, p)`` of zones 1, 2, and 3.

        Zone 1 is the free stream, zone 2 sits between the incident and the
        reflected shock, and zone 3 sits behind the reflected shock.  The
        steady solution has to hold these values; a display can derive any
        scalar field from them for comparison against the computed field.
        """
        return [(self.density, self.velocity, 0.0, self.pressure),
                (self.density2, self.velocity2[0], self.velocity2[1],
                 self.pressure2),
                (self.density3, self.velocity3[0], self.velocity3[1],
                 self.pressure3)]

    def shock_path(self):
        """Return the analytic shock polyline over the built mesh.

        Three corners: where the incident shock enters at the upper-left
        corner, where it reflects off the bottom wall, and where the
        reflected shock leaves the domain.  Needs :meth:`build_numerical`,
        which sets the mesher whose extents the path is cut to.
        """
        if None is self.mesher:
            raise ValueError("mesh is not built; call build_numerical()")
        msh = self.mesher
        xhit = msh.x0 + (msh.y1 - msh.y0) / math.tan(self.shock_angle)
        if xhit >= msh.x1:
            # The domain is too short for the reflection; the incident
            # shock leaves through the outflow.
            yout = msh.y1 - (msh.x1 - msh.x0) * math.tan(self.shock_angle)
            return [(msh.x0, msh.y1), (msh.x1, yout)]
        # The reflected shock runs at shock_angle2 from the zone-2 flow,
        # which is already theta below horizontal.
        slope = math.tan(self.shock_angle2 - self.theta)
        xend = min(msh.x1, xhit + (msh.y1 - msh.y0) / slope)
        return [(msh.x0, msh.y1), (xhit, msh.y0),
                (xend, msh.y0 + (xend - xhit) * slope)]

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
