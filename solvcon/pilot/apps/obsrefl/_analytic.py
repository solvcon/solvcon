# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Analytic solution of the oblique-shock reflection.

The analytic solution states where the two shocks stand and what the flow is
between them, so every measurement here is a comparison: the flow in each
zone against the state the zone has to hold, the fitted incident-shock angle
against the angle the relations give, and a line profile against the
three-zone step it converges to::

    y1 +--------------------------------------------------------+
       |'-._                                                    |
       |    '-._              zone 2                            |
       |        '-._                                        _,-'|
       |            '-._                                _,-'    |
       |                '-._                       _,-'         |
       |    zone 1          '-._              _,-'              |
       |                        '-._      _,-'      zone 3      |
    y0 +----------------------------X_,-'-----------------------+
       x0                           ^ reflection point         x1

The free stream (zone 1) enters from the left and crosses the incident
shock, which runs from the upper-left corner down to the slip wall along
y0.  The reflected shock leaves the wall at the reflection point and turns
the flow back to horizontal, so zone 3 sits under it and zone 2 fills the
wedge between the two.  The top boundary carries the zone-2 state, which is
what anchors the incident shock where it stands.
"""

import dataclasses
import math

import numpy as np

from .... import core
from ....multidim.euler import EulerField

__all__ = [
    'AngleFit',
    'HorizontalProfile',
    'Reflection',
    'WallPoint',
    'ZoneInfo',
]


@dataclasses.dataclass
class ZoneInfo(object):
    """One zone of :meth:`Reflection.zone_info`."""

    #: Everything :attr:`zone` accepts.  The analytic solution has three
    #: zones, and None is for a reading that names none of them.
    ZONES = (None, 1, 2, 3)

    #: Stored behind :attr:`zone`, which is what checks it.
    _zone: int
    #: Body cells the average ran over; zero when the margin empties the
    #: zone, which leaves the value ``nan``.
    count: int
    #: Mean of the computed field over those cells.
    computed: float
    #: Value the analytic solution holds in the zone.
    analytic: float
    #: Error of :attr:`computed` relative to :attr:`analytic`.
    error: float

    def __post_init__(self):
        # The generated __init__ writes the field behind the accessor, so
        # run the value back through it once the record is built.
        self.zone = self._zone

    @property
    def zone(self):
        """Which zone the reading is of, counted 1, 2, 3 downstream, or
        None where it is of no single zone."""
        return self._zone

    @zone.setter
    def zone(self, value):
        if value not in self.ZONES:
            raise ValueError(
                f"zone must be one of {self.ZONES}, not {value!r}")
        self._zone = value


@dataclasses.dataclass
class AngleFit(object):
    """The outcome of :meth:`Reflection.fit_incident_angle`."""

    #: Incident-shock angle fitted from the field, in degrees.
    degree: float
    #: Angle the oblique-shock relations give, in degrees.
    analytic: float
    #: Error of :attr:`degree` relative to :attr:`analytic`.
    error: float
    #: Crossings the fit ran through; under two leave the angle ``nan``.
    npoint: int


@dataclasses.dataclass
class WallPoint(object):
    """The outcome of :meth:`Reflection.reflection_point`."""

    #: Abscissa where the fitted incident shock meets the wall.
    x: float
    #: Abscissa where the analytic path turns; ``nan`` on a domain too
    #: short to hold the reflection.
    analytic: float
    #: Error of :attr:`x` relative to :attr:`analytic`.
    error: float


@dataclasses.dataclass
class HorizontalProfile(object):
    """A horizontal cut of the domain, as :meth:`Reflection.profile` takes
    it."""

    #: Abscissa of each cell on the cut, ascending.
    x: np.ndarray
    #: Computed field at those abscissae.
    computed: np.ndarray
    #: Analytic three-zone step at the same abscissae.
    analytic: np.ndarray


class Reflection(object):
    """Measure one reflection run against the analytic solution.

    The measurements are read-only and are meant to be taken while the run
    marches, so none of them raises on a field that has not developed the
    shocks yet; a measurement without enough data reports ``nan`` instead.

    :ivar shock: The :class:`~._driver.ObliqueShock` being measured.
    :ivar field: The :class:`~solvcon.multidim.euler.EulerField` reading its
        body cells.
    :ivar arms: The analytic shock path as a
        :class:`~solvcon.core.SegmentPadFp64`: the incident shock, and the
        reflected shock when the domain holds one.
    :ivar margin: Clearance kept from the shocks, as passed to
        :meth:`__init__`.
    """

    #: Default ``margin`` of :meth:`__init__`.
    MARGIN_FRACTION = 0.1

    def __init__(self, shock, margin=MARGIN_FRACTION):
        """Read one run against the analytic solution of its problem.

        :param shock: The :class:`~._driver.ObliqueShock` whose solver holds
            the field to measure.
        :param margin: Clearance kept from the shocks, as a fraction of the
            domain height.  The shock wave captured needs a few cells, while
            the analytic path is a line of zero thickness, so a zone average
            has to stand back from it.  When calculating the average value
            in a zone, only take the cells that is away from the shock.  The
            margin is the clearance.
        """
        if None is shock.svr:
            raise ValueError("solver is not built; call build_numerical()")
        self.shock = shock
        self.field = EulerField(shock.svr, shock.mesh)
        self.margin = margin
        # The analytic path is fixed once the mesh is built, so its corners
        # are walked once into the segments the measurements read.
        # TODO: SegmentPad takes a Segment3d built from two Point3d, unlike
        # PointPad, which appends loose coordinates.  An append(x0, y0, x1,
        # y1) overload would drop the two constructions per arm.
        self.arms = core.SegmentPadFp64(ndim=2)
        path = shock.shock_path()
        for head, tail in zip(path[:-1], path[1:]):
            # TODO: the endpoints are 3D even on a 2D pad, so a plane user
            # pins z twice per segment.  A Segment2d, or a pad that carries
            # only ndim coordinates per point, would take the zeros away.
            self.arms.append(core.Segment3dFp64(
                core.Point3dFp64(head[0], head[1], 0.0),
                core.Point3dFp64(tail[0], tail[1], 0.0)))

    @property
    def incident(self):
        """The segment the incident shock runs along."""
        return self.arms[0]

    @property
    def reflected(self):
        """The segment the reflected shock runs along, or None."""
        return self.arms[1] if self.has_reflection else None

    @staticmethod
    def _relative(value, target):
        """Return the error of ``value`` relative to ``target``."""
        return (value - target) / target if target else float('nan')

    # TODO: the three helpers below are the plane geometry of a segment, not
    # of this problem, and every user of Segment3d has to write them again.
    # Slope, direction, and normal belong on the class, and the offset wants
    # to stay vectorised: a point-to-line distance taking arrays.

    @staticmethod
    def _slope(seg):
        """Return the slope of the line ``seg`` lies on."""
        return (seg.y1 - seg.y0) / (seg.x1 - seg.x0)

    @classmethod
    def _offset(cls, seg, xs, ys):
        """Return the perpendicular distance from ``seg``, positive above
        it."""
        slope = cls._slope(seg)
        return (ys - seg.y0 - slope * (xs - seg.x0)) / math.hypot(1.0, slope)

    @classmethod
    def _abscissa(cls, seg, height):
        """Return where the line of ``seg`` reaches ``height``."""
        return seg.x0 + (height - seg.y0) / cls._slope(seg)

    @property
    def has_reflection(self):
        """Whether the incident shock reflects inside the domain.

        A domain too short for the reflection carries the incident shock out
        through the outflow, leaving no reflected shock and no zone 3, so
        the path is one segment instead of two.
        """
        return 2 == len(self.arms)

    def zone_masks(self, margin=None):
        """Return the body-cell masks of zones 1, 2, and 3.

        A cell belongs to a zone when its centroid is on the zone's side of
        both shocks and stands at least ``margin`` (:attr:`margin` by
        default, in fractions of the domain height) away from them.  The
        cells inside the margin belong to no zone, so the three masks
        partition the domain only when the margin is zero.
        """
        margin = self.margin if None is margin else margin
        mesher = self.shock.mesher
        gap = margin * (mesher.y1 - mesher.y0)
        cnd = self.field.centroid
        xs, ys = cnd[:, 0], cnd[:, 1]
        incident = self._offset(self.incident, xs, ys)
        # The reflected shock runs up from the wall, so the line it lies on
        # passes below the domain floor upstream of the reflection point and
        # cuts zone 3 out on its own.
        reflected = (self._offset(self.reflected, xs, ys)
                     if self.has_reflection
                     else np.full(xs.shape, np.inf, dtype='float64'))
        return [incident < -gap,
                (incident > gap) & (reflected > gap),
                reflected < -gap]

    def zone_fields(self):
        """Return every :attr:`EulerField.FIELDS` value of zones 1, 2, and 3.

        The states of :meth:`ObliqueShock.zone_states` are primitives, so
        the four they carry need no work and the other three follow from
        the same ideal-gas relations the field reader applies to the
        computed solution.
        """
        gamma = self.shock.gamma
        rho, velx, vely, pressure = np.array(
            self.shock.zone_states(), dtype='float64').T
        speed2 = velx * velx + vely * vely
        return {'density': rho, 'velx': velx, 'vely': vely,
                'speed': np.sqrt(speed2), 'pressure': pressure,
                'mach': np.sqrt(speed2 / (gamma * pressure / rho)),
                'total_energy': pressure / (gamma - 1.0) + 0.5 * rho * speed2}

    def zone_field(self, name):
        """Return the named field's analytic value in zones 1, 2, and 3."""
        fields = self.zone_fields()
        if name not in fields:
            raise ValueError(f"unknown field '{name}'")
        return fields[name]

    def zone_conserved(self):
        """Return the analytic zone states as conserved rows, ``[3, neq]``.

        Packing the zones the way the solver holds them is what lets a
        caller seed the analytic answer straight into a solution field.
        """
        fields = self.zone_fields()
        rho = fields['density']
        return np.column_stack([rho, rho * fields['velx'],
                                rho * fields['vely'],
                                fields['total_energy']])

    def zone_info(self, name='density'):
        """Return the per-zone comparison of the named field.

        Each :class:`ZoneInfo` holds the mean of the computed field over
        the zone's cells against the analytic value the steady solution has
        to reach.  A zone the margin leaves empty reports ``nan``.
        """
        values = self.field.field(name)
        analytic = self.zone_field(name)
        report = []
        for it, mask in enumerate(self.zone_masks()):
            count = int(mask.sum())
            computed = float(values[mask].mean()) if count else float('nan')
            target = float(analytic[it])
            report.append(ZoneInfo(it + 1, count, computed, target,
                                   self._relative(computed, target)))
        return report

    def crossings(self, nbin=None):
        """Return where the density crosses its mid value, ``[npoint, 2]``.

        Upstream of the reflection point the incident shock is the only
        discontinuity a vertical line meets, so the density steps once from
        the free stream to the post-shock state along it.  The domain up to
        that point is cut into ``nbin`` equal-width column bins, one per
        mesh column by default, and in each the body cells are walked
        upward from the wall until a pair straddles the mid density
        ``(rho1 + rho2) / 2``, which is interpolated linearly for the
        crossing height.  Walking upward picks the incident shock even when
        the reflected one has smeared into the bin.

        A bin whose cells all sit on one side carries no crossing and is
        dropped.  That is what leaves out the columns where the shock has
        not formed yet, and the ones it enters above the top row of cells.
        """
        mesher = self.shock.mesher
        # The incident arm ends at the reflection point, or at the outflow
        # on a domain too short to hold one.
        xend = self.incident.x1
        if None is nbin:
            nbin = max(1, round((xend - mesher.x0) / mesher.cell_extent[0]))
        cnd = self.field.centroid
        rho = self.field.density
        mid = 0.5 * (self.shock.density + self.shock.density2)
        edges = np.linspace(mesher.x0, xend, nbin + 1, dtype='float64')
        found = []
        for it in range(nbin):
            pick = ((cnd[:, 0] >= edges[it]) & (cnd[:, 0] < edges[it + 1]))
            order = np.argsort(cnd[pick, 1])
            ys, vs = cnd[pick, 1][order], rho[pick][order]
            above = np.nonzero(vs >= mid)[0]
            if not above.size or 0 == above[0]:
                continue
            ihi = above[0]
            ylo, yhi = ys[ihi - 1], ys[ihi]
            vlo, vhi = vs[ihi - 1], vs[ihi]
            found.append((0.5 * (edges[it] + edges[it + 1]),
                          ylo + (mid - vlo) * (yhi - ylo) / (vhi - vlo)))
        return np.array(found, dtype='float64').reshape(-1, 2)

    def fit_line(self, nbin=None):
        """Return the ``(slope, intercept, npoint)`` least-squares line
        through the :meth:`crossings`; the slope and the intercept are
        ``nan`` when fewer than two bins carry one."""
        pts = self.crossings(nbin)
        if len(pts) < 2:
            return float('nan'), float('nan'), len(pts)
        slope, intercept = np.polyfit(pts[:, 0], pts[:, 1], 1)
        return float(slope), float(intercept), len(pts)

    def fit_incident_angle(self, nbin=None):
        """Return the fitted incident-shock angle in degrees.

        The fitted line descends as it runs downstream, so the shock angle
        is the arctangent of the negated slope, measured from the free
        stream the same way :attr:`ObliqueShock.shock_angle` is.
        """
        slope, _, npoint = self.fit_line(nbin)
        analytic = math.degrees(self.shock.shock_angle)
        degree = (math.degrees(math.atan(-slope)) if slope < 0.0
                  else float('nan'))
        return AngleFit(degree, analytic, self._relative(degree, analytic),
                        npoint)

    def reflection_point(self, nbin=None):
        """Return where the fitted incident shock meets the wall.

        The analytic abscissa is the corner of the shock path; it is ``nan``
        on a domain too short to hold the reflection.
        """
        slope, intercept, _ = self.fit_line(nbin)
        mesher = self.shock.mesher
        xfit = ((mesher.y0 - intercept) / slope if slope < 0.0
                else float('nan'))
        analytic = self.reflected.x0 if self.has_reflection else float('nan')
        return WallPoint(xfit, analytic, self._relative(xfit, analytic))

    def profile(self, height, name='density', halfwidth=None):
        """Return the named field along the horizontal line at ``height``.

        The cells within ``halfwidth`` of the line are sorted by abscissa and
        paired with :meth:`analytic_profile` at the same abscissae, so the
        two curves of a plot share their sampling.  The default half a box
        picks the row the line runs through, and both rows when it runs
        along the seam between two.
        """
        if None is halfwidth:
            halfwidth = 0.5 * self.shock.mesher.cell_extent[1]
        cnd = self.field.centroid
        pick = np.abs(cnd[:, 1] - height) <= halfwidth
        order = np.argsort(cnd[pick, 0])
        xs = cnd[pick, 0][order]
        return HorizontalProfile(
            xs, self.field.field(name)[pick][order],
            self.analytic_profile(xs, height, name))

    def analytic_profile(self, xs, height, name='density'):
        """Return the analytic three-zone step at the given abscissae.

        A horizontal line at ``height`` leaves the free stream where the
        incident shock crosses it and enters zone 3 where the reflected one
        does, so the analytic field along it is a step function of the two
        crossings.
        """
        zones = self.zone_field(name)
        xs = np.asarray(xs)
        out = np.where(xs < self._abscissa(self.incident, height),
                       zones[0], zones[1])
        if self.has_reflection:
            out = np.where(xs >= self._abscissa(self.reflected, height),
                           zones[2], out)
        return out

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
