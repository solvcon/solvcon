# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The parameters of a reflection run, and the checks that keep them
valid.
"""

import dataclasses
import math
import numbers

from ._driver import check_reflection_solvable
from ._session import ReflectionSession

__all__ = [
    'ReflectionSetup',
    'max_deflection',
]

#: The deflection angle, in degrees, the bisection of
#: :func:`max_deflection` brackets from above; no gas reflects past it.
_ANGLE_CEILING = 90.0


def _attached(gamma, mach, angle):
    """Return whether both shocks stay attached at ``angle``.

    The shock relations take square roots that go negative where the
    deflection is out of reach, so an arithmetic error reads as detached.
    """
    try:
        return check_reflection_solvable(gamma, mach, angle)
    except (ValueError, ZeroDivisionError):
        return False


def max_deflection(gamma, mach, tol=1e-3):
    """Return the largest deflection angle, in degrees, at which the
    reflection of a ``mach`` stream of ``gamma`` keeps both shocks attached,
    to within ``tol`` below the true bound.

    The reflected shock turns a slower stream than the incident one, so it
    detaches first, and every angle under the bound attaches both; that is
    what lets a bisection find it.
    """
    lo, hi = 0.0, _ANGLE_CEILING
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if _attached(gamma, mach, mid):
            lo = mid
        else:
            hi = mid
    return lo


def _text(value):
    """Return ``value`` as short as it reads, ``1e-6`` and not ``1e-06``."""
    text = f"{value:g}"
    if 'e' in text:
        mantissa, exponent = text.split('e')
        text = f"{mantissa}e{int(exponent)}"
    return text


@dataclasses.dataclass(frozen=True)
class ReflectionSetup(object):
    """The parameters of one reflection run, as the forms edit them: the
    free-stream state the problem is posed with and the numerics it is
    marched with.  They are not data the run computes.

    A setup holds whatever it is given, so a form can carry an invalid
    edit.  :meth:`check` names what is wrong, field by field, with the
    range each field has to lie in, and :meth:`session` builds the
    :class:`~._session.ReflectionSession` of a valid setup only.
    """

    #: Ratio of specific heats.
    gamma: float = 1.4
    #: Free-stream density.
    density: float = 1.0
    #: Free-stream pressure.
    pressure: float = 1.0
    #: Free-stream Mach number.
    mach: float = 3.0
    #: Flow deflection across the incident shock, in degrees.
    angle: float = 10.0
    #: Mesh intervals along x.
    nx: int = 64
    #: Mesh intervals along y.
    ny: int = 16
    #: The time step of the march.
    time_increment: float = 2e-3
    #: The mesh flavor, one of :attr:`CELL_TYPES`.
    cell_type: str = 'unstructured'

    #: Every field, in the order of the forms and of :meth:`check`.
    FIELDS = ('gamma', 'density', 'pressure', 'mach', 'angle',
              'nx', 'ny', 'time_increment', 'cell_type')
    #: The fields of the free-stream group.
    FREE_STREAM = FIELDS[:5]
    #: The fields of the numerics group.
    NUMERICS = FIELDS[5:]
    #: The fields that take whole numbers.
    INTEGERS = ('nx', 'ny')
    #: The mesh flavors the mesher builds.
    CELL_TYPES = ('unstructured', 'quad', 'triangle')
    #: The closed range of each number.  A density or pressure spans
    #: decades either way, the time step stays under the domain height, and
    #: the mesh stays coarse enough to march at the frame rate.
    RANGES = {'gamma': (1.01, 3.0), 'density': (1e-3, 1e6),
              'pressure': (1e-3, 1e6), 'mach': (1.1, 20.0),
              'angle': (0.5, 45.0), 'nx': (4, 1024), 'ny': (2, 256),
              'time_increment': (1e-6, 1.0)}
    #: What a form and a message call each field.
    TITLES = {'gamma': "Gamma", 'density': "Density", 'pressure': "Pressure",
              'mach': "Mach", 'angle': "Deflection angle", 'nx': "nx",
              'ny': "ny", 'time_increment': "Time step",
              'cell_type': "Cell type"}

    def replace(self, **values):
        """Return a copy with ``values`` in place, unchecked.

        :raise TypeError: For a name that is not in :attr:`FIELDS`.
        """
        return dataclasses.replace(self, **values)

    def check(self):
        """Return what is wrong, ``{field: message}`` in :attr:`FIELDS`
        order, or an empty dict for a valid setup.

        Every field is checked on its own first.  The angle is then checked
        against the bound :func:`max_deflection` gives, but only while the
        gamma, the Mach number, and the angle are each in range, since the
        bound of a bad one would mislead.
        """
        errors = {}
        for name in self.FIELDS:
            message = self._check_field(name)
            if message:
                errors[name] = message
        if not errors.keys() & {'gamma', 'mach', 'angle'}:
            message = self._check_attached()
            if message:
                errors['angle'] = message
        return errors

    def _check_field(self, name):
        """Return what is wrong with one field on its own, or ''."""
        value = getattr(self, name)
        title = self.TITLES[name]
        if 'cell_type' == name:
            if value in self.CELL_TYPES:
                return ''
            return (f"{title} must be one of {', '.join(self.CELL_TYPES)}; "
                    f"got {value!r}.")
        lo, hi = self.RANGES[name]
        span = f"{_text(lo)} to {_text(hi)}"
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return f"{title} must be a number in {span}; got {value!r}."
        if not math.isfinite(value):
            return f"{title} must be a number in {span}; got {value}."
        if name in self.INTEGERS and value != int(value):
            return f"{title} must be a whole number in {span}; got {value:g}."
        if not lo <= value <= hi:
            return f"{title} must lie in {span}; got {_text(value)}."
        return ''

    def _check_attached(self):
        """Return why the angle detaches a shock at this Mach number and
        gamma, naming the largest angle that does not, or ''."""
        if _attached(self.gamma, self.mach, self.angle):
            return ''
        title = self.TITLES['angle']
        where = f"at Mach {_text(self.mach)} and gamma {_text(self.gamma)}"
        # Floored, so the bound read off the message is itself admissible.
        bound = math.floor(100.0 * max_deflection(self.gamma, self.mach))
        bound /= 100.0
        lo, hi = self.RANGES['angle']
        if bound < lo:
            return (f"{title} must stay below {bound:.2f} deg {where}, "
                    f"under its range of {_text(lo)} to {_text(hi)}; raise "
                    f"Mach or lower gamma.")
        return (f"{title} must stay below {bound:.2f} deg {where}, or the "
                f"reflected shock detaches.")

    def title(self):
        """Return the window title, ``Oblique shock reflection: Mach 3, 10
        deg, 64 x 16``."""
        return f"Oblique shock reflection: {self.summary()}"

    def summary(self):
        """Return the short name of the run, ``Mach 3, 10 deg, 64 x 16``."""
        return (f"Mach {_text(self.mach)}, {_text(self.angle)} deg, "
                f"{_text(self.nx)} x {_text(self.ny)}")

    def session(self, max_steps=ReflectionSession.MAX_STEPS,
                steps_per_chunk=ReflectionSession.STEPS_PER_CHUNK):
        """Build the :class:`~._session.ReflectionSession` of the setup,
        ready to march from step 0.

        :raise ValueError: Naming the first field :meth:`check` finds
            wrong, so no session is built of an invalid setup.
        """
        errors = self.check()
        if errors:
            raise ValueError(next(iter(errors.values())))
        return ReflectionSession(
            gamma=float(self.gamma), density=float(self.density),
            pressure=float(self.pressure), mach=float(self.mach),
            angle=float(self.angle), cell_type=self.cell_type,
            time_increment=float(self.time_increment), nx=int(self.nx),
            ny=int(self.ny), steps_per_chunk=steps_per_chunk,
            max_steps=max_steps)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
