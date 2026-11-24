# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The parameters of a shock-tube run, and the checks that keep them
valid.
"""

import dataclasses
import math
import numbers

from ._riemann import ExactRiemann
from ._session import ShockTubeSession

__all__ = [
    'ShockTubeSetup',
    'number_text',
]


def number_text(value):
    """Return ``value`` as short as it reads, ``1e-6`` and not ``1e-06``.

    :param value: The number.
    """
    text = f"{value:g}"
    if 'e' in text:
        mantissa, exponent = text.split('e')
        text = f"{mantissa}e{int(exponent)}"
    return text


@dataclasses.dataclass(frozen=True)
class ShockTubeSetup(object):
    """Hold the parameters of one shock-tube run, as the forms edit them:
    the two gases at rest on either side of the diaphragm at x = 0, and the
    grid and the time step the run marches on.  The defaults pose Sod's
    problem.

    A setup holds whatever it is given, valid or not.  :meth:`check` names
    what is wrong, field by field, with the range each field has to lie
    in, and :meth:`session` builds the
    :class:`~._session.ShockTubeSession` of a valid setup only.
    """

    #: Density of the left gas.
    rho_left: float = 1.0
    #: Pressure of the left gas.
    p_left: float = 1.0
    #: Density of the right gas.
    rho_right: float = 0.125
    #: Pressure of the right gas.
    p_right: float = 0.1
    #: Ratio of specific heats.
    gamma: float = 1.4
    #: The left end of the tube.
    xmin: float = -10.0
    #: The right end of the tube.
    xmax: float = 10.0
    #: The grid points, an odd count.
    ncoord: int = 201
    #: The time step of the march.
    time_increment: float = 0.05

    #: Every field, in the order of the forms and of :meth:`check`.
    FIELDS = ('rho_left', 'p_left', 'rho_right', 'p_right', 'gamma',
              'xmin', 'xmax', 'ncoord', 'time_increment')
    #: The fields of the problem group.
    PROBLEM = FIELDS[:5]
    #: The fields of the numerics group.
    NUMERICS = FIELDS[5:]
    #: The fields that take whole numbers.
    INTEGERS = ('ncoord',)
    # A density or pressure spans decades either way, the tube stays
    # inside a range a float spans finely, and the grid stays coarse
    # enough to march at the frame rate.
    #: The closed range of each number.
    RANGES = {'rho_left': (1e-6, 1e6), 'p_left': (1e-6, 1e6),
              'rho_right': (1e-6, 1e6), 'p_right': (1e-6, 1e6),
              'gamma': (1.01, 3.0), 'xmin': (-1e6, 1e6),
              'xmax': (-1e6, 1e6), 'ncoord': (5, 20001),
              'time_increment': (1e-9, 1e3)}
    #: What a form and a message call each field.
    TITLES = {'rho_left': "Left density", 'p_left': "Left pressure",
              'rho_right': "Right density", 'p_right': "Right pressure",
              'gamma': "Gamma", 'xmin': "x min", 'xmax': "x max",
              'ncoord': "Points", 'time_increment': "Time step"}
    #: The largest Courant number the march stays stable at.
    MAX_COURANT = 1.0

    def replace(self, **values):
        """Return a copy with ``values`` in place, unchecked.

        :raise TypeError: For a name that is not in :attr:`FIELDS`.
        """
        return dataclasses.replace(self, **values)

    def check(self):
        """Return what is wrong, ``{field: message}`` in :attr:`FIELDS`
        order, or an empty dict for a valid setup.

        Each field is checked against its range, the ends of the tube
        against the diaphragm at x = 0, and the time step against the
        Courant limit.
        """
        errors = {}
        for name in self.FIELDS:
            message = self._check_field(name)
            if message:
                errors[name] = message
        # The solver and the exact solution both put the diaphragm at 0.
        for name, side, inside in (('xmin', "left", self.xmin < 0.0),
                                   ('xmax', "right", self.xmax > 0.0)):
            if name not in errors and not inside:
                errors[name] = (f"{self.TITLES[name]} must lie {side} of "
                                f"the diaphragm at x = 0; got "
                                f"{number_text(getattr(self, name))}.")
        # A limit read off a field out of range would mislead.
        if not errors.keys() & set(self.FIELDS):
            message = self._check_courant()
            if message:
                errors['time_increment'] = message
        return {name: errors[name] for name in self.FIELDS
                if name in errors}

    def _check_field(self, name):
        """Return what is wrong with one field on its own, or ''."""
        value = getattr(self, name)
        title = self.TITLES[name]
        lo, hi = self.RANGES[name]
        span = f"{number_text(lo)} to {number_text(hi)}"
        if isinstance(value, bool) or not isinstance(value, numbers.Real):
            return f"{title} must be a number in {span}; got {value!r}."
        if not math.isfinite(value):
            return f"{title} must be a number in {span}; got {value}."
        if name in self.INTEGERS and value != int(value):
            return f"{title} must be a whole number in {span}; got {value:g}."
        if not lo <= value <= hi:
            return f"{title} must lie in {span}; got {number_text(value)}."
        # The solver keeps every other point, so it takes an odd count.
        if 'ncoord' == name and 0 == int(value) % 2:
            return f"{title} must be odd; got {int(value)}."
        return ''

    def _check_courant(self):
        """Return why the time step outruns the grid, naming the longest
        one that does not, or ''."""
        courant = self.courant()
        if courant <= self.MAX_COURANT:
            return ''
        bound = self.time_increment * self.MAX_COURANT / courant
        # Floored to three digits, so the bound read off the message is
        # itself admissible.
        scale = 10.0 ** (2 - math.floor(math.log10(bound)))
        bound = math.floor(bound * scale) / scale
        return (f"Time step must stay under {number_text(bound)} on "
                f"{number_text(self.ncoord)} points, or the march blows up "
                f"(Courant number {courant:.2f}).")

    def spacing(self):
        """Return the distance between two solution points, every other
        point of the grid."""
        # The solver keeps every other grid point and takes its CFL number
        # over this distance.
        return 2.0 * (self.xmax - self.xmin) / (self.ncoord - 1)

    def courant(self):
        """Return the Courant number of the fastest signal of the exact
        solution: the time step over the time the signal takes to cross
        :meth:`spacing`."""
        return (self.time_increment * self.exact().max_speed()
                / self.spacing())

    def exact(self):
        """Return the :class:`~._riemann.ExactRiemann` of the two gases."""
        return ExactRiemann((self.rho_left, 0.0, self.p_left),
                            (self.rho_right, 0.0, self.p_right),
                            gamma=self.gamma)

    def is_sod(self):
        """Return whether the two gases pose Sod's problem."""
        sod = type(self)()
        return all(getattr(self, name) == getattr(sod, name)
                   for name in self.PROBLEM)

    def problem(self):
        """Return the name of the problem, ``Sod`` or ``rho 1 | 0.125, p 1 |
        0.1, gamma 1.4``."""
        if self.is_sod():
            return "Sod"
        text = number_text
        return (f"rho {text(self.rho_left)} | {text(self.rho_right)}, "
                f"p {text(self.p_left)} | {text(self.p_right)}, "
                f"gamma {text(self.gamma)}")

    def title(self):
        """Return the window title, ``Shock tube: Sod``."""
        return f"Shock tube: {self.problem()}"

    def summary(self):
        """Return the short name of the run, ``Sod, 201 points``."""
        return f"{self.problem()}, {number_text(self.ncoord)} points"

    def session(self, max_steps=ShockTubeSession.MAX_STEPS,
                steps_per_chunk=ShockTubeSession.STEPS_PER_CHUNK):
        """Build the :class:`~._session.ShockTubeSession` of the setup,
        ready to march from step 0.

        :param max_steps: The step the run ends at.
        :param steps_per_chunk: The steps one advance marches.
        :raise ValueError: Naming the first field :meth:`check` finds
            wrong, so no session is built of an invalid setup.
        """
        errors = self.check()
        if errors:
            raise ValueError(next(iter(errors.values())))
        return ShockTubeSession(
            rho_left=float(self.rho_left), p_left=float(self.p_left),
            rho_right=float(self.rho_right), p_right=float(self.p_right),
            gamma=float(self.gamma), xmin=float(self.xmin),
            xmax=float(self.xmax), ncoord=int(self.ncoord),
            time_increment=float(self.time_increment),
            steps_per_chunk=steps_per_chunk, max_steps=max_steps)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
