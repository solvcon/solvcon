# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
The exact solution of the Riemann problem for the Euler equations of an
ideal gas.

Every quantity is dimensionless in this module.
"""

import math
import typing

import numpy as np

__all__ = [
    'GasState',
    'Wave',
    'ExactRiemann',
    'QUANTITIES',
    'derive',
]

#: The spacing of float64 at 1.
EPSILON = float(np.finfo('float64').eps)


class GasState(typing.NamedTuple):
    """A uniform ideal-gas state in the primitive variables."""

    rho: float
    u: float
    p: float


class Wave(typing.NamedTuple):
    """One wave of the solution of a Riemann problem.

    :ivar kind: ``'shock'``, ``'rarefaction'``, or ``'contact'``, and
        ``'vacuum'`` for the region between the two fans when the data
        generate vacuum.
    :ivar side: ``'left'``, ``'middle'``, or ``'right'``.
    :ivar speeds: ``(speed,)`` for a shock or a contact, ``(head, tail)``
        for a fan, and ``(left front, right front)`` for vacuum.
    """

    kind: str
    side: str
    speeds: tuple


class ExactRiemann(object):
    """Solve the Riemann problem of the ideal-gas Euler equations exactly
    on construction, and sample its solution.

    The diaphragm at ``x0`` separates the ``left`` and ``right`` states at t
    = 0.  The solution is self-similar in (x - x0) / t: the left wave, the
    contact, and the right wave bound the star region, where the pressure
    ``p_star`` and the velocity ``u_star`` are uniform and the density jumps
    from ``rho_star_left`` to ``rho_star_right`` across the contact.

    The left and right waves each connect an initial state to the star
    region by a curve of velocity against pressure: the Rankine-Hugoniot
    conditions, written with the mass flux through the shock, above the
    initial pressure, and the Riemann invariant of the isentropic fan below
    it.  The star pressure is where the two curves meet, and the
    self-similar solution follows in (x - x0) / t.  The theory is in chapter
    4 of E. F. Toro, Riemann Solvers and Numerical Methods for Fluid
    Dynamics, 3rd ed., Springer, 2009.  It takes any two states, so either
    wave may be a shock or a rarefaction, and it reports vacuum instead of
    failing on it.

    :ivar vacuum: True when the data generate vacuum, that is when the two
        fans expanding to zero pressure still cannot close the gap the
        streams open: 2 (c_L + c_R) / (gamma - 1) <= u_R - u_L.  The middle
        of the solution is then a vacuum between two fans: ``p_star`` and
        the star densities are 0 and ``u_star`` is NaN.
    :ivar waves: ``[left wave, contact, right wave]`` as :class:`Wave`, with
        the contact replaced by the vacuum region when :attr:`vacuum` holds.
    :ivar strengths: The relative jump across each of :attr:`waves`, in
        pressure for the outer waves and in density for the contact; 0 marks
        a wave that is not there, such as the fans of an isolated contact.
    """

    #: A relative jump at or below this is round-off, not a wave.
    TRIVIAL_JUMP = 1.e-12

    #: Newton's method stops once a step moves the pressure by no more than
    #: this fraction of it.
    TOLERANCE = 1.e-12
    MAX_ITERATION = 100

    def __init__(self, left, right, gamma=1.4, x0=0.0):
        self.gamma = float(gamma)
        self.left = GasState(*map(float, left))
        self.right = GasState(*map(float, right))
        self.x0 = float(x0)
        self._check_input()

        gamma = self.gamma
        self.c_left = math.sqrt(gamma * self.left.p / self.left.rho)
        self.c_right = math.sqrt(gamma * self.right.p / self.right.rho)
        # The velocity gap two fans can close by expanding to zero pressure,
        # less the gap the streams open.
        room = (self.c_left + self.c_right
                - 0.5 * (gamma - 1.0) * (self.right.u - self.left.u))
        self.vacuum = room <= 0.0

        if self.vacuum:
            self.p_star = 0.0
            self.u_star = math.nan
            self.rho_star_left = self.rho_star_right = 0.0
            self._c_star = (0.0, 0.0)
        elif not self._solve_two_fans(room):
            self._solve_star()

        self.waves = [self._outer_wave(self.left, self.c_left, -1.0),
                      self._middle_wave(),
                      self._outer_wave(self.right, self.c_right, 1.0)]
        if self.vacuum:
            jumps = (1.0, 1.0, 1.0)
        else:
            dense = max(self.rho_star_left, self.rho_star_right)
            contact = abs(self.rho_star_left - self.rho_star_right)
            jumps = (abs(self.p_star - self.left.p) / self.left.p,
                     contact / dense if dense > 0.0 else 0.0,
                     abs(self.p_star - self.right.p) / self.right.p)
        self.strengths = tuple(jump if jump > self.TRIVIAL_JUMP else 0.0
                               for jump in jumps)

    def _check_input(self):
        if not (math.isfinite(self.gamma) and self.gamma > 1.0):
            raise ValueError(f"gamma must be greater than 1, got {self.gamma}")
        if not math.isfinite(self.x0):
            raise ValueError(f"x0 must be finite, got {self.x0}")
        for side, state in (('left', self.left), ('right', self.right)):
            if not all(math.isfinite(value) for value in state):
                raise ValueError(f"the {side} state must be finite, got "
                                 f"{tuple(state)}")
            if state.rho <= 0.0 or state.p <= 0.0:
                raise ValueError(f"the {side} density and pressure must be "
                                 f"positive, got {tuple(state)}")

    @property
    def pattern(self):
        """The wave pattern as letters, left to right: ``'R-C-S'`` for a
        left rarefaction, the contact, and a right shock, ``'R-R'`` when the
        contact carries no jump, and ``'V'`` for the vacuum."""
        letters = {'shock': 'S', 'rarefaction': 'R', 'contact': 'C',
                   'vacuum': 'V'}
        return '-'.join(letters[wave.kind] for wave, strength
                        in zip(self.waves, self.strengths) if strength)

    def _mass_flux(self, pressure, state):
        """Return the mass flux through a shock that raises the pressure of
        ``state`` to ``pressure``."""
        gamma = self.gamma
        return math.sqrt(0.5 * state.rho * ((gamma + 1.0) * pressure
                                            + (gamma - 1.0) * state.p))

    def _velocity_loss(self, pressure, state, sound):
        """Return how much a wave that brings ``state`` to ``pressure``
        slows the gas toward the other side: u_L minus the star velocity on
        the left, the star velocity minus u_R on the right.

        A shock compresses the gas and slows it by the pressure jump over
        the mass flux; a fan expands it along the Riemann invariant u +- 2 c
        / (gamma - 1), and speeds it up instead.
        """
        if pressure > state.p:
            return (pressure - state.p) / self._mass_flux(pressure, state)
        gamma = self.gamma
        ratio = (pressure / state.p) ** ((gamma - 1.0) / (2.0 * gamma))
        return 2.0 / (gamma - 1.0) * sound * (ratio - 1.0)

    def _velocity_loss_slope(self, pressure, state, sound):
        """Return the derivative of :meth:`_velocity_loss` in pressure; it
        grows without bound as the pressure of a fan goes to 0."""
        gamma = self.gamma
        if pressure > state.p:
            flux = self._mass_flux(pressure, state)
            return (1.0 - 0.25 * (gamma + 1.0) * state.rho
                    * (pressure - state.p) / flux ** 2) / flux
        ratio = pressure / state.p
        return (ratio ** (-0.5 * (gamma + 1.0) / gamma)
                / (state.rho * sound))

    def _solve_two_fans(self, room):
        """Set the star state and return True when both waves are fans.

        Along both fans c / c_K = (p / p_K)^((gamma - 1) / (2 gamma)), so
        equating the Riemann invariants from the two sides is linear in z =
        p*^((gamma - 1) / (2 gamma)): z sum_K c_K / p_K^((gamma - 1) / (2
        gamma)) = ``room``.  The ratios c*_K / c_K follow from z and stay
        representable next to vacuum, where p* itself underflows.  The fans
        hold when z leaves p* at or below both initial pressures.
        """
        gamma = self.gamma
        left, right = self.left, self.right
        power = (gamma - 1.0) / (2.0 * gamma)
        level = room / (self.c_left / left.p ** power
                        + self.c_right / right.p ** power)
        if level > min(left.p, right.p) ** power:
            return False
        ratio_left = level / left.p ** power
        ratio_right = level / right.p ** power
        self.p_star = level ** (1.0 / power)
        span = 2.0 / (gamma - 1.0)
        self.u_star = 0.5 * ((left.u + span * self.c_left * (1.0 - ratio_left))
                             + (right.u - span * self.c_right
                                * (1.0 - ratio_right)))
        self.rho_star_left = left.rho * ratio_left ** span
        self.rho_star_right = right.rho * ratio_right ** span
        self._c_star = (self.c_left * ratio_left, self.c_right * ratio_right)
        return True

    def _solve_star(self):
        """Set the star state when at least one wave is a shock, which puts
        p* above the smaller initial pressure."""
        gamma = self.gamma
        left, right = self.left, self.right
        self.p_star = self._find_pressure()
        loss_left = self._velocity_loss(self.p_star, left, self.c_left)
        loss_right = self._velocity_loss(self.p_star, right, self.c_right)
        self.u_star = 0.5 * ((left.u - loss_left) + (right.u + loss_right))
        self.rho_star_left = self._star_density(left)
        self.rho_star_right = self._star_density(right)
        power = (gamma - 1.0) / (2.0 * gamma)
        self._c_star = tuple(sound * (self.p_star / state.p) ** power
                             for state, sound in ((left, self.c_left),
                                                  (right, self.c_right)))

    def _acoustic_pressure(self):
        """Return the star pressure of the linearized problem, where each
        wave is a sound wave across which dp = -+ rho c du."""
        left, right = self.left, self.right
        imp_left = left.rho * self.c_left
        imp_right = right.rho * self.c_right
        return ((imp_right * left.p + imp_left * right.p
                 - imp_left * imp_right * (right.u - left.u))
                / (imp_left + imp_right))

    def _find_pressure(self):
        """Return p*, where the left and the right waves leave the gas the
        same velocity, when it lies above the smaller initial pressure.

        The mismatch of the two velocities rises with the pressure and is
        concave in it, so a Newton step from below the root never passes
        it.  A step from above may land below the smaller initial pressure,
        where the mismatch is known to be negative; the iterate is held
        there, and climbs to the root from below.
        """
        left, right = self.left, self.right
        floor = min(left.p, right.p)
        gap = right.u - left.u
        pressure = max(self._acoustic_pressure(), floor)
        for _ in range(self.MAX_ITERATION):
            loss_left = self._velocity_loss(pressure, left, self.c_left)
            loss_right = self._velocity_loss(pressure, right, self.c_right)
            mismatch = loss_left + loss_right + gap
            # A mismatch at the round-off of its terms cannot shrink further.
            noise = abs(loss_left) + abs(loss_right) + abs(gap)
            if abs(mismatch) <= 4.0 * EPSILON * noise:
                return pressure
            slope = (self._velocity_loss_slope(pressure, left, self.c_left)
                     + self._velocity_loss_slope(pressure, right,
                                                 self.c_right))
            update = max(pressure - mismatch / slope, floor)
            if abs(update - pressure) <= self.TOLERANCE * update:
                return update
            pressure = update
        raise RuntimeError(f"the star pressure did not converge in "
                           f"{self.MAX_ITERATION} iterations")

    def _star_density(self, state):
        """Return the star density on the side of ``state``.

        Behind a shock the mass flux m fixes the jump in specific volume,
        1 / rho_K - 1 / rho* = (p* - p_K) / m^2; behind a fan the gas keeps
        its entropy, so rho* = rho_K (p* / p_K)^(1 / gamma).
        """
        if self.p_star > state.p:
            flux = self._mass_flux(self.p_star, state)
            jump = (self.p_star - state.p) / flux ** 2
            return 1.0 / (1.0 / state.rho - jump)
        return state.rho * (self.p_star / state.p) ** (1.0 / self.gamma)

    def _outer_wave(self, state, sound, sign):
        """Return the wave on the side of ``state``; ``sign`` is -1 on the
        left and +1 on the right, the direction the wave faces."""
        side = 'left' if sign < 0.0 else 'right'
        head = state.u + sign * sound
        if self.vacuum:
            return Wave('rarefaction', side, (head, self._vacuum_front(sign)))
        if self.p_star > state.p:
            # The gas crosses the shock at the mass flux m, so the shock
            # runs at m / rho_K relative to the gas ahead of it.
            flux = self._mass_flux(self.p_star, state)
            return Wave('shock', side, (state.u + sign * flux / state.rho,))
        c_star = self._c_star[0 if sign < 0.0 else 1]
        return Wave('rarefaction', side, (head, self.u_star + sign * c_star))

    def _middle_wave(self):
        if self.vacuum:
            return Wave('vacuum', 'middle', (self._vacuum_front(-1.0),
                                             self._vacuum_front(1.0)))
        return Wave('contact', 'middle', (self.u_star,))

    def _vacuum_front(self, sign):
        """Return the speed of the vacuum front, the tail of the fan on the
        side ``sign`` (-1 left, +1 right), where the gas meets vacuum."""
        state, sound = ((self.left, self.c_left) if sign < 0.0
                        else (self.right, self.c_right))
        return state.u - sign * 2.0 / (self.gamma - 1.0) * sound

    def sample(self, x, t):
        """Return the solution at positions ``x`` and time ``t``.

        Inside a vacuum rho and p are 0, and u = (x - x0) / t continues the
        velocity of the two fans, whose tails move with the gas: a
        convention for drawing, as no mass carries that velocity.

        :param x: Positions, array-like of any shape.
        :param t: Time; ``t <= 0`` returns the initial data, the left state
            where ``x < x0`` and the right state elsewhere.
        :return: ``{'rho', 'u', 'p'}`` float64 arrays shaped like ``x``.
        """
        xs = np.asarray(x, dtype='float64')
        rho = np.empty(xs.shape, dtype='float64')
        vel = np.empty(xs.shape, dtype='float64')
        pre = np.empty(xs.shape, dtype='float64')
        out = (rho, vel, pre)

        if t <= 0.0:
            on_left = xs < self.x0
            _fill(out, on_left, self.left)
            _fill(out, ~on_left, self.right)
            return {'rho': rho, 'u': vel, 'p': pre}

        speed = (xs - self.x0) / t
        if self.vacuum:
            low, high = self.waves[1].speeds
            middle = (speed >= low) & (speed <= high)
            _fill(out, middle, (0.0, 0.0, 0.0))
            vel[middle] = speed[middle]
            sides = (speed < low, speed > high)
        else:
            on_left = speed <= self.u_star
            sides = (on_left, ~on_left)

        for mask, sign in zip(sides, (-1.0, 1.0)):
            self._sample_side(out, speed, mask, sign)
        return {'rho': rho, 'u': vel, 'p': pre}

    def _sample_side(self, out, speed, mask, sign):
        """Fill ``out`` where ``mask`` holds, on the side ``sign`` of the
        contact (-1 left, +1 right).  The gas the outer front has not
        reached keeps its initial state, and a sample on that front counts
        as not reached.
        """
        if sign < 0.0:
            wave, state, sound = self.waves[0], self.left, self.c_left
            rho_star = self.rho_star_left
        else:
            wave, state, sound = self.waves[2], self.right, self.c_right
            rho_star = self.rho_star_right
        untouched = mask & (sign * (speed - wave.speeds[0]) >= 0.0)
        if 'shock' == wave.kind:
            inside = mask & ~untouched
            _fill(out, inside, (rho_star, self.u_star, self.p_star))
        else:
            past_tail = mask & (sign * (speed - wave.speeds[1]) < 0.0)
            in_fan = mask & ~untouched & ~past_tail
            _fill(out, past_tail, (rho_star, self.u_star, self.p_star))
            if in_fan.any():
                _fill(out, in_fan,
                      self._fan(speed[in_fan], state, sound, sign))
        _fill(out, untouched, state)

    def _fan(self, speed, state, sound, sign):
        """Return (rho, u, p) inside the fan on the side ``sign`` at the
        given values of (x - x0) / t.

        Each ray of the fan is a characteristic, so (x - x0) / t = u + sign
        c on it; the gas also keeps the Riemann invariant of the other
        family, u - sign 2 c / (gamma - 1), and its entropy, from the outer
        state.  The two relations give c, then u, then rho and p.
        """
        gamma = self.gamma
        invariant = state.u - sign * 2.0 / (gamma - 1.0) * sound
        # Round-off may leave the sound speed a hair below 0 at a vacuum
        # front.
        snd = np.maximum(sign * (gamma - 1.0) / (gamma + 1.0)
                         * (speed - invariant), 0.0)
        vel = speed - sign * snd
        ratio = snd / sound
        return (state.rho * ratio ** (2.0 / (gamma - 1.0)), vel,
                state.p * ratio ** (2.0 * gamma / (gamma - 1.0)))

    def positions(self, t):
        """Return the wave fronts at time ``t`` for annotation.

        :return: ``[(label, x)]`` with the label ``'head'`` or ``'tail'`` for
            the edges of a fan, ``'contact'``, or ``'shock'``, ordered from
            left to right.  With vacuum the two fan tails bound the vacuum.
            A wave without a jump (see :attr:`strengths`) has no front.
        """
        elapsed = max(float(t), 0.0)
        fronts = []
        for wave, strength in zip(self.waves, self.strengths):
            if wave.kind == 'vacuum' or not strength:
                continue
            if wave.kind == 'rarefaction':
                pairs = list(zip(('head', 'tail'), wave.speeds))
                if wave.side == 'right':
                    pairs.reverse()
            else:
                pairs = [(wave.kind, wave.speeds[0])]
            fronts.extend((label, self.x0 + speed * elapsed)
                          for label, speed in pairs)
        return fronts


def _fill(out, mask, values):
    for array, value in zip(out, values):
        array[mask] = value


#: The quantities :func:`derive` computes, in display order.
QUANTITIES = ('rho', 'u', 'p', 'e', 's', 'mach')


def derive(name, rho, u, p, gamma):
    """Return the quantity ``name`` of :data:`QUANTITIES` from the primitive
    variables, element by element.

    ``e`` is the specific internal energy p / ((gamma - 1) rho), ``s`` the
    entropy function p / rho**gamma, and ``mach`` the Mach number |u| / c with
    the sound speed c = sqrt(gamma p / rho).  Vacuum (rho = 0) and an
    unphysical state give NaN or infinity instead of an error.

    :return: A new float64 array.
    """
    rho = np.asarray(rho, dtype='float64')
    vel = np.asarray(u, dtype='float64')
    pre = np.asarray(p, dtype='float64')
    with np.errstate(divide='ignore', invalid='ignore'):
        if 'rho' == name:
            return rho.copy()
        if 'u' == name:
            return vel.copy()
        if 'p' == name:
            return pre.copy()
        if 'e' == name:
            return pre / ((gamma - 1.0) * rho)
        if 's' == name:
            return pre / rho ** gamma
        if 'mach' == name:
            return np.abs(vel) / np.sqrt(gamma * pre / rho)
    raise ValueError(f"unknown quantity '{name}'; choose from {QUANTITIES}")

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
