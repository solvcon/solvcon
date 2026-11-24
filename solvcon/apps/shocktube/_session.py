# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Drive one shock-tube run.

A run is marched a chunk at a time, so a caller stays responsive between
chunks and keeps what each one measured.  :class:`ShockTubeSession` holds
both, so the GUI timer, a script loop, and a test drive the same run the
same way::

    sess = ShockTubeSession()
    while not sess.finished:
        sess.advance()
    sess.star_zones()

The run is judged against the exact solution of its Riemann problem by
:class:`~._analytic.TubeAnalysis`, and ends on the step cap or on request.
"""

import collections

import numpy as np

from ._analytic import NAMES, TubeAnalysis
from ._riemann import ExactRiemann
from ._shocktube import ShockTube

__all__ = [
    'RunRecord',
    'ShockTubeSession',
]


#: What one marched chunk leaves behind: the step it ended on, the time it
#: reached, the mass the tube held there, and the largest CFL number.
RunRecord = collections.namedtuple('RunRecord',
                                   ['step', 'time', 'mass', 'cfl'])


class ShockTubeSession(object):
    """Own one shock-tube run.

    Building the session builds the solver and the exact solution, so a
    session exists only around a run that is ready to march.  Each
    :meth:`advance` marches a chunk and records what it measured, and the
    density of the chunk for the x-t diagram; the run ends on the step cap
    or on :meth:`stop`, and :attr:`stop_reason` says which.

    The keyword arguments are the fields of
    :class:`~._setup.ShockTubeSetup`, which checks them, and the starting
    :attr:`steps_per_chunk` and :attr:`max_steps`.

    :ivar tube: The :class:`~._shocktube.ShockTube` being marched.
    :ivar analysis: The :class:`~._analytic.TubeAnalysis` of its points.
    :ivar records: The :class:`RunRecord` of each chunk, the first of step
        0, the newest :attr:`HISTORY` kept.
    :ivar density_rows: ``(time, density)`` of the newest chunks, the
        density at the points, as many as :attr:`ROW_BUDGET` values hold.
    :ivar steps_per_chunk: The steps one :meth:`advance` marches.
    :ivar max_steps: The step the run ends at.
    :ivar step: The steps marched so far.
    :ivar stop_reason: What ended the run, or None while it runs.
    """

    #: Default :attr:`steps_per_chunk`, one frame of the GUI timer.
    STEPS_PER_CHUNK = 1
    #: Default :attr:`max_steps`.
    MAX_STEPS = 50
    #: The records kept.
    HISTORY = 4096
    #: The values the rows of the x-t diagram may hold in all.
    ROW_BUDGET = 1 << 20

    def __init__(self, rho_left=1.0, p_left=1.0, rho_right=0.125,
                 p_right=0.1, gamma=1.4, xmin=-10.0, xmax=10.0, ncoord=201,
                 time_increment=0.05, steps_per_chunk=STEPS_PER_CHUNK,
                 max_steps=MAX_STEPS):
        self.tube = ShockTube()
        self.tube.build_constant(gamma=gamma, pressure1=p_left,
                                 density1=rho_left, pressure5=p_right,
                                 density5=rho_right)
        self.tube.build_numerical(xmin=xmin, xmax=xmax, ncoord=ncoord,
                                  time_increment=time_increment)
        exact = ExactRiemann((rho_left, 0.0, p_left),
                             (rho_right, 0.0, p_right), gamma=gamma)
        self.analysis = TubeAnalysis(exact, self.x, (xmin, xmax))
        self.time_increment = float(time_increment)
        self.records = collections.deque(maxlen=self.HISTORY)
        # A fine grid keeps fewer rows rather than more memory.
        nrow = max(2, min(self.HISTORY, self.ROW_BUDGET // len(self.x)))
        self.density_rows = collections.deque(maxlen=nrow)
        self.steps_per_chunk = steps_per_chunk
        self.max_steps = max_steps
        self.step = 0
        self.stop_reason = None
        #: The mass the tube holds at step 0.
        self.initial_mass = self._record(cfl=0.0).mass

    @property
    def exact(self):
        """The :class:`~._riemann.ExactRiemann` the run is judged by."""
        return self.analysis.exact

    @property
    def x(self):
        """The solution points."""
        svr = self.tube.svr
        return np.array(svr.coord, dtype='float64')[svr.xindices]

    @property
    def xlim(self):
        """The ends of the tube, ``(xmin, xmax)``."""
        return self.analysis.xlim

    @property
    def time(self):
        """The time marched, the steps times the time increment."""
        return self.step * self.time_increment

    @property
    def t_end(self):
        """The time at the step cap."""
        return self.max_steps * self.time_increment

    @property
    def finished(self):
        """Whether the run has ended."""
        return self.stop_reason is not None

    def values(self):
        """Return the run's quantities of :data:`~._analytic.NAMES` at the
        solution points, each a new array."""
        svr = self.tube.svr
        return {name: np.array(getattr(svr, name),
                               dtype='float64')[svr.xindices]
                for name in NAMES}

    def advance(self):
        """March the next chunk and record it.

        Returns the new :class:`RunRecord`, or None once the run has ended.
        The last chunk is trimmed so the march lands exactly on the step cap.
        """
        if self.finished:
            return None
        steps = min(self.steps_per_chunk, self.max_steps - self.step)
        self.tube.svr.march_alpha2(steps=steps)
        self.step += steps
        svr = self.tube.svr
        cfl = np.array(svr.cfl, dtype='float64')[svr.xindices]
        record = self._record(float(np.max(cfl)))
        if self.step >= self.max_steps:
            self.stop_reason = 'cap'
        return record

    def _record(self, cfl):
        density = self.values()['density']
        record = RunRecord(self.step, self.time,
                           self.analysis.mass(density), cfl)
        self.records.append(record)
        self.density_rows.append((self.time, density))
        return record

    def run(self):
        """March until the run ends; returns :attr:`stop_reason`."""
        while not self.finished:
            self.advance()
        return self.stop_reason

    def set_max_steps(self, max_steps):
        """Move the step cap: a run at or past the new cap ends there, and
        a run the old cap ended goes on under a higher one.

        :param max_steps: The new step cap.
        """
        self.max_steps = max_steps
        if self.step >= max_steps:
            if not self.finished:
                self.stop_reason = 'cap'
        elif 'cap' == self.stop_reason:
            self.stop_reason = None

    def stop(self):
        """End the run where it stands, keeping the field and the records."""
        if not self.finished:
            self.stop_reason = 'stopped'

    def mass_drift(self):
        """Return the mass the tube gained since step 0, over the mass it
        held then."""
        return (self.records[-1].mass - self.initial_mass) / self.initial_mass

    def star_zones(self):
        """See :meth:`~._analytic.TubeAnalysis.star_zones`."""
        return self.analysis.star_zones(self.values(), self.time)

    def front_fits(self):
        """See :meth:`~._analytic.TubeAnalysis.front_fits`."""
        return self.analysis.front_fits(self.values(), self.time)

    def l1_error(self, name='density'):
        """Return how far the run strays from the exact cell means of
        ``name``, by :meth:`~._analytic.TubeAnalysis.l1_error`.

        :param name: The quantity, one of :data:`~._analytic.NAMES`.
        """
        means = self.analysis.means(self.time)[name]
        return self.analysis.l1_error(self.values()[name], means)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
