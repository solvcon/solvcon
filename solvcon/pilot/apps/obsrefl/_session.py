# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Drive one oblique-shock reflection run.

A reflection run is one long march that has to be watched, so it is driven a
chunk at a time: a caller stays responsive between chunks and keeps what each
one measured.  :class:`ReflectionSession` holds both, so the GUI timer, a
script loop, and a test drive the same run the same way::

    sess = ReflectionSession(mach=3.0, angle=10.0)
    while not sess.finished:
        sess.advance()
    sess.zone_info()

The march is time-accurate, and the session judges it against the analytic
solution rather than against itself: what a run is worth is what
:mod:`._analytic` measures of its field, not how little the field moved
between two steps.  A run therefore ends on the step cap or on request, and
how long to march is the caller's to choose.
"""

import collections

from ._analytic import Reflection
from ._driver import ObliqueShock

__all__ = [
    'ReflectionSession',
    'RunHistory',
    'RunRecord',
]


#: What one marched chunk leaves behind: the step it ended on and the mass the
#: domain held there, which the inflow and the outflow move as the flow
#: develops.
RunRecord = collections.namedtuple('RunRecord', ['step', 'mass'])


class RunHistory(object):
    """Keep the newest ``length`` records of a run.

    A run marches for as long as the user lets it, so the history is bounded
    and drops its oldest record once it is full.  The bound is large enough
    that a run reaching the default step cap never loses one.
    """

    def __init__(self, length=2048):
        self.records = collections.deque(maxlen=length)

    def __len__(self):
        return len(self.records)

    def append(self, step, mass):
        """Record one chunk; returns the new :class:`RunRecord`."""
        record = RunRecord(step, mass)
        self.records.append(record)
        return record

    @property
    def last(self):
        """The newest record, or None before the first chunk."""
        return self.records[-1] if self.records else None

    @property
    def masses(self):
        """The ``(step, mass)`` pairs, oldest first."""
        return [(rec.step, rec.mass) for rec in self.records]


class ReflectionSession(object):
    """Own one oblique-shock reflection run.

    Building the session builds the flow constants, the mesh, and the solver,
    so a session exists only around a run that is ready to march.  Each
    :meth:`advance` marches a chunk and records what it measured; the run ends
    on the step cap or on :meth:`stop`, and :attr:`stop_reason` says which.

    :ivar shock: The :class:`~._driver.ObliqueShock` being marched.
    :ivar analysis: The :class:`~._analytic.Reflection` of its field.
    :ivar history: The :class:`RunHistory` of the chunks marched so far.
    :ivar steps_per_chunk: Full CESE steps one :meth:`advance` marches.
    :ivar max_steps: Steps after which the run ends.
    :ivar step: Steps marched so far.
    :ivar stop_reason: What ended the run, or None while it runs.
    """

    #: Default :attr:`steps_per_chunk`, one frame of the GUI timer.
    STEPS_PER_CHUNK = 5
    #: Default :attr:`max_steps`.
    MAX_STEPS = 2000

    def __init__(self, gamma=1.4, density=1.0, pressure=1.0, mach=3.0,
                 angle=10.0, cell_type='unstructured', time_increment=2.e-3,
                 nx=64, ny=16, steps_per_chunk=STEPS_PER_CHUNK,
                 max_steps=MAX_STEPS):
        self.shock = ObliqueShock()
        self.shock.build_constant(gamma=gamma, density=density,
                                  pressure=pressure, mach=mach, angle=angle)
        self.shock.build_numerical(cell_type=cell_type,
                                   time_increment=time_increment,
                                   nx=nx, ny=ny)
        self.analysis = Reflection(self.shock)
        self.history = RunHistory()
        self.steps_per_chunk = steps_per_chunk
        self.max_steps = max_steps
        self.step = 0
        self.stop_reason = None

    @property
    def field(self):
        """The body-cell reader of the running solution."""
        return self.analysis.field

    @property
    def finished(self):
        """Whether the run has ended."""
        return self.stop_reason is not None

    def advance(self):
        """March the next chunk and record it.

        Returns the new :class:`RunRecord`, or None once the run has ended.
        The last chunk is trimmed so the march lands exactly on the step cap.
        """
        if self.finished:
            return None
        steps = min(self.steps_per_chunk, self.max_steps - self.step)
        self.shock.march(steps)
        self.step += steps
        record = self.history.append(self.step,
                                     self.field.calc_overall_mass())
        if self.step >= self.max_steps:
            self.stop_reason = 'cap'
        return record

    def run(self):
        """March until the run ends; returns :attr:`stop_reason`."""
        while not self.finished:
            self.advance()
        return self.stop_reason

    def stop(self):
        """End the run where it stands, keeping the field and the history."""
        if not self.finished:
            self.stop_reason = 'stopped'

    def zone_info(self, name='density'):
        """See :meth:`Reflection.zone_info`."""
        return self.analysis.zone_info(name)

    def fit_incident_angle(self, nbin=None):
        """See :meth:`Reflection.fit_incident_angle`."""
        return self.analysis.fit_incident_angle(nbin)

    def reflection_point(self, nbin=None):
        """See :meth:`Reflection.reflection_point`."""
        return self.analysis.reflection_point(nbin)

    def profile(self, height, name='density', halfwidth=None):
        """See :meth:`Reflection.profile`."""
        return self.analysis.profile(height, name, halfwidth)

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
