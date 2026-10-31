# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""The arithmetic of the shock-tube plots.
"""

# Code in this file should not use Qt.

import math

__all__ = [
    'Axis',
    'place',
    'region_spans',
]


class Axis:
    """One axis mapping, linear or log10, from data onto pixels.

    :ivar lo: The value, or its decade on a log axis, at pixel ``first``.
    :ivar hi: The value, or its decade on a log axis, at pixel ``last``.
    """

    def __init__(self, lo, hi, first, last, log=False):
        self.lo, self.hi, self.log = lo, hi, log
        self.first, self.last = first, last

    def __call__(self, value):
        if self.log:
            value = math.log10(value) if value > 0 else -math.inf
        span = self.hi - self.lo
        return self.first + (value - self.lo) / span * (self.last - self.first)


def place(labels, low, high):
    """Return where the labels of a band go, each in the first of its rows
    that has room for it.

    :param labels: ``(center, width, rows)`` per label, in pixels and in
        the order the labels claim their room; ``rows`` are the rows the
        label may take, tried in turn.  A label crossing ``low`` or
        ``high``, or overlapping one placed before it in every row it may
        take, is left out.
    :return: ``[(index, row, left)]`` of the labels placed, with the left
        pixel of each.
    """
    taken = {}
    out = []
    for index, (center, width, rows) in enumerate(labels):
        left = center - width / 2
        right = left + width
        if left < low or right > high:
            continue
        for row in rows:
            boxes = taken.setdefault(row, [])
            if not any(left < other_right and other_left < right
                       for other_left, other_right in boxes):
                boxes.append((left, right))
                out.append((index, row, left))
                break
    return out


def region_spans(fronts, xmin, xmax, vacuum=False):
    """Name the regions between the wave fronts.

    :param fronts: ``(label, x)`` pairs, left first.
    :param vacuum: Whether the region between the two fans is a vacuum.
    :return: ``(name, left, right)`` per region: L, a fan, L*, R*, R, and
        a star region without a contact, ``*``, or the vacuum, ``V``.  Two
        equal states make no wave, and the tube is one region.
    """
    if not fronts:
        return [('L = R', xmin, xmax)]
    labels = [label for label, _ in fronts]
    edges = [xmin] + [pos for _, pos in fronts] + [xmax]
    names = ['L']
    for it in range(len(labels) - 1):
        pair = (labels[it], labels[it + 1])
        if pair in (('head', 'tail'), ('tail', 'head')):
            names.append('fan')
        elif 'contact' == labels[it + 1]:
            names.append('L*')
        elif 'contact' == labels[it]:
            names.append('R*')
        else:
            names.append('V' if vacuum else '*')
    names.append('R')
    return list(zip(names, edges[:-1], edges[1:]))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4 tw=79:
