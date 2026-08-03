# Copyright (c) 2026, solvcon team <contact@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Body-cell views of the multi-dimensional Euler solution.

Every solver and mesh table carries the ghost rows in front of the body
cells, so a raw :attr:`ndarray` slice needs the ghost offset that
:attr:`EulerCore.ngstcell` reports.  Reading the tables through these
helpers applies it once instead of at every call site.
"""

import numpy as np

__all__ = [
    'body_rows',
    'centroids',
    'conserved',
    'density',
    'total_mass',
    'volumes',
]


def body_rows(array, svr):
    """Return the body-cell rows of the ghost-padded ``array`` as a view.

    The view aliases the solver memory, so writing into it seeds the field.
    """
    ngst = svr.ngstcell
    return array.ndarray[ngst:ngst + svr.ncell]


def conserved(svr):
    """Return the newest order-0 solution, ``[ncell, neq]``.

    :meth:`EulerCore.march` leaves the newest step in ``so0n``.
    """
    return body_rows(svr.so0n, svr)


def density(svr):
    """Return the newest density over the body cells, ``[ncell]``."""
    return conserved(svr)[:, 0]


def centroids(mesh, svr):
    """Return the body-cell centroids, ``[ncell, ndim]``."""
    return body_rows(mesh.clcnd, svr)


def volumes(mesh, svr):
    """Return the body-cell volumes, ``[ncell]``."""
    return body_rows(mesh.clvol, svr)


def total_mass(mesh, svr):
    """Return the density integrated over the whole domain."""
    return float(np.sum(density(svr) * volumes(mesh, svr)))

# vim: set ff=unix fenc=utf8 et sw=4 ts=4 sts=4:
