# -*- coding: UTF-8 -*-
#
# Copyright (c) 2014 Yung-Yu Chen <yyc@solvcon.net>
# BSD 3-Clause License, see COPYING


"""
Boundary-condition treatments for :py:class:`~.solver.GasPlusSolver`.
"""


import solvcon as sc
from solvcon import march


class GasPlusBC(sc.BC):
    """
    Base class for all boundary conditions of the gas solver.
    """

    #: Ghost geometry calculator type.
    _ghostgeom_ = "mirror"


class GasPlusNonRefl(GasPlusBC):
    pass


class GasPlusSlipWall(GasPlusBC):
    pass


class GasPlusInlet(GasPlusBC):
    vnames = ['rho', 'v1', 'v2', 'v3', 'p', 'gamma']
    vdefaults = {
        'rho': 1.0, 'p': 1.0, 'gamma': 1.4, 'v1': 0.0, 'v2': 0.0, 'v3': 0.0,
    }

# vim: set ff=unix fenc=utf8 ft=python nobomb et sw=4 ts=4 tw=79:
