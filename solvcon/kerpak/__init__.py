# -*- coding: UTF-8 -*-
#
# Copyright (C) 2008-2011 Yung-Yu Chen <yyc@solvcon.net>.
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

"""
Solver packages (SPs), including numerical algorithms and physical models.

A SP inludes one or more solver kernels (SKs) and corresponding analysis
functions.  Codes in this subpackage should use absolute import to access
functionalities provided in SOLVCON, so that each SP can be migrated outside
the solvcon package namespace.

Available numerical algorithms include:
 - cese: second order, multi-dimensional CESE method.
 - cuse: second order, multi-dimensional, CUDA-enabled CESE method.
 - lincese: second order CESE method specialized for linear equations.
 - lincuse: second order CESE method specialized for linear equations with CUDA
   support.

Available physical models:
 - euler: the euler equations for gas dynamics.
 - elaslin: the velocity-stress equations for anisotropic, linear elastic
   solids.
 - vslin: the velocity-stress equations for anisotropic, linear elastic solids
   with CUDA support.
 - gasdyn: CUDA-enabled, gas-dynamics simulator.
"""

__all__ = ['cese', 'cuse', 'elaslin', 'euler', 'gasdyn', 'lincese', 'lincuse',
'vslin']
