# -*- coding: UTF-8 -*-
#
# Copyright (C) 2012 Yung-Yu Chen <yyc@solvcon.net>.
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
Epydoc tool for SCons.
"""

def build_epydoc(target, source, env):
    import sys
    sys.path.insert(0, '.')
    from solvcon.helper import generate_apidoc
    generate_apidoc()

def generate(env):
    from SCons.Builder import Builder
    env.Append(BUILDERS={
        'BuildEpydoc': Builder(
            action=build_epydoc,
    )})

def exists(env):
    return env.Detect('sphinx')
