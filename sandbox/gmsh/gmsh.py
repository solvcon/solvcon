# -*- coding: UTF-8 -*-
#
# Copyright (C) 2011 Yung-Yu Chen <yyc@solvcon.net>.
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

def main():
    import sys
    from solvcon.kerpak.cuse import CuseNonrefl
    from solvcon.io.gmsh import GmshIO
    from solvcon.io.vtkxml import VtkXmlUstGridWriter
    bct = CuseNonrefl
    bcname_mapper = dict(
        WallBoundary=(bct, {}),
        SymmetryBoundary=(bct, {}),
        InletBoundary=(bct, {}),
        ExitBoundary=(bct, {}),
    )
    gmhio = GmshIO()
    blk = gmhio.load(sys.argv[1])
    #blk = gmhio.load('../../tmp/bump_tri_level1.msh',
    #    bcname_mapper=bcname_mapper)
    print blk
    for bc in blk.bclist:
        print bc
    print blk.grpnames
    wtr = VtkXmlUstGridWriter(blk, appended=False, binary=True)
    outf = open(sys.argv[2], 'wb')
    wtr.write(outf)
    outf.close()

if __name__ == '__main__':
    main()
