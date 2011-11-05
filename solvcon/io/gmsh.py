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

"""
This is a loader for Gmsh format.  Currently only the ASCII format is
supported.

For more information about Gmsh ASCII file, please refer to 
http://www.geuz.org/gmsh/doc/texinfo/gmsh.html#MSH-ASCII-file-format
"""

from .core import FormatIO

class Gmsh(object):
    """
    Gmsh mesh object.  Indices nodes and elements in Gmsh is 1-based (Fortran
    convention) instead of 0-based (C convention) that is used throughout
    SOLVCON.  However, physics groups are using 0-based index.

    @cvar ELMAP: Element definition map.  The key is Gmsh element type ID.  The
        value is a 4-tuple: (i) dimension, (ii) number of total nodes, (iii) 
        SOLVCON cell type ID, and (iv) SOLVCON cell node ordering.
    @ctype ELMAP: dict
    @ivar stream: Input stream of the mesh data.
    @itype stream: file
    @ivar ndim: Number of dimension of this mesh.
    @itype ndim: int
    @ivar nodes: Three-dimensional coordinates of all nodes.  The shape is
        (number of Gmsh nodes, 3).  Note for even two-dimensional meshes the
        array still stores three-dimensional coordinates.
    @itype nodes: numpy.ndarray
    @ivar usnds: Indices (0-based) of the nodes really useful for SOLVCON.
    @itype usnds: numpy.ndarray
    @ivar ndmap: A mapping array from Gmsh node indices (0-based) to SOLVCON
        node indices (0-based).
    @itype ndmap: numpy.ndarray
    @ivar cltpn: SOLVCON cell type ID for each Gmsh element.
    @itype cltpn: numpy.ndarray
    @ivar elgrp: Group number of each Gmsh element.
    @itype elgrp: numpy.ndarray
    @ivar eldim: Dimension of each Gmsh element.
    @itype eldim: numpy.ndarray
    @ivar elems: Gmsh node indices (1-based) of each Gmsh element.
    @itype elems: numpy.ndarray
    @ivar intels: Indeces (0-based) of the elements inside the domain.
    @itype intels: numpy.ndarray
    @ivar physics: Physics groups as a list of 3-tuples: (i) dimension, (ii)
        index (0-based), and (iii) name.  If a physics group has the same 
        dimension as the mesh, it is an interior group.  Otherwise, the physics
        group must have one less dimension than the mesh, and it must be used
        as the boundary definition.
    @itype physics: list
    """

    ELMAP = {
        # ID: dimension, number of nodes, type id, node ordering.
        1: (1, 2, 1, [0, 1]),  # 2-node line.
        2: (2, 3, 3, [0, 1, 2]),   # 3-node triangle.
        3: (2, 4, 2, [0, 1, 2, 3]),    # 4-node quadrangle.
        4: (3, 4, 5, [0, 1, 2, 3]),    # 4-node tetrahedron.
        5: (3, 8, 4, [0, 1, 2, 3, 4, 5, 6, 7]),    # 8-node hexahedron.
        6: (3, 6, 6, [0, 2, 1, 3, 5, 4]),  # 6-node prism.
        7: (3, 5, 7, [0, 1, 2, 3, 4]), # 5-node pyramid.
        8: (1, 3, 1, [0, 1]),  # 3-node line.
        9: (2, 6, 3, [0, 1, 2]),   # 6-node triangle.
        10: (2, 9, 2, [0, 1, 2, 3]),   # 9-node quadrangle.
        11: (3, 10, 5, [0, 1, 2, 3]),  # 10-node tetrahedron.
        12: (3, 27, 4, [0, 1, 2, 3, 4, 5, 6, 7]),  # 27-node hexahedron.
        13: (3, 18, 6, [0, 2, 1, 3, 5, 4]),    # 18-node prism.
        14: (3, 14, 7, [0, 1, 2, 3, 4]),   # 14-node pyramid.
        15: (0, 1, 0, [0]),    # 1-node point.
        16: (2, 8, 2, [0, 1, 2, 3]),   # 8-node quadrangle.
        17: (3, 20, 4, [0, 1, 2, 3, 4, 5, 6, 7]),  # 20-node hexahedron.
        18: (3, 15, 6, [0, 2, 1, 3, 5, 4]),    # 15-node prism.
        19: (3, 13, 7, [0, 1, 2, 3, 4]),   # 13-node pyramid.
        20: (2, 9, 3, [0, 1, 2]),  # 9-node incomplete triangle.
        21: (2, 10, 3, [0, 1, 2]), # 10-node triangle.
        22: (2, 12, 3, [0, 1, 2]), # 12-node incomplete triangle.
        23: (2, 15, 3, [0, 1, 2]), # 15-node triangle.
        24: (2, 15, 3, [0, 1, 2]), # 15-node incomplete triangle.
        25: (2, 21, 3, [0, 1, 2]), # 21-node incomplete triangle.
        26: (1, 4, 1, [0, 1]), # 4-node edge.
        27: (1, 5, 1, [0, 1]), # 5-node edge.
        28: (1, 6, 1, [0, 1]), # 6-node edge.
        29: (3, 20, 5, [0, 1, 2, 3]),  # 20-node tetrahedron.
        30: (3, 35, 5, [0, 1, 2, 3]),  # 35-node tetrahedron.
        31: (3, 56, 5, [0, 1, 2, 3]),  # 56-node tetrahedron.
        92: (3, 64, 4, [0, 1, 2, 3, 4, 5, 6, 7]),  # 64-node hexahedron.
        93: (3, 125, 4, [0, 1, 2, 3, 4, 5, 6, 7]), # 125-node hexahedron.
    }

    def __init__(self, stream):
        self.stream = stream
        self.ndim = None
        self.nodes = None
        self.usnds = None
        self.ndmap = None
        self.cltpn = None
        self.elgrp = None
        self.eldim = None
        self.elems = None
        self.intels = None
        self.physics = list()

    @property
    def nnode(self):
        """
        Number of nodes that is useful for SOLVCON.
        """
        return self.usnds.shape[0]
    @property
    def ncell(self):
        """
        Number of cells that is useful for SOLVCON and interior.
        """
        return self.intels.shape[0]

    def load(self):
        """
        Load mesh data from storage.

        @return: nothing.
        """
        stream = self.stream
        loader_map = {
            '$MeshFormat': self._load_meta,
            '$Nodes': self._load_nodes,
            '$Elements': self._load_elements,
            '$PhysicalNames': self._load_physics,
        }
        while True:
            key = stream.readline().strip()
            if key:
                loader_map[key]()
            else:
                break
        self._parse_physics()
    def _load_meta(self):
        """
        Load and check the meta data of the mesh.

        @return: nothing.
        """
        stream = self.stream
        version_number, file_type, data_size = stream.readline().split()
        if stream.readline().strip() != '$EndMeshFormat':
            return False
        version_number = float(version_number)
        file_type = int(file_type)
        data_size = int(data_size)
        assert version_number > 2
        assert file_type == 0
        assert data_size == 8
    def _load_nodes(self):
        """
        Load node coordinates of the mesh data.  Because of the internal data
        structure of Python, Numpy, and SOLVCON, the loaded nodes are using 
        the 0-based index.

        @return: Successfully loaded or not.
        @rtype: bool
        """
        from numpy import empty
        stream = self.stream
        nnode = int(stream.readline().strip())
        self.nodes = empty((nnode, 3), dtype='float64')
        ind = 0
        while ind < nnode:
            dat = stream.readline().split()[1:]
            self.nodes[ind,:] = [float(ent) for ent in dat]
            ind += 1
        if stream.readline().strip() != '$EndNodes':
            return False
        else:
            return True
    def _load_elements(self):
        """
        Load element definition of the mesh data.  The node indices defined for
        each element are still 1-based.

        @return: Successfully loaded or not.
        @rtype: bool
        """
        from numpy import empty, array, arange, unique
        from ..block import Block
        stream = self.stream
        usnds = []
        nelem = int(stream.readline().strip())
        self.cltpn = empty(nelem, dtype='int32')
        self.elgrp = empty(nelem, dtype='int32')
        self.eldim = empty(nelem, dtype='int32')
        self.elems = empty((nelem, Block.CLMND+1), dtype='int32')
        self.elems.fill(-1)
        self.ndim = 0
        iel = 0
        while iel < nelem:
            dat = [int(ent) for ent in stream.readline().split()[1:]]
            tpn = dat[0]
            tag = dat[2:2+dat[1]]
            nds = dat[2+dat[1]:]
            elmap = self.ELMAP[tpn]
            self.cltpn[iel] = elmap[2]
            self.elgrp[iel] = tag[0]
            self.eldim[iel] = elmap[0]
            nnd = len(elmap[3])
            self.elems[iel,0] = nnd
            nds = array(nds, dtype='int32')[elmap[3]]
            usnds.extend(nds)
            self.elems[iel,1:nnd+1] = nds
            self.ndim = elmap[0] if elmap[0] > self.ndim else self.ndim
            iel += 1
        usnds = array(usnds, dtype='int32') - 1
        usnds.sort()
        usnds = unique(usnds)
        self.ndmap = empty(self.nodes.shape[0], dtype='int32')
        self.ndmap.fill(-1)
        self.ndmap[usnds] = arange(usnds.shape[0], dtype='int32')
        self.usnds = usnds
        if stream.readline().strip() != '$EndElements':
            return False
        else:
            return True
    def _load_physics(self):
        """
        Load physics groups of the mesh data.

        @return: Successfully loaded or not.
        @rtype: bool
        """
        from numpy import arange, concatenate, unique
        stream = self.stream
        while True:
            line = stream.readline().strip()
            if line == '$EndPhysicalNames':
                return True
            else:
                self.physics.append(line)
    def _parse_physics(self):
        """
        Parse physics groups of the mesh data.

        @return: Successfully loaded or not.
        @rtype: bool
        """
        from numpy import arange, concatenate, unique
        elidx = arange(self.elems.shape[0], dtype='int32')
        if not self.physics:
            self.intels = elidx[self.eldim == self.ndim]
            return False
        physics = []
        intels = []
        nphy = int(self.physics[0])
        iph = 0
        while iph < nphy:
            dat = self.physics[1+iph]
            dat = dat.split('"')[0].split() + [dat.split('"')[1]]
            dim = int(dat[0])
            phser = int(dat[1])
            name = dat[2].strip(' "\'')
            physics.append((name, dim, elidx[self.elgrp == phser]))
            if dim == self.ndim:
                intels.append(physics[-1][-1])
            elif dim < self.ndim:
                pass
            else:
                raise ValueError(
                    'Dimension of PhysicalName #%d:%s = %d < %d' % (
                        iph, name, dim, self.ndim))
            iph += 1
        self.physics = physics
        intels = concatenate(intels)
        intels.sort()
        self.intels = unique(intels)
        return True

    def toblock(self, onlybcnames=None, bcname_mapper=None, fpdtype=None,
            use_incenter=False):
        """
        Convert the loaded Gmsh data into a Block object.

        @keyword onlybcnames: positively list wanted names of BCs.
        @type onlybcnames: list
        @keyword bcname_mapper: map name to bc type number.
        @type bcname_mapper: dict
        @keyword fpdtype: floating-point dtype.
        @type fpdtype: str
        @keyword use_incenter: use incenter when creating block.
        @type use_incenter: bool

        @return: Block object.
        @rtype: solvcon.block.Block
        """
        from numpy import empty, arange
        from ..block import Block
        blk = Block(ndim=self.ndim, nnode=self.nnode, ncell=self.ncell,
            fpdtype=fpdtype, use_incenter=use_incenter)
        self._convert_interior(blk)
        blk.build_interior()
        self._convert_boundary(onlybcnames, bcname_mapper, fpdtype, blk)
        blk.build_boundary()
        blk.build_ghost()
        return blk
    def _convert_interior(self, blk):
        """
        Convert interior information from Gmsh to SOLVCON block.

        @param blk: The target SOLVCON block object.
        @type blk: solvcon.block.Block

        @return: nothing.
        """
        from numpy import empty, arange
        # basic information.
        blk.ndcrd[:] = self.nodes[self.usnds,:self.ndim]
        blk.cltpn[:] = self.cltpn[self.intels]
        blk.clnds[:] = self.elems[self.intels]
        for nds in blk.clnds:
            idx = nds[1:1+nds[0]]
            idx[:] = self.ndmap[idx-1]
        # groups.
        if self.physics:
            ecidx = empty(self.elems.shape[0], dtype='int32')
            ecidx.fill(-1)
            ecidx[self.intels] = arange(self.intels.shape[0], dtype='int32')
            iph = 0
            for name, dim, els in self.physics:
                if dim != self.ndim:
                    continue    # skip interior group assignment.
                blk.grpnames.append(name)
                cls = ecidx[els]
                blk.clgrp[cls] = len(blk.grpnames) - 1
                iph += 1
        else:
            blk.clgrp.fill(0)
            blk.grpnames.append('default')
    def _convert_boundary(self, onlybcnames, bcname_mapper, fpdtype, blk):
        """
        Convert boundary information from Gmsh to SOLVCON block.

        @param onlybcnames: positively list wanted names of BCs.
        @type onlybcnames: list
        @param bcname_mapper: map name to bc type number.
        @type bcname_mapper: dict
        @param fpdtype: floating-point dtype.
        @type fpdtype: str
        @param blk: The target SOLVCON block object.
        @type blk: solvcon.block.Block

        @return: nothing.
        """
        from numpy import arange, empty
        from ..boundcond import BC
        bfcs = arange(blk.nface, dtype='int32')[blk.fccls[:,1] < 0]
        nbfc = bfcs.shape[0]
        bfcndh = dict()
        for ifc in bfcs:
            for ind in blk.fcnds[ifc,1:blk.fcnds[ifc,0]+1]:
                lst = bfcndh.get(ind, list())
                lst.append(ifc)
                bfcndh[ind] = lst
        for name, dim, els in self.physics:
            if dim == self.ndim:
                continue
            if onlybcnames and name not in onlybcnames: # skip unwanted BCs.
                continue
            bndfcs = []
            for iel in els:
                elem = self.elems[iel]
                nnd = elem[0]
                # search for face.
                nds = self.ndmap[elem[1:1+nnd]-1]
                fset = set(bfcndh[nds[0]])
                for ind in nds[1:]:
                    fset &= set(bfcndh[ind])
                assert len(fset) == 1   # should find only 1 face.
                bndfcs.append(fset.pop())
            assert len(bndfcs) == len(els)  # must find everything.
            if not bndfcs:  # skip empty physics group.
                continue
            bcname_mapper = dict() if bcname_mapper is None else bcname_mapper
            bct, vdict = bcname_mapper.get(name, (BC, dict()))
            bc = bct(fpdtype=blk.fpdtype)
            bc.name = name
            bc.facn = empty((len(bndfcs), 3), dtype='int32')
            bc.facn.fill(-1)
            bc.facn[:,0] = bndfcs
            bc.feedValue(vdict)
            bc.sern = len(blk.bclist)
            bc.blk = blk
            blk.bclist.append(bc)

class GmshIO(FormatIO):
    """
    Proxy to Gmsh file format.
    """
    def load(self, stream, bcrej=None, bcmapper=None):
        """
        Load block from stream with BC mapper applied.

        @keyword stream: file object or file name to be read.
        @type stream: file or str
        @keyword bcrej: names of the BC to reject.
        @type bcrej: list
        @keyword bcmapper: map name to bc type number.
        @type bcmapper: dict
        @return: the loaded block.
        @rtype: solvcon.block.Block
        """
        import gzip
        # load Gmsh file.
        if isinstance(stream, basestring):
            if stream.endswith('.gz'):
                opener = gzip.open
            else:
                opener = open
            stream = opener(stream)
        gmh = Gmsh(stream)
        gmh.load()
        stream.close()
        # convert loaded Gmsh object into block object.
        if bcrej:
            onlybcnames = list()
            for bc in [it[2] for it in gmh.physics if it[0] == gmh.ndim-1]:
                if bc.name not in bcrej:
                    onlybcnames.append(bc.name)
        else:
            onlybcnames = None
        blk = gmh.toblock(onlybcnames=onlybcnames, bcname_mapper=bcmapper)
        return blk
