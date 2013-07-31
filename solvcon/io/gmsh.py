# -*- coding: UTF-8 -*-
#
# Copyright (c) 2011, Yung-Yu Chen <yyc@solvcon.net>
#
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# - Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
# - Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# - Neither the name of the SOLVCON nor the names of its contributors may be
#   used to endorse or promote products derived from this software without
#   specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

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
    convention), but 0-based (C convention) indices are used throughout
    SOLVCON.  However, physics groups are using 0-based index.

    """

    #: Element definition map.  The key is Gmsh element type ID.  The
    #: value is a 4-tuple: (i) dimension, (ii) number of total nodes, (iii) 
    #: SOLVCON cell type ID, and (iv) SOLVCON cell node ordering.
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

    def __init__(self, stream, load=False):
        """
        >>> # sample data.
        >>> import StringIO
        >>> data = \"\"\"$MeshFormat
        ... 2.2 0 8
        ... $EndMeshFormat
        ... $Nodes
        ... 3
        ... 1 -1 0 0
        ... 2 1 0 0
        ... 3 0 1 0
        ... $EndNodes
        ... $Elements
        ... 1
        ... 1 2 2 1 22 1 2 3
        ... $EndElements
        ... $PhysicalNames
        ... 1
        ... 2 1 "lower"
        ... $EndPhysicalNames
        ... $Periodic
        ... 1
        ... 0 1 3
        ... 1
        ... 1 3
        ... $EndPeriodic\"\"\"

        Creation of the object doesn't load data:

        >>> gmsh = Gmsh(StringIO.StringIO(data))
        >>> None is gmsh.ndim
        True
        >>> gmsh.load()
        >>> gmsh.ndim
        2
        >>> gmsh.stream.close() # it's a good habit :-)

        We can request to load data on creation by setting *load=True*.  Note
        the stream will be closed after creation+loading.  The default behavior
        is different to :py:meth:`load`.

        >>> gmsh = Gmsh(StringIO.StringIO(data), load=True)
        >>> gmsh.ndim
        2
        >>> gmsh.stream.closed
        True
        """
        #: Input stream (:py:class:`file`) of the mesh data.
        self.stream = stream
        #: Number of dimension of this mesh (py:class:`int`).  Stored by
        #: :py:meth:`_load_elements`.
        self.ndim = None
        #: Three-dimensional coordinates of all nodes
        # (:py:class:`numpy.ndarray`).  The shape is (number
        #: of Gmsh nodes, 3).  Note for even two-dimensional meshes the
        #: array still stores three-dimensional coordinates.  Stored by
        #: :py:meth:`_load_nodes`.
        self.nodes = None
        #: Indices (0-based) of the nodes really useful for SOLVCON
        #: (:py:class:`numpy.ndarray`).  Stored by :py:meth:`_load_elements`.
        self.usnds = None
        #: A mapping array from Gmsh node indices (0-based) to SOLVCON node
        #: indices (0-based) (:py:class:`numpy.ndarray`).  Stored by
        #: :py:meth:`_load_elements`.
        self.ndmap = None
        #: SOLVCON cell type ID for each Gmsh element
        #: (py:class:`numpy.ndarray`).  Stored by :py:meth:`_load_elements`.
        self.cltpn = None
        #: Physics group number of each Gmsh element; the first tag
        #: (:py:class:`numpy.ndarray`).  Stored by :py:meth:`_load_elements`.
        self.elgrp = None
        #: Geometrical gropu number of each Gmsh element; the second tag
        #: (:py:class:`numpy.ndarray`).  Stored by :py:meth:`_load_elements`.
        self.elgeo = None
        #: Dimension of each Gmsh element (:py:class:`numpy.ndarray`).  Stored
        #: by :py:meth:`_load_elements`.
        self.eldim = None
        #: Gmsh node indices (1-based) of each Gmsh element
        #: (:py:class:`numpy.ndarray`).  Stored by :py:meth:`_load_elements`.
        self.elems = None
        #: Indices (0-based) of the elements inside the domain
        #: (:py:class:`numpy.ndarray`).  Stored by :py:meth:`_parse_physics`.
        self.intels = None
        #: Physics groups as a :py:class:`list` of 3-tuples: (i) dimension,
        #: (ii) index (0-based), and (iii) name.  If a physics group has the
        #: same dimension as the mesh, it is an interior group.  Otherwise, the
        #: physics group must have one less dimension than the mesh, and it
        #: must be used as the boundary definition.  Stored by
        #: :py:meth:`_load_physics` and then processed by
        #: :py:meth:`_parse_physics`.
        self.physics = list()
        #: Periodic relation :py:class:`list`.  Each item is a
        #: :py:class:`dict`:.  Stored by :py:meth:`_load_periodic`.
        self.periodics = list()

        # load file if requested.
        if load:
            self.load(close=True)

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

    ############################################################################
    # Loading methods.
    def load(self, close=False):
        """
        Load mesh data from storage.

        >>> # sample data.
        >>> import StringIO
        >>> data = \"\"\"$MeshFormat
        ... 2.2 0 8
        ... $EndMeshFormat
        ... $Nodes
        ... 3
        ... 1 -1 0 0
        ... 2 1 0 0
        ... 3 0 1 0
        ... $EndNodes
        ... $Elements
        ... 1
        ... 1 2 2 1 22 1 2 3
        ... $EndElements
        ... $PhysicalNames
        ... 1
        ... 2 1 "lower"
        ... $EndPhysicalNames
        ... $Periodic
        ... 1
        ... 0 1 3
        ... 1
        ... 1 3
        ... $EndPeriodic\"\"\"

        Load the mesh data after creation of the object.  Note the stream is
        left opened after loading.

        >>> stream = StringIO.StringIO(data)
        >>> gmsh = Gmsh(stream)
        >>> gmsh.load()
        >>> stream.closed
        False
        >>> stream.close() # it's a good habit :-)

        We can ask :py:meth:`load` to close the stream after loading by using
        *close=True*:

        >>> gmsh = Gmsh(StringIO.StringIO(data))
        >>> gmsh.load(close=True)
        >>> gmsh.stream.closed
        True
        """
        loader_map = {
            '$MeshFormat': lambda: Gmsh._check_meta(self.stream),
            '$Nodes': lambda: Gmsh._load_nodes(self.stream),
            '$Elements': lambda: Gmsh._load_elements(self.stream, self.nodes),
            '$PhysicalNames': lambda: Gmsh._load_physics(self.stream),
            '$Periodic': lambda: Gmsh._load_periodic(self.stream),
        }
        while True:
            key = self.stream.readline().strip()
            if key:
                self.__dict__.update(loader_map[key]())
            else:
                break
        self._parse_physics()
        if close:
            self.stream.close()

    @staticmethod
    def _check_meta(stream):
        """
        Load and check the meta data of the mesh.  It doesn't return anything
        to be stored.
        
        >>> import StringIO
        >>> stream = StringIO.StringIO(\"\"\"$MeshFormat
        ... 2.2 0 8
        ... $EndMeshFormat\"\"\")
        >>> stream.readline()
        '$MeshFormat\\n'
        >>> Gmsh._check_meta(stream)
        {}
        >>> stream.readline()
        ''
        """
        version_number, file_type, data_size = stream.readline().split()
        if stream.readline().strip() != '$EndMeshFormat':
            return False
        version_number = float(version_number)
        file_type = int(file_type)
        data_size = int(data_size)
        assert version_number > 2
        assert file_type == 0
        assert data_size == 8
        return dict()

    @staticmethod
    def _load_nodes(stream):
        """
        Load node coordinates of the mesh data.  Because of the internal data
        structure of Python, Numpy, and SOLVCON, the loaded :py:attr:`nodes`
        are using the 0-based index.

        >>> import StringIO
        >>> stream = StringIO.StringIO(\"\"\"$Nodes
        ... 3
        ... 1 -1 0 0
        ... 2 1 0 0
        ... 3 0 1 0
        ... $EndNodes\"\"\") # a triangle.
        >>> stream.readline()
        '$Nodes\\n'
        >>> Gmsh._load_nodes(stream) # doctest: +NORMALIZE_WHITESPACE
        {'nodes': array([[-1.,  0.,  0.], [ 1.,  0.,  0.], [ 0.,  1.,  0.]])}
        >>> stream.readline()
        ''
        """
        from numpy import empty
        nnode = int(stream.readline().strip())
        nodes = empty((nnode, 3), dtype='float64')
        ind = 0
        while ind < nnode:
            dat = stream.readline().split()[1:]
            nodes[ind,:] = [float(ent) for ent in dat]
            ind += 1
        # return.
        assert stream.readline().strip() == '$EndNodes'
        return dict(nodes=nodes)

    @classmethod
    def _load_elements(cls, stream, nodes):
        """
        Load element definition of the mesh data.  The node indices defined for
        each element are still 1-based.  It returns :py:attr:`cltpn`,
        :py:attr:`eldim`, :py:attr:`elems`, :py:attr:`elgeo`, :py:attr:`elgrp`,
        :py:attr:`ndim`, :py:attr:`ndmap`, and :py:attr:`usnds` for storage.

        >>> from numpy import array
        >>> nodes = array([[-1.,  0.,  0.], [ 1.,  0.,  0.], [ 0.,  1.,  0.]])
        >>> import StringIO
        >>> stream = StringIO.StringIO(\"\"\"$Elements
        ... 1
        ... 1 2 2 1 22 1 2 3
        ... $EndElements\"\"\") # a triangle.
        >>> stream.readline()
        '$Elements\\n'
        >>> sorted(Gmsh._load_elements(
        ...     stream, nodes).items()) # doctest: +NORMALIZE_WHITESPACE
        [('cltpn', array([3], dtype=int32)),
         ('eldim', array([2], dtype=int32)),
         ('elems', array([[ 3,  1,  2,  3, -1, -1, -1, -1, -1]], dtype=int32)),
         ('elgeo', array([22], dtype=int32)),
         ('elgrp', array([1], dtype=int32)),
         ('ndim', 2),
         ('ndmap', array([0, 1, 2], dtype=int32)),
         ('usnds', array([0, 1, 2], dtype=int32))]
        >>> stream.readline()
        ''
        """
        from numpy import empty, array, arange, unique
        from ..block import Block
        usnds = []
        nelem = int(stream.readline().strip())
        cltpn = empty(nelem, dtype='int32')
        elgrp = empty(nelem, dtype='int32')
        elgeo = empty(nelem, dtype='int32')
        eldim = empty(nelem, dtype='int32')
        elems = empty((nelem, Block.CLMND+1), dtype='int32')
        elems.fill(-1)
        ndim = 0
        iel = 0
        while iel < nelem:
            dat = [int(ent) for ent in stream.readline().split()[1:]]
            tpn = dat[0]
            tag = dat[2:2+dat[1]]
            nds = dat[2+dat[1]:]
            elmap = cls.ELMAP[tpn]
            cltpn[iel] = elmap[2]
            elgrp[iel] = tag[0]
            elgeo[iel] = tag[1]
            eldim[iel] = elmap[0]
            nnd = len(elmap[3])
            elems[iel,0] = nnd
            nds = array(nds, dtype='int32')[elmap[3]]
            usnds.extend(nds)
            elems[iel,1:nnd+1] = nds
            ndim = elmap[0] if elmap[0] > ndim else ndim
            iel += 1
        usnds = array(usnds, dtype='int32') - 1
        usnds.sort()
        usnds = unique(usnds)
        ndmap = empty(nodes.shape[0], dtype='int32')
        ndmap.fill(-1)
        ndmap[usnds] = arange(usnds.shape[0], dtype='int32')
        # returns.
        assert stream.readline().strip() == '$EndElements'
        return dict(ndim=ndim, cltpn=cltpn, elgrp=elgrp, elgeo=elgeo,
                    eldim=eldim, elems=elems, ndmap=ndmap, usnds=usnds)

    @staticmethod
    def _load_physics(stream):
        """
        Load physics groups of the mesh data.  Return :py:attr:`physics` for
        storage.

        >>> import StringIO
        >>> stream = StringIO.StringIO(\"\"\"$PhysicalNames
        ... 1
        ... 2 1 "lower"
        ... $EndPhysicalNames\"\"\")
        >>> stream.readline()
        '$PhysicalNames\\n'
        >>> Gmsh._load_physics(stream)
        {'physics': ['1', '2 1 "lower"']}
        >>> stream.readline()
        ''
        """
        physics = list()
        while True:
            line = stream.readline().strip()
            if line == '$EndPhysicalNames':
                return dict(physics=physics)
            else:
                physics.append(line)

    @staticmethod
    def _load_periodic(stream):
        """
        Load periodic definition of the mesh data.  Return :py:attr:`periodics`
        for storage.

        >>> import StringIO
        >>> stream = StringIO.StringIO(\"\"\"$Periodic
        ... 1
        ... 0 1 3
        ... 1
        ... 1 3
        ... $EndPeriodic\"\"\") # a triangle.
        >>> stream.readline()
        '$Periodic\\n'
        >>> Gmsh._load_periodic(stream) # doctest: +NORMALIZE_WHITESPACE
        {'periodics': [{'ndim': 0,
                        'stag': 1,
                        'nodes': array([[1, 3]], dtype=int32),
                        'mtag': 3}]}
        >>> stream.readline()
        ''
        """
        from numpy import array
        # read the total number of periodic relations.
        nent = int(stream.readline())
        # loop over relations.
        periodics = list()
        while len(periodics) < nent:
            # read data for this periodic realtion.
            ndim, stag, mtag = map(int, stream.readline().split())
            nnode = int(stream.readline())
            nodes = array([map(int, stream.readline().split())
                           for it in xrange(nnode)], dtype='int32')
            # append the relation.
            periodics.append(dict(
                ndim=ndim, stag=stag, mtag=mtag, nodes=nodes))
        # return.
        assert '$EndPeriodic' == stream.readline().strip()
        return dict(periodics=periodics)

    def _parse_physics(self):
        """
        Parse physics groups of the mesh data.  Process :py:attr:`physics` and
        stores :py:attr:`intels`.
        """
        from numpy import arange, concatenate, unique
        elidx = arange(self.elems.shape[0], dtype='int32')
        if not self.physics:
            self.intels = elidx[self.eldim == self.ndim]
            return
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
    # Loading methods.
    ############################################################################

    ############################################################################
    # Converting methods.
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
        assert len(self.usnds) > 0
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
        # make a hash table for boundary faces with nodes as keys.
        bfcndh = dict()
        for ifc in bfcs:
            for ind in blk.fcnds[ifc,1:blk.fcnds[ifc,0]+1]:
                bfcndh.setdefault(ind, set()).add(ifc)
        for name, dim, els in self.physics:
            if dim == self.ndim:
                continue
            if onlybcnames and name not in onlybcnames: # skip unwanted BCs.
                continue
            bndfcs = []
            for iel in els:
                elem = self.elems[iel]
                nnd = elem[0]
                nds = self.ndmap[elem[1:1+nnd]-1]
                # search for face.
                fset = bfcndh[nds[0]] & bfcndh[nds[1]]
                for ind in nds[2:]:
                    fset &= bfcndh[ind]
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
    # Converting methods.
    ############################################################################


class GmshIO(FormatIO):
    """
    Proxy to Gmsh file format.
    """

    def load(self, stream, bcrej=None, bcmapper=None):
        """
        Load block from stream with BC mapper applied.
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
