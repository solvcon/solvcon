# -*- coding: UTF-8 -*-
#
# Copyright (c) 2008, Yung-Yu Chen <yyc@solvcon.net>
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
Gambit Neutral file.
"""

from .core import FormatIO

class ElementGroup(object):
    """
    One single element group information in Gambit Neutral file.

    @ivar ngp: element group index (1-based).
    @type ngp: int
    @ivar nelgp: number elements in this group.
    @type nelgp: int
    @ivar mtyp: material type (0: undefined, 1: conjugate, 2: fluid, 3: porous,
        4: solid, 5: deformable).
    @type mtyp: int
    @ivar nflags: number of solver dependent flags.
    @type nflags: int
    @ivar solver: array of solver dependent flags of shape of (nflags).
    @type solver: numpy.ndarray
    @ivar elems: elements array of shape of (nelgp).
    @type elems: numpy.ndarray
    """
    def __init__(self, data=None):
        from numpy import empty
        self.ngp = None # retained as 1-based.
        self.nelgp = None
        self.mtyp = None
        self.nflags = None
        self.elmmat = ''
        self.solver = empty(0)
        self.elems = empty(0)
        # run parser.
        if data != None: self._parse(data)

    def __str__(self):
        return '[Group #%d(%s): %d elements]' % (
            self.ngp, self.elmmat, self.nelgp)

    def _parse(self, data):
        """
        Parse given string data for element group.  Set all instance variables.

        @param data: string data for element group.
        @type data: string
        @return: nothing
        """
        from numpy import fromstring
        # parse header.
        control, enttype, solver, data = data.split('\n', 3)
        # parse control.
        self.ngp, self.nelgp, self.mtyp, self.nflags = [
                int(val) for val in control.split()[1::2]]
        # get name.
        self.elmmat = enttype.strip()
        # get solver flags.
        self.solver = fromstring(solver, dtype='int32', sep=' ')
        # parse into array and renumber.
        self.elems = fromstring(data, dtype='int32', sep=' ')-1

class BoundaryCondition(object):
    """
    Hold boundary condition values.

    @cvar CLFCS_RMAP: map clfcs definition back from block object to neutral 
        object.
    @type CLFCS_RMAP: dict

    @ivar name: name of boundary condition.
    @type name: str
    @ivar itype: type of data (0: nodal, 1: elemental).
    @type itype: int
    @ivar nentry: number of entry (nodes or elements/cells).
    @type nentry: int
    @ivar nvalues: number of values for each data record.
    @type nvalues: int
    @ivar ibcode: 1D array of boundary condition code.
    @type ibcode: numpy.ndarray
    @ivar values: array of values attached to each record.
    @type values: numpy.ndarray
    """
    def __init__(self, data=None):
        from numpy import empty
        self.name = ''
        self.itype = None
        self.nentry = None
        self.nvalues = None
        self.ibcode = empty(0)
        self.elems = empty(0)
        self.values = empty(0)
        # run parser.
        if data != None: self._parse(data)

    def __str__(self):
        return '[BC "%s": %d entries with %d values]' % (
            self.name, self.nentry, self.nvalues)

    def _parse(self, data):
        """
        Parse given data string to boundary condition set.  Set all instance
        variables.

        @param data: string data for boundary condition set.
        @type data: str
        @return: nothing
        """
        from numpy import fromstring
        # parse header.
        header, data = data.split('\n', 1)
        self.name = header[:32].strip()
        tokens = fromstring(header[32:], dtype='int32', sep=' ')
        self.itype, self.nentry, self.nvalues = tokens[:3]
        self.ibcode = tokens[3:].copy()
        # parse entries.
        if self.itype == 0: # for nodes.
            arr = fromstring(data, dtype='int32', sep=' ').reshape(
                (self.nentry, self.nvalues+1))
            self.elems = (arr[:,0]-1).copy()
            arr = fromstring(data, dtype='float64', sep=' ').reshape(
                (self.nentry, self.nvalues+1))
            self.values = (arr[:,1:]).copy()
        elif self.itype == 1: # for elements/cells.
            arr = fromstring(data, dtype='int32', sep=' ').reshape(
                (self.nentry, self.nvalues+3))
            self.elems = arr[:,:3].copy()
            self.elems[:,0] -= 1
            arr = fromstring(data, dtype='float64', sep=' ').reshape(
                (self.nentry, self.nvalues+3))
            self.values = (arr[:,3:]).copy()
        else:
            raise ValueError, \
                "itype has to be either 0/1, but get %d"%self.itype

    # define map for clfcs (from block to neu).
    CLFCS_RMAP = {}
    # tpn=1: edge.
    CLFCS_RMAP[1] = [1,2]
    # tpn=2: quadrilateral.
    CLFCS_RMAP[2] = [1,2,3,4]
    # tpn=3: triangle.
    CLFCS_RMAP[3] = [1,2,3]
    # tpn=4: hexahedron.
    CLFCS_RMAP[4] = [5,2,6,4,1,3]
    # tpn=5: tetrahedron.
    CLFCS_RMAP[5] = [1,2,4,3]
    # tpn=6: prism.
    CLFCS_RMAP[6] = [4,5,3,1,2]
    # tpn=6: pyramid.
    CLFCS_RMAP[7] = [5,2,3,4,1]

    def tobc(self, blk):
        """
        Extract gambit boundary condition information from self into BC object.  
        Only process element/cell type of (gambit) boundary condition, and 
        return None while nodal BCs encountered.

        @param blk: Block object for reference, nothing will be altered.
        @type blk: solvcon.block.Block
        @return: generic BC object.
        @rtype: solvcon.boundcond.BC
        """
        from numpy import empty
        from ..boundcond import BC
        clfcs_rmap = self.CLFCS_RMAP
        # process only element/cell type of bc.
        if self.itype != 1:
            return None
        # extrace boundary face list.
        facn = empty((self.nentry,3), dtype='int32')
        facn.fill(-1)
        ibnd = 0
        for entry in self.elems:
            icl, nouse, it = entry[:3]
            tpn = blk.cltpn[icl]
            facn[ibnd,0] = blk.clfcs[icl, clfcs_rmap[tpn][it-1]]
            ibnd += 1
        # craft BC object.
        bc = BC(fpdtype=blk.fpdtype)
        bc.name = self.name
        slct = facn[:,0].argsort()   # sort face list for bc object.
        bc.facn = facn[slct]
        bc.value = self.values[slct]
        # finish.
        return bc

class GambitNeutralParser(object):
    """
    Parse and store information of a Gambit Neutral file.

    @ivar data: data to be parsed.
    @type data: str
    @ivar neu: GambitNeutral object to be saved.
    @type neu: solvcon.io.gambit.neutral.GambitNeutral
    """
    def __init__(self, data, neu):
        """
        @param data: data to be parsed.
        @type data: str
        @param neu: GambitNeutral object to be saved.
        @type neu: solvcon.io.gambit.neutral.GambitNeutral
        """
        self.data = data
        self.neu = neu

    def parse(self):
        data = self.data
        neu = self.neu
        sections = data.split('ENDOFSECTION\n')
        for section in sections:
            header = section.split('\n', 1)[0]
            processor = None
            for mark in self.processors:
                if mark in header:
                    processor = self.processors[mark]
                    break
            if processor:
                processor(section, neu)

    processors = {}

    def _control_info(data, neu):
        """
        Take string data for "CONTROL INFO" and parse it to GambitNeutral
        object.  Set:
            - header
            - title
            - data_source
            - numnp
            - nelem
            - ngrps
            - nbsets
            - ndfcd
            - ndfvl

        @param data: sectional data.
        @type data: str
        @param neu: object to be saved.
        @type neu: solvcon.io.gambit.neutral.GambitNetral
        @return: nothing
        """
        from numpy import fromstring
        data = data.rstrip()
        records = data.splitlines()
        neu.header = records[1].strip()
        neu.title = records[2].strip()
        neu.data_source = records[3].strip()
        values = fromstring(records[6], dtype='int32', sep=' ')
        neu.numnp, neu.nelem, neu.ngrps, \
            neu.nbsets, neu.ndfcd, neu.ndfvl = values
    processors['CONTROL INFO'] = _control_info

    def _nodal_coordinate(data, neu):
        """
        Take string data for "NODAL COORDINATES" and parse it to GambitNuetral
        object. Set:
            - nodes

        @param data: sectional data.
        @type data: str
        @param neu: object to be saved.
        @type neu: solvcon.io.gambit.neutral.GambitNetral
        @return: nothing
        """
        from numpy import fromstring, empty
        # discard header.
        data = data.split('\n', 1)[-1]
        # parse into array and reshape to 2D array.
        nodes = fromstring(data, dtype='float64', sep=' ')
        nodes = nodes.reshape((neu.numnp, (neu.ndfcd+1)))
        # renumber according to first value of each line.
        # NOTE: unused number contains garbage.
        number = nodes[:,0].astype(int) - 1
        newnodes = empty((number.max()+1,neu.ndfcd))
        newnodes[number] = nodes[number,1:]
        # set result to neu.
        neu.nodes = newnodes
    processors['NODAL COORDINATE'] = _nodal_coordinate

    def _elements_cells(data, neu):
        """
        Take string data for "ELEMENTS/CELLS" and parse it to GambitNeutral
        object. Set:
            - elems

        @param data: sectional data.
        @type data: str
        @param neu: object to be saved.
        @type neu: solvcon.io.gambit.neutral.GambitNetral
        @return: nothing
        """
        from numpy import fromstring, empty
        # discard header.
        data = data.split('\n', 1)[-1]
        # parse into array.
        serial = fromstring(data, dtype='int32', sep=' ')
        # parse element data -- 1st pass:
        # element index, shape, and number of nodes.
        meta = empty((neu.nelem, 3), dtype='int32')
        ielem = 0
        ival = 0
        while ielem < neu.nelem:
            meta[ielem,:] = serial[ival:ival+3]
            ival += 3+meta[ielem,2]
            ielem += 1
        # parse element data -- 2nd pass:
        # node definition.
        maxnnode = meta[:,2].max()
        elems = empty((neu.nelem, maxnnode+2), dtype='int32')
        ielem = 0
        ival = 0
        while ielem < neu.nelem:
            elems[ielem,2:2+meta[ielem,2]] = serial[ival+3:ival+3+meta[ielem,2]]
            ival += 3+meta[ielem,2]
            ielem += 1
        elems[:,:2] = meta[:,1:]    # copy the first two columns from meta.
        elems[:,2:] -= 1    # renumber node indices in elements.
        # set result to neu.
        neu.elems = elems
    processors['ELEMENTS/CELLS'] = _elements_cells

    def _element_group(data, neu):
        """
        Take string data for "ELEMENTS GROUP" and parse it to GambitNeutral
        object. Set:
            - grps

        @param data: sectional data.
        @type data: str
        @param neu: object to be saved.
        @type neu: solvcon.io.gambit.neutral.GambitNetral
        @return: nothing
        """
        from numpy import fromstring, empty
        # discard header.
        data = data.split('\n', 1)[-1]
        # build group.
        neu.grps.append(ElementGroup(data))
    processors['ELEMENT GROUP'] = _element_group

    def _boundary_conditions(data, neu):
        """
        Take string data for "BOUNDARY CONDITIONS" and parse it to
        GambitNeutral object. Set:
            - bcs

        @param data: sectional data.
        @type data: str
        @param neu: object to be saved.
        @type neu: solvcon.io.gambit.neutral.GambitNetral
        @return: nothing
        """
        from numpy import fromstring, empty
        # discard header.
        data = data.split('\n', 1)[-1]
        # build group.
        neu.bcs.append(BoundaryCondition(data))
    processors['BOUNDARY CONDITIONS'] = _boundary_conditions

class GambitNeutralReader(object):
    """
    Read and store information of a Gambit Neutral file line by line.

    @ivar neuf: source file.
    @itype neuf: file
    @ivar neu: GambitNeutral object to be saved to.
    @itype neu: solvcon.io.gambit.neutral.GambitNeutral
    """
    def __init__(self, neuf, neu):
        self.neuf = neuf
        self.neu = neu
    def read(self):
        neuf = self.neuf
        neu = self.neu
        while True:
            toks = neuf.readline()[:20].strip().lower().split()
            header = []
            for tok in toks:
                header.extend(tok.split('/'))
            header = '_'.join(header)
            method = getattr(self, '_'+header, None)
            if method != None:
                method(neuf, neu)
                assert neuf.readline().strip() == 'ENDOFSECTION'
            else:
                break
    @staticmethod
    def _control_info(neuf, neu):
        neu.header = neuf.readline().strip()
        neu.title = neuf.readline().strip()
        neu.data_source = neuf.readline().strip()
        for i in range(2): neuf.readline()
        line = neuf.readline().rstrip()
        neu.numnp = int(line[1:10])
        neu.nelem = int(line[11:20])
        neu.ngrps = int(line[21:30])
        neu.nbsets = int(line[31:40])
        neu.ndfcd = int(line[41:50])
        neu.ndfvl = int(line[51:60])
    @staticmethod
    def _nodal_coordinates(neuf, neu):
        from numpy import empty
        nodes = empty((neu.numnp, neu.ndfcd), dtype='float64')
        nodeids = empty(neu.numnp, dtype='int32')
        ndim = neu.ndfcd
        nnode = neu.numnp
        ind = 0
        while ind < nnode:
            line = neuf.readline()
            nodeids[ind] = int(line[:10])
            for idm in range(ndim):
                nodes[ind,idm] = float(line[10+20*idm:10+20*(idm+1)])
            ind += 1
        # renumber according to first value of each line.
        # NOTE: unused number contains garbage.
        nodeids -= 1
        neu.nodes = empty((nodeids.max()+1, neu.ndfcd), dtype='float64')
        neu.nodes[nodeids] = nodes[nodeids]
    @staticmethod
    def _elements_cells(neuf, neu):
        from numpy import empty
        from ..block import MAX_CLNND
        ncell = neu.nelem
        elems = empty((ncell, MAX_CLNND+2), dtype='int32')
        icl = 0
        while icl < ncell:
            line = neuf.readline()
            elems[icl,0] = int(line[9:11])
            elems[icl,1] = ncl = int(line[12:14])
            for it in range(min(ncl, 7)):
                elems[icl,2+it] = int(line[15+8*it:15+8*(it+1)]) - 1
            if ncl > 7:
                line = neuf.readline()
            elems[icl,2+7] = int(line[15:15+8]) - 1
            icl += 1
        neu.elems = elems
    @classmethod
    def _element_group(cls, neuf, neu):
        emg = ElementGroup()
        # group statistics.
        line = neuf.readline()
        emg.ngp = int(line[7:7+10])
        emg.nelgp = int(line[28:28+10])
        emg.mtyp = int(line[49:49+10])
        emg.nflags = int(line[68:68+10])
        # group name.
        line = neuf.readline()
        emg.elmmat = line.strip()
        # solver data.
        emg.solver = cls._read_values(neuf, 8, emg.nflags, 'int32')
        # element data.
        emg.elems = cls._read_values(neuf, 8, emg.nelgp, 'int32')-1
        # append group.
        neu.grps.append(emg)
    @staticmethod
    def _boundary_conditions(neuf, neu):
        from numpy import empty
        bc = BoundaryCondition()
        # control record.
        line = neuf.readline()
        bc.name = line[:32].strip()
        bc.itype = int(line[32:32+10])
        bc.nentry = nbfc = int(line[42:42+10])
        bc.nvalues = nval = int(line[52:52+10])
        if bc.itype == 0: # nodes.
            bc.elems = elems = empty(nbfc, dtype='int32')
            bc.values = values = empty((nbfc, nval), dtype='float64')
            ibfc = 0
            while ibfc < nbfc:
                line = neuf.readline()
                elems[ibfc] = int(line[0:10])
                values[ibfc] = [float(line[10+20*it:10+20*(it+1)]) for it in
                    range(nval)]
                ibfc += 1
        elif bc.itype == 1: # elements/cells.
            bc.elems = elems = empty((nbfc, 3), dtype='int32')
            bc.values = values = empty((nbfc, nval), dtype='float64')
            ibfc = 0
            while ibfc < nbfc:
                line = neuf.readline()
                elems[ibfc] = (int(line[0:10])-1,
                    int(line[10:15]), int(line[15:20]))
                values[ibfc] = [float(line[20+20*it:20+20*(it+1)]) for it in
                    range(nval)]
                ibfc += 1
        else:
            raise ValueError, 'only 0/1 of itype is allowed'
        assert ibfc == nbfc
        # append.
        neu.bcs.append(bc)

    @staticmethod
    def _read_values(neuf, width, nval, dtype):
        """
        Read homogeneous values from the current position of the opened
        neutral file.

        @param neuf: neutral file.
        @type neuf: file
        @param width: character width per value.
        @type width: int
        @param nval: number of values to read.
        @type nval: int
        @param dtype: dtype string to construct ndarray.
        @type dtype: str
        @return: read array.
        @rtype: numpy.ndarray
        """
        from numpy import empty
        # determine type.
        if dtype.startswith('int'):
            vtype = int
        elif dtype.startswith('float'):
            vtype = float
        else:
            raise TypeError, '%s not supported'%dtype
        # allocate array.
        arr = empty(nval, dtype=dtype)
        # read.
        iline = 0
        ival = 0
        while ival < nval:
            line = neuf.readline()
            iline += 1
            nchar = len(line)
            line = line.rstrip()
            nc = len(line)
            if nc%width != 0:
                raise IndexError, 'not exact chars at line %d'%(ival/iline)
            nt = nc/width
            arr[ival:ival+nt] = [vtype(line[8*it:8*(it+1)]) for it in range(nt)]
            ival += nt
        assert ival == nval
        return arr

class GambitNeutral(object):
    """
    Represent information in a Gambit Neutral file.

    @cvar CLTPN_MAP: map cltpn from self to block.
    @type CLTPN_MAP: numpy.ndarray
    @cvar CLNDS_MAP: map clnds definition from self to block.
    @type CLNDS_MAP: dict
    @cvar CLFCS_RMAP: map clfcs definition back from block to self.
    @type CLFCS_RMAP: dict

    @ivar header: file header string.
    @type header: str
    @ivar title: title for this file.
    @type title: str
    @ivar data_source: identify the generation of the file from which program 
        and version.
    @type data_source: str
    @ivar numnp: number of nodes.
    @type numnp: int
    @ivar nelem: number of elements.
    @type nelem: int
    @ivar ngrps: number of element groups.
    @type ngrps: int
    @ivar nbsets: number of boundary condition sets.
    @type nbsets: int
    @ivar ndfcd: number of coordinate directions (2/3).
    @type ndfcd: int
    @ivar ndfvl: number of velocity components (2/3).
    @type ndfvl: int
    @ivar nodes: nodes array of shape of (numnp, ndfcd).
    @type nodes: numpy.ndarray
    @ivar elems: elements array of shape of (nelem, :).
    @type elems: numpy.ndarray
    @ivar grps: list of ElementGroup objects.
    @type grps: list
    @ivar bcs: list of BoundaryCondition objects.
    @type bcs: list
    """
    def __init__(self, data):
        from numpy import empty
        # control info.
        self.header = ''
        self.title = ''
        self.data_source = ''
        self.numnp = None
        self.nelem = None
        self.ngrps = None
        self.nbsets = None
        self.ndfcd = None
        self.ndfvl = None
        # node info.
        self.nodes = empty(0)
        # element/cell info.
        self.elems = empty(0)
        # element group info.
        self.grps = []
        # boundary conditions info.
        self.bcs = []
        # parse/read.
        if hasattr(data, 'read'):
            GambitNeutralReader(data, self).read()
        else:
            GambitNeutralParser(data, self).parse()

    def __str__(self):
        return '[Neutral (%s): %d nodes, %d elements, %d groups, %d bcs]' % (
            self.title, self.numnp, self.nelem, len(self.grps), len(self.bcs))

    @property
    def ndim(self):
        return self.ndfcd
    @property
    def nnode(self):
        return self.nodes.shape[0]
    @property
    def ncell(self):
        return self.elems.shape[0]

    def toblock(self, onlybcnames=None, bcname_mapper=None, fpdtype=None,
            use_incenter=False):
        """
        Convert GambitNeutral object to Block object.

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
        from ..block import Block
        # create corresponding block according to GambitNeutral object.
        blk = Block(ndim=self.ndim, nnode=self.nnode, ncell=self.ncell,
            fpdtype=fpdtype, use_incenter=use_incenter)
        self._convert_interior_to(blk)
        blk.build_interior()
        self._convert_bc_to(blk,
            onlynames=onlybcnames, name_mapper=bcname_mapper)
        blk.build_boundary()
        blk.build_ghost()
        return blk

    from numpy import array
    # define map for cltpn (from self to block).
    CLTPN_MAP = array([0, 1, 2, 3, 4, 6, 5, 7], dtype='int32')
    # define map for clnds (from self to block).
    CLNDS_MAP = {}
    # tpn=1: edge.
    CLNDS_MAP[1] = {}
    CLNDS_MAP[1][2] = [2,3] # 2 nodes.
    CLNDS_MAP[1][3] = [2,4] # 3 nodes.
    # tpn=2: quadrilateral.
    CLNDS_MAP[2] = {}
    CLNDS_MAP[2][4] = [2,3,4,5] # 4 nodes.
    CLNDS_MAP[2][8] = [2,4,6,8] # 8 nodes.
    CLNDS_MAP[2][9] = [2,4,6,8] # 9 nodes.
    # tpn=3: triangle.
    CLNDS_MAP[3] = {}
    CLNDS_MAP[3][3] = [2,3,4]   # 3 nodes.
    CLNDS_MAP[3][6] = [2,4,6]   # 6 nodes.
    CLNDS_MAP[3][7] = [2,4,6]   # 7 nodes.
    # tpn=4: brick.
    CLNDS_MAP[4] = {}
    CLNDS_MAP[4][8] = [2,3,5,4,6,7,9,8] # 8 nodes.
    CLNDS_MAP[4][20] = [2,4,9,7,14,16,21,19]    # 20 nodes.
    CLNDS_MAP[4][27] = [2,4,10,8,20,22,28,26]   # 27 nodes.
    # tpn=5: tetrahedron.
    CLNDS_MAP[5] = {}
    CLNDS_MAP[5][4] = [2,3,4,5] # 4 nodes.
    CLNDS_MAP[5][10] = [2,4,7,11]   # 10 nodes.
    # tpn=6: wedge.
    CLNDS_MAP[6] = {}
    CLNDS_MAP[6][6] = [2,4,3,5,7,6] # 6 nodes.
    CLNDS_MAP[6][15] = [2,7,4,11,16,13] # 15 nodes.
    CLNDS_MAP[6][18] = [2,7,4,14,19,16] # 18 nodes.
    # tpn=7: pyramid.
    CLNDS_MAP[7] = {}
    CLNDS_MAP[7][5] = [2,3,5,4,6]   # 5 nodes.
    CLNDS_MAP[7][13] = [2,4,9,7,14] # 13 nodes.
    CLNDS_MAP[7][14] = [2,4,10,8,15]    # 14 nodes.
    CLNDS_MAP[7][18] = [2,4,10,8,19]    # 18 nodes.
    CLNDS_MAP[7][19] = [2,4,10,8,20]    # 19 nodes.

    def _convert_interior_to(self, blk):
        """
        Convert interior information, i.e., connectivities, from GambitNeutral 
        to Block object.

        @param blk: to-be-written Block object.
        @type blk: solvcon.block.Block
        @return: nothing.
        """
        from numpy import array
        from ..block import elemtype

        cltpn_map = self.CLTPN_MAP
        clnds_map = self.CLNDS_MAP

        # copy nodal coordinate data.
        blk.ndcrd[:,:] = self.nodes[:,:]
        # copy node difinition in cells.
        cltpn = blk.cltpn
        clnds = blk.clnds
        ncell = self.ncell
        icell = 0
        while icell < ncell:
            # translate tpn from GambitNeutral to Block.
            tpn = cltpn_map[self.elems[icell,0]]
            cltpn[icell] = tpn
            # translate clnds from GambitNeutral to Block.
            nnd = elemtype[tpn,2]
            nnd_self = self.elems[icell,1]
            clnds[icell,0] = nnd
            clnds[icell,1:nnd+1] = self.elems[icell,clnds_map[tpn][nnd_self]]
            # advance cell.
            icell += 1

        # create cell groups for the block.
        clgrp = blk.clgrp
        for grp in self.grps:
            igrp = len(blk.grpnames)
            assert grp.ngp == igrp+1
            clgrp[grp.elems] = igrp
            blk.grpnames.append(grp.elmmat)

    def _convert_bc_to(self, blk, onlynames=None, name_mapper=None):
        """
        Convert boundary condition information from GambitNeutral object into 
        Block object.
        
        @param blk: to-be-written Block object.
        @type blk: solvcon.block.Block
        @keyword onlynames: positively list wanted names of BCs.
        @type onlynames: list
        @keyword name_mapper: map name to bc type and value dictionary; the two
            objects are organized in a tuple.
        @type name_mapper: dict
        @return: nothing.
        """
        # process all neutral bc objects.
        for neubc in self.bcs:
            # extract boundary faces from neutral bc object.
            bc = neubc.tobc(blk)
            if bc is None:   # skip if got nothing.
                continue
            # skip unwanted BCs.
            if onlynames:
                if bc.name not in onlynames:
                    continue
            # recreate BC according to name mapping.
            if name_mapper is not None:
                bct, vdict = name_mapper.get(bc.name, None)
                if bct is not None:
                    bc = bct(bc=bc)
                    bc.feedValue(vdict)
            # save to block object.
            bc.sern = len(blk.bclist)
            bc.blk = blk
            blk.bclist.append(bc)

class NeutralIO(FormatIO):
    """
    Proxy to gambit neutral file format.
    """
    def load(self, stream, bcrej=None):
        """
        Load block from stream with BC mapper applied.

        @keyword stream: file object or file name to be read.
        @type stream: file or str
        @keyword bcrej: names of the BC to reject.
        @type bcrej: list
        @return: the loaded block.
        @rtype: solvcon.block.Block
        """
        import gzip
        # load gambit neutral file.
        if isinstance(stream, basestring):
            if stream.endswith('.gz'):
                opener = gzip.open
            else:
                opener = open
            stream = opener(stream)
        neu = GambitNeutral(stream)
        stream.close()
        # convert loaded neutral object into block object.
        if bcrej:
            onlybcnames = list()
            for bc in neu.bcs:
                if bc.name not in bcrej:
                    onlybcnames.append(bc.name)
        else:
            onlybcnames = None
        blk = neu.toblock(onlybcnames=onlybcnames)
        return blk

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        fname = sys.argv[1]
        neu = GambitNeutral(open(fname).read())
        sys.stdout.write("Gambit Neutral object: %s" % neu)
        if neu.grps or neu.bcs:
            sys.stdout.write(", with:\n")
        for lst in neu.grps, neu.bcs:
            if len(lst) > 0:
                for obj in lst:
                    sys.stdout.write("  %s\n" % obj)
            else:
                sys.stdout.write("\n")
    else:
        sys.stdout.write("usage: %s <file name>\n" % sys.argv[0])
