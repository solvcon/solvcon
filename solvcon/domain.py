# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2009 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""
Domain decomposition.
"""

class Domain(object):
    """
    Abstraction of computation domain.  It is the most basic domain that holds
    one single block.

    @ivar blk: the whole block.  Can be None.
    @itype blk: solvcon.block.Block
    """
    def __init__(self, blk, *args, **kw):
        """
        @param blk: the Block object to be used.  If no blk is intended, None
            should be passed in explicitly.
        @type blk: solvcon.block.Block
        """
        self.blk = blk
        super(Domain, self).__init__(*args, **kw)

    def bind(self):
        self.blk.bind()
    def unbind(self):
        self.blk.unbind()

class Partitioner(object):
    """
    Interface to METIS library for domain partition.

    @ivar blk: Block object to be partitioned.
    @itype blk: solvcon.block.Block
    """

    def __init__(self, blk):
        self.blk = blk
        super(Partitioner, self).__init__()

    def __call__(self, npart):
        return self._partgraph(self.blk, npart)

    @classmethod
    def _calculate_rcells(cls, blk):
        """
        Calculate related cells.
        """
        from ctypes import POINTER, c_int, byref
        from numpy import empty
        from .dependency import _clib_solvcon_d
        from .conf import env
        intptr = POINTER(c_int)
        rcells = empty((blk.ncell, blk.CLMFC), dtype='int32')
        rcellno = empty(blk.ncell, dtype='int32')
        if not env.use_fortran:
            _clib_solvcon_d.build_rcells(
                byref(blk.create_msd()),
                rcells.ctypes.data_as(intptr),
                rcellno.ctypes.data_as(intptr),
            )
        else:
            _clib_solvcon_d.build_rcells_( 
                byref(c_int(blk.CLMFC)),
                byref(c_int(blk.nface)),
                byref(c_int(blk.ncell)),
                blk.clfcs.ctypes.data_as(intptr),
                blk.fccls.ctypes.data_as(intptr),
                # output.
                rcells.ctypes.data_as(intptr),
                rcellno.ctypes.data_as(intptr),
            )
        return rcells, rcellno

    @classmethod
    def _generate_csr(cls, blk):
        """
        Interpret a valid unstructured grid block into CSR (Compressed Storage
        Foramt) that ParMETIS/METIS uses.
        """
        from numpy import empty, add
        ncell = blk.ncell
        # set convenient local names.
        rcells, rcellno = cls._calculate_rcells(blk)
        # build xadj: cell boundaries.
        xadj = empty(ncell+1, dtype='int32')
        xadj[0] = 0
        xadj[1:] = add.accumulate(rcellno)
        # build adjncy: edge/relations.
        adjncy = empty(xadj[-1], dtype='int32')
        i = 0
        while i < ncell:
            trcells = rcells[i]
            adjncy[xadj[i]:xadj[i+1]] = trcells[trcells!=-1]
            i += 1
        return xadj, adjncy

    @classmethod
    def _partgraph(cls, blk, npart, vwgt=None):
        """
        Call METIS to do the partition.
        """
        from ctypes import POINTER, c_int, byref
        from numpy import empty
        from .dependency import _clib_metis
        intptr = POINTER(c_int)
        xadj, adjncy = cls._generate_csr(blk)
        # weighting.
        if vwgt == None:
            vwgt = empty(1, dtype='int32')
        if len(vwgt) == blk.ncell:
            wgtflag = 2
        else:
            vwgt.fill(0)
            wgtflag = 0
        # FIXME: not consistent when len(vwgt) == blk.ncell.
        adjwgt = empty(1, dtype='int32')
        adjwgt.fill(0)
        # options.
        options = empty(5, dtype='int32')
        options.fill(0)
        # do the partition.
        part = empty(blk.ncell, dtype='int32')
        edgecut = c_int(0)
        _clib_metis.METIS_PartGraphKway(
            byref(c_int(blk.ncell)),
            xadj.ctypes.data_as(intptr),
            adjncy.ctypes.data_as(intptr),
            vwgt.ctypes.data_as(intptr),
            adjwgt.ctypes.data_as(intptr),
            byref(c_int(wgtflag)),
            byref(c_int(0)),
            byref(c_int(npart)),
            options.ctypes.data_as(intptr),
            # output.
            byref(edgecut),
            part.ctypes.data_as(intptr),
        )
        return edgecut.value, part

class Collective(Domain, list):
    """
    Domain retaining the relationship between the collective and the decomposed
    blocks.

    @ivar edgecut: number of edge cut by METIS.
    @itype edgecut: int
    @ivar part: array holding the partitioned indices.
    @itype part: numpy.ndarray
    @ivar idxinfo: a list contains another list that holds for nodes, faces,
        and cells that belong to each partitioned sub-block.  It looks like
        [[mynds, myfcs, mycls], [mynds, myfcs, mycls], ...].  The index
        information actually can serve as local to global index mapper.  For
        example, dom.idxinfo[0][2][33] indicates the global index of the 34-th 
        local cell in block #0.
    @itype idxinfo: list
    @ivar mappers: a list contains mapper for nodes, faces, and cells from
        global index to local index.  It looks like [ndmaps, fcmaps, clmaps].
        The shape of fcmaps and clmaps is always (nface, 5) and (ncell, 2),
        respectively.  The shape of ndmaps is (nnode, 1+2*ndmblk), where ndmblk
        is the maximal number of blocks sharing a single node.
    @itype mappers: list
    @ivar interfaces: a list contains pairs of all the interface BC objects.
        It looks like [[ijbc, jibc], [ijbc, jibc], ...].  Interfaces must occur
        in pairs.  An interface related to block #jblk from block #iblk
        guarantees the other interface related to block #iblk from block #jblk.
        The numbers of border faces are the same for these two mutual
        interfaces.
    @itype interfaces: list
    """

    IFSLEEP = 1.e-10    # in seconds.

    def __init__(self, *args, **kw):
        super(Collective, self).__init__(*args, **kw)
        self.edgecut = 0
        self.part = None
        self.idxinfo = list()
        self.mappers = list()
        self.interfaces = list()

    def bind(self):
        super(Collective, self).bind()
        for blk in self:
            blk.bind()
    def unbind(self):
        super(Collective, self).unbind()
        for blk in self:
            blk.unbind()

    @property
    def ifacelists(self):
        """
        Create the ifacelist for each solver object to initialize the
        interface exchanger.

        @return: list of ifacelist.
        @rtype: list
        """
        from numpy import array, concatenate
        interfaces = self.interfaces
        nblk = len(interfaces)
        # create exchange pairs.
        pairlist = list()
        for iblk in range(nblk):
            for jblk in range(nblk):
                if iblk >= jblk:
                    continue
                if interfaces[iblk][jblk] != None:
                    pairlist.append((iblk,jblk))
        # determine exchanging order.
        stages = list()
        while len(pairlist) > 0:
            # filter for the current stage.
            stage = list()
            instage = list()
            ipair = 0
            while ipair < len(pairlist):
                pair = pairlist[ipair]
                if pair[0] not in instage and pair[1] not in instage:
                    stage.append(pair)
                    instage.extend(pair)
                    del pairlist[ipair]
                else:
                    ipair += 1
            # assert that all indices in the stage are different to each other.
            instage.sort()
            for it in range(len(instage)-1):
                assert instage[it+1] > instage[it]
            # append to stages.
            stages.append(stage)
        # create ifacelists.
        iflists = [list() for iblk in range(nblk)]
        istage = 0
        for stage in stages:
            for pair in stage:
                iflists[pair[0]].append(pair)
                iflists[pair[1]].append(pair)
            for iflist in iflists:
                if len(iflist) <= istage:
                    iflist.append(-self.IFSLEEP)    # negative for skip.
            istage += 1
        ## checking.
        hasmax = False
        for iblk in range(nblk):
            iflist = iflists[iblk]
            # assert the pair is for the blk.
            for pair in iflist:
                if not pair < 0:
                    assert iblk in pair
        return iflists

    @classmethod
    def _count_max_nodeinblock(cls, blk, part):
        """
        @param blk: the global Block object to count.
        @type blk: solvcon.block.Block
        @param part: partition.
        @type part: numpy.ndnarray

        @return: maximal number of blocks that a node is shared with.
        @rtype: int
        """
        from ctypes import c_int, byref, POINTER
        from .dependency import _clib_solvcon_d
        from .conf import env
        intptr = POINTER(c_int)
        max_ndcnt = c_int(0)
        if not env.use_fortran:
            _clib_solvcon_d.count_max_nodeinblock(
                byref(blk.create_msd()),
                part.ctypes.data_as(intptr),
                c_int(part.max()+1),
                byref(max_ndcnt),
            )
        else:
            _clib_solvcon_d.count_max_nodeinblock_(
                byref(c_int(blk.CLMND)),
                byref(c_int(blk.ncell)),
                byref(c_int(part.max()+1)),
                byref(c_int(blk.nnode)),
                blk.clnds.ctypes.data_as(intptr),
                part.ctypes.data_as(intptr),
                byref(max_ndcnt),
            )
        return max_ndcnt.value

    def partition(self, nblk):
        """
        Partition the whole block into sub-blocks.  Set self.idxinfo .

        @param nblk: number of sub-blocks to be partitioned.
        @type nblk: int
        """
        assert self.blk
        assert len(self.idxinfo) == 0
        from numpy import empty, arange, unique
        blk = self.blk
        # call partitioner.
        edgecut, part = Partitioner(blk)(nblk)
        self.edgecut = edgecut
        self.part = part
        # numbering.
        clidx = arange(blk.ncell, dtype='int32')
        idxinfo = list()
        for iblk in range(nblk):
            mycls = clidx[part==iblk]
            myfcs = unique(blk.clfcs[mycls,1:].flatten())
            myfcs = myfcs[myfcs>-1]
            myfcs.sort()
            mynds = unique(blk.clnds[mycls,1:].flatten())
            mynds = mynds[mynds>-1]
            mynds.sort()
            idxinfo.append([mynds, myfcs, mycls])
        self.idxinfo = idxinfo
        # prepare mappers.
        ndmblk = self._count_max_nodeinblock(self.blk, part)
        ndmaps = empty((self.blk.nnode, 1+2*ndmblk), dtype='int32')
        fcmaps = empty((self.blk.nface, 5), dtype='int32')
        clmaps = empty((self.blk.ncell, 2), dtype='int32')
        ndmaps.fill(-1)
        ndmaps[:,0] = 0
        fcmaps.fill(-1)
        fcmaps[:,0] = 0
        clmaps.fill(-1)
        self.mappers.extend([ndmaps, fcmaps, clmaps])

    @staticmethod
    def _reindex(bemap, idxmap, cond=lambda arr: arr>=0):
        """
        Reindex generic arrays.
        """
        assert len(bemap.shape) == 1
        want = cond(bemap)
        bemap[want] = idxmap[bemap][want]

    @classmethod
    def _reindex_conn(cls, conn, idxmap):
        """
        Reindex connectivity arrays.
        """
        bemap = conn[:,1:].ravel()
        #want = bemap>=0
        #bemap[want] = idxmap[bemap][want]
        cls._reindex(bemap, idxmap)
        conn[:,1:] = bemap.reshape((conn.shape[0], conn.shape[1]-1))[:,:]

    def split(self, nblk=None, interface_type=None):
        """
        Split the whole block according to the partitioning information
        (self.idxinfo).

        In the body of this method, the length of the xxmap (i.e., ndmap,
        fcmap, and clmap) arrays are actually nxxxx+1, and xxmap[nxxxx] is set
        to be -1.  By doing this, the negative value (literally -1) in the
        index array to be mapped can be easily mapped to the correct -1,
        without complex branching logic.

        Outline the algorithm:
          1. Distribute connectivity information from whole-block into each
             sub-block.  At this point, the sub-blocks are instantiated.  BC
             objects remains untouched.
          2. With the knowledge of the split in mind, book-keep the
             connectivity across splitted boundary.  All the indices of nodes,
             faces, and cells are kept unchange at this point.
          3. Map the global indices of nodes, faces, and cells to be local to
             each sub-block.  At the same time, BC objects are distributed to
             sub-blocks.
          4. Create interface BC objects for the splitting.
          5. Supplement the ghost information for splitted blocks.  Also copy
             relating data across both ends of interfaces.

        @keyword nblk: if None, use already done partition information, else
            take it as an integer and call partition.
        @type nblk: int
        @keyword interface_type: BC type for the interface.  Must be a subclass
            of solvcon.boundcond.interface.  Setting it to None will use the
            default solvcon.boundcond.interface.
        @type interface_type: solvcon.boundcond.interface

        @return: nothing.
        """
        assert self.blk
        assert len(self) == 0
        from numpy import empty, arange
        from .block import Block
        from .boundcond import bctregy
        if isinstance(nblk, int):
            self.partition(nblk)
        part = self.part
        ndmaps, fcmaps, clmaps = self.mappers

        # Step 1: Distribute all data from the whole-block to each sub-block.
        iblk = 0
        for mynds, myfcs, mycls in self.idxinfo:
            blk = Block(ndim=self.blk.ndim,
                nnode=mynds.shape[0],
                nface=myfcs.shape[0],
                ncell=mycls.shape[0],
                fpdtype=self.blk.fpdtype,
            )
            blk.blkn = iblk
            # names of cell groups.
            blk.grpnames = self.blk.grpnames    # OK to use a single object.
            # basic metrics.
            blk.ndcrd[:,:] = self.blk.ndcrd[mynds,:]
            # type.
            blk.fctpn[:] = self.blk.fctpn[myfcs]
            blk.cltpn[:] = self.blk.cltpn[mycls]
            blk.clgrp[:] = self.blk.clgrp[mycls]
            # connectivity.
            blk.fcnds[:,:] = self.blk.fcnds[myfcs,:]
            blk.fccls[:,:] = self.blk.fccls[myfcs,:]
            blk.clnds[:,:] = self.blk.clnds[mycls,:]
            blk.clfcs[:,:] = self.blk.clfcs[mycls,:]
            # append.
            self.append(blk)
            iblk += 1

        # Step 2: Compute neighboring block information (also set rblkclmap).
        clmap = empty(self.blk.ncell+1, dtype='int32')
        clmap.fill(-1)
        #rblkclmap = empty(self.blk.ncell+1, dtype='int32')    # result.
        #rblkclmap.fill(-1)
        iblk = 0
        for blk in self:
            belong = blk.fccls[:,0]
            neibor = blk.fccls[:,1]
            neiblk = blk.fccls[:,2]
            neibcl = blk.fccls[:,3]
            mynode, myface, mycell = self.idxinfo[iblk]
            myfcidx = arange(blk.nface, dtype='int32')
            # Set cell map for whole-indices -> sub-indeces for each sub-block.
            clmap[mycell] = arange(blk.ncell, dtype='int32')
            # Swap belong and neibor for whose neighboring cell is not in the
            # current block.  This action ensures that if the face is connected
            # with a cell in the current block within either belong or neibor
            # array in the original whole block, the cell will appear in the
            # belong array in the sub-block.
            notmine = myfcidx[part[belong]!=iblk]
            buf = empty(len(notmine), dtype='int32')
            buf   [:]       = neibor[notmine]
            neibor[notmine] = belong[notmine]
            belong[notmine] = buf   [:]
            # Find out the faces with the non-ghost neibor which are not in the
            # current block.  Save neighboring block information (index) into
            # neiblk.  
            neiblk.fill(-1) # reset neiblk.
            notmine = myfcidx[(neibor>=0) & (part[neibor]!=iblk)]
            neiblk[notmine] = (part[neibor])[notmine]
            # Move the global indices of the neighboring cells in the
            # neighboring blocks to neibcl, and then set the corresponding
            # neibor to be negative.
            neibcl[notmine] = neibor[notmine]
            nbnd = (neibor<0).sum()
            neibor[notmine] = -nbnd-1   # it doesn't matter to be how negative.
            ## Save cell indices with neighboring block for later use.
            #notmine = belong[notmine]   # change meaning of "notmine" from
            #                            # face indices to cell indices.
            #rblkclmap[notmine] = clmap[notmine]
            # next.
            iblk += 1
        # Record maps for cells.
        clmaps[:,0] = clmap[:-1]
        clmaps[:,1] = part[:]

        # Step 3: Reindex nodes, faces, and cells, and distribute BCs.
        ndmap = empty(self.blk.nnode+1, dtype='int32')
        fcmap = empty(self.blk.nface+1, dtype='int32')
        clmap = clmap
        #rblkclmap = rblkclmap
        iblk = 0
        for blk in self:
            mynode, myface, mycell = self.idxinfo[iblk]
            # Build mapping. clmap is reused and needs not to be built here,
            # because there will be no coincident cells.
            ndmap.fill(-1)
            fcmap.fill(-1)
            ndmap[mynode] = arange(blk.nnode, dtype='int32')
            fcmap[myface] = arange(blk.nface, dtype='int32')
            # Reindex nodes and faces.
            self._reindex_conn(blk.fcnds, ndmap)
            self._reindex_conn(blk.clnds, ndmap)
            self._reindex_conn(blk.clfcs, fcmap)
            # Record maps for nodes and cells.
            locs = ndmaps[mynode,0]
            ndmaps[mynode,1+locs*2] = ndmap[mynode]
            ndmaps[mynode,1+locs*2+1] = iblk
            ndmaps[mynode,0] += 1
            locs = fcmaps[myface,0]
            fcmaps[myface,1+locs*2] = fcmap[myface]
            fcmaps[myface,1+locs*2+1] = iblk
            fcmaps[myface,0] += 1
            # Distribute BCs.
            bcs = list()
            for oldbc in self.blk.bclist:    # loop over all old BCs.
                # reindex face indices.
                fcs = fcmap[oldbc.facn[:,0]]
                fcs = fcs[fcs>=0]
                if len(fcs) == 0:   # judge if there are any faces to me?
                    continue    # null BC, skip to process next oldbc.
                fcs.sort()
                facn = empty((len(fcs),3), dtype='int32')
                facn.fill(-1)
                facn[:,0] = fcs[:]
                # create new BC object of the same type.
                bctype = type(oldbc)
                bc = bctype(fpdtype=oldbc.fpdtype)
                oldbc.cloneTo(bc)
                bc.sern = len(bcs)
                bc.blk  = blk
                bc.facn = facn
                bcs.append(bc)
            blk.bclist = bcs    # set newly created BC list to the block.
            # Reindex cells.
            blk.fccls[:,0] = clmap[blk.fccls[:,0]]
            neibor = blk.fccls[:,1].copy()
            neibor[neibor<0] = -1
            want = (blk.fccls[:,2]==-1)
            blk.fccls[want,1] = clmap[neibor][want]
            #want = (blk.fccls[:,2]!=-1)
            #blk.fccls[want,3] = rblkclmap[neibor][want]
            neibcl = blk.fccls[:,3]
            blk.fccls[:,3] = clmap[neibcl]
            # next.
            iblk += 1

        # Step 4: Build interface BC objects.
        if interface_type == None:
            interface_type = bctregy.interface
        assert issubclass(interface_type, bctregy.interface)
        ## create placeholder for the interface list.
        interfaces = self.interfaces
        for blk in self:
            interfaces.append([None]*len(self))
        ## build BC objects.
        iblk = 0
        for blk in self:
            # setup markers.
            slct = blk.fccls[:,1] < 0
            nbound = slct.sum()
            allfacn = arange(blk.nface, dtype='int32')[slct]
            specified = empty(nbound, dtype='bool')
            specified.fill(False)
            # mark boundary faces associated with a certain BC object.
            bcs = list()
            for bc in blk.bclist:
                slct = allfacn.searchsorted(bc.facn[:,0])
                specified[slct] = True
            # get unspecified faces.  If there are no faces left, continue to
            # the next block since there is nothing to do.
            leftfcs = allfacn[specified==False]
            neiblk = blk.fccls[leftfcs,2]
            if len(leftfcs) == 0:
                continue
            # create BC objects for interfaces.
            for jblk in range(len(self)):
                # take left faces connecting the current block (indexed with 
                # jblk).
                slct = (neiblk==jblk)
                leftj = leftfcs[slct]
                if jblk == iblk:
                    assert len(leftj) == 0
                if len(leftj) == 0: # nothing to do.
                    continue
                # find out faces in the related block.
                idx1 = min(iblk, jblk)
                idx2 = max(iblk, jblk)
                slct = ((fcmaps[:,2] == idx1) & (fcmaps[:,4] == idx2))
                dupfcs = fcmaps[slct]
                if jblk > iblk:
                    rfcs = dupfcs[:,3]
                else:
                    rfcs = dupfcs[:,1]
                # create interface BC object.
                bc = interface_type()
                bc.name = "interface_%d_%d" % (iblk, jblk)
                bc.sern = len(blk.bclist)
                bc.blk = self[iblk]
                bc.blkn = iblk
                bc.rblkn = jblk
                bc.facn = empty((len(leftj),3), dtype='int32')
                bc.facn[:,0] = leftj[:] # facn[:,1] set in the next step.
                bc.facn[:,2] = rfcs[:]
                blk.bclist.append(bc)
                # assign to interface list.
                interfaces[iblk][jblk] = bc
            # next.
            iblk += 1

        # Step 5: Supplement the rest of the blocks.
        for blk in self:
            blk.calc_metric()
            blk.build_boundary()
            blk.build_ghost()
        # copy ghost information between interface.
        for blk in self:
            for bc in blk.bclist:
                if not isinstance(bc, bctregy.interface):
                    continue
                bc.relateCells(self)
                rblk = self[bc.rblkn]
                slctm = bc.rclp[:,0] + blk.ngstcell
                slctr = bc.rclp[:,1] + rblk.ngstcell
                blk.shcltpn[slctm] = rblk.shcltpn[slctr]
                blk.shclgrp[slctm] = rblk.shclgrp[slctr]
                blk.shclcnd[slctm,:] = rblk.shclcnd[slctr,:]
                blk.shclvol[slctm,:] = rblk.shclvol[slctr,:]

    def merge(self):
        assert self.nblk > 1

class Distributed(Collective):
    """
    Domain distributed over the network.
    """
