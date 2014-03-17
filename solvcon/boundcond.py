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
Primitive boundary condition definition.

Basic supportive logic for interface BCs, periodic BCs, and glued BCs is
defined here.
"""

import warnings

from .gendata import TypeNameRegistry, TypeWithBinder
try: # for readthedocs to work.
    from . import mesh
except ImportError:
    warnings.warn("solvcon.mesh isn't built", RuntimeWarning)

class Glue(object):
    """
    Glue two boundary conditions which are considered to be collocated.

    @cvar CACHE_KEYS_ENABLER: names of arrays whose original values to be
        cached when the glue is enabled.
    @ctype CACHE_KEYS_ENABLER: tuple
    @ivar sbc: source BC object.
    @itype sbc: solvcon.boundcond.BC
    @ivar reciprocal: the glue object on the other side.
    @itype reciprocal: Glue
    @ivar ref: a reference point for matching bounding faces.
    @itype ref: numpy.ndarray
    @ivar cache: cache for mesh data.
    @itype cache: dict
    @ivar scls: source cell list shifted by ngstcell.
    @itype scls: numpy.ndarray
    @ivar dcls: destination cell list shifted by ngstcell.
    @itype dcls: numpy.ndarray
    """

    CACHE_KEYS_ENABLER = ('cltpn', 'clgrp', 'clvol', 'clcnd')

    def __init__(self, sbc, dbc, ref=None, reciprocal=None):
        """
        The constructor will create a PAIR of Glue objects attached to the
        input source BC object (sbc) and destination BC object (dbc).  Each
        Glue object in the pair is the reciprocal of the other.  The
        constructor DOES NOT modify associated BC or BlockSolver objects except
        setting BC objects' glue property to self.  If no reference point is
        given through ref keyword, a random reference point is generated for
        sorting boundary faces to be glued.

        @param sbc: source BC object.
        @type sbc: solvcon.boundcond.BC
        @param dbc: destination BC object.
        @type dbc: solvcon.boundcond.BC
        @keyword ref: a reference point for matching bounding faces.
        @type ref: numpy.ndarray
        @keyword reciprocal: the glue object on the other side.
        @type reciprocal: Glue
        """
        from random import random
        from numpy import array
        svr = sbc.svr
        ngstface = svr.ngstface
        ngstcell = svr.ngstcell
        ndim = svr.ndim
        fpdtype = svr.fpdtype
        # set source BC object and cache container.
        self.sbc = sbc
        self.cache = dict()
        # calculate and set a reference point and use it to sort faces.
        if ref is None:
            ref = array([random() for it in range(ndim)], dtype=fpdtype)
        assert len(ref) == ndim
        self.ref = ref
        # calculate and set cell lists.
        self.scls = svr.fccls[self.__sortfcs(sbc,ref),1] + ngstcell
        self.dcls = svr.fccls[self.__sortfcs(dbc,ref),0] + ngstcell
        assert (self.scls<ngstcell).all()
        assert (self.dcls>=ngstcell).all()
        # set properties to source BC.
        assert hasattr(sbc, 'glue')
        sbc.glue = self
        # create and set reciprocal if self is the first in a pair.
        if reciprocal is None:
            self.reciprocal = Glue(dbc, sbc, ref=ref, reciprocal=self)
        else:
            assert isinstance(reciprocal, Glue)
            self.reciprocal = reciprocal

    @staticmethod
    def __sortfcs(bc, ref):
        """
        Get the sorted face list of a BC object according to the reference
        point.

        @param bc: a BC object.
        @type bc: solvcon.boundcond.BC
        @param ref: reference point.
        @type ref: numpy.ndarray
        @return: sorted face indices shifted by ngstface.
        @rtype: numpy.ndarray
        """
        cnd = bc.svr.fccnd[bc.facn[:,0]+bc.svr.ngstface,:]
        dist = ((cnd - ref)**2).sum(axis=1)
        return bc.facn[dist.argsort(),0]+bc.svr.ngstface

    def enable(self, *args, **kw):
        """
        Enable this glue by setting ghost information from interior and store
        original values in a cache.  Arguments are the keys of arrays to be
        cached.

        @keyword with_default: use CACHE_KEYS_ENABLER anyway.  Default True.
        @type with_default: bool
        @return: nothing
        """
        keys = list(args)
        if args and kw.get('with_default', True):
            keys.extend(self.CACHE_KEYS_ENABLER)
        for key in keys:
            arr = getattr(self.sbc.svr, key)
            self.cache[key] = arr[self.scls].copy()
            self.take(key)

    def disable(self, *args, **kw):
        """
        Disable this glue by restoring ghost information from cached values.
        Arguments are the keys of cached arrays.

        @keyword with_default: use CACHE_KEYS_ENABLER anyway.  Default True.
        @type with_default: bool
        @return: nothing
        """
        keys = list(args)
        if args and kw.get('with_default', True):
            keys.extend(self.CACHE_KEYS_ENABLER)
        for key in keys:
            arr = getattr(self.sbc.svr, key)
            arr[self.scls] = self.cache[key]

    def take(self, key):
        """
        Take array values from interior cells to ghost cells.

        @param key: the name of the array.
        @type key: str
        @return: nothing
        """
        arr = getattr(self.sbc.svr, key)
        arr[self.scls] = arr[self.dcls]

bctregy = TypeNameRegistry()  # registry singleton.
class BCMeta(TypeWithBinder):
    """
    Meta class for boundary condition class.
    """
    def __new__(cls, name, bases, namespace):
        newcls = super(BCMeta, cls).__new__(cls, name, bases, namespace)
        bctregy.register(newcls)
        return newcls

# Base/abstract BC type.
class BC(object):
    """
    Generic boundary condition abstract class; the base class that all boundary
    condition classes should subclass.

    FIXME: provide doctests as examples.
    """

    #: Boundary face relation number.
    BFREL = 3

    __metaclass__ = BCMeta

    #: Holding names as pointers.  Used for binder.
    _pointers_ = []

    #: Settable value names.
    vnames = []
    #: Default values.
    vdefaults = {}

    def __init__(self, bc=None, fpdtype=None):
        """
        Initialize object with empty values or from another BC object.

        @param bc: Another BC object.
        @type bc: solvcon.boundcond.BC
        """
        import numpy as np
        from numpy import empty
        from .conf import env
        # determine fpdtype.
        if fpdtype == None:
            if bc == None:
                self.fpdtype = env.fpdtype
            else:
                self.fpdtype = bc.fpdtype
        else:
            self.fpdtype = fpdtype
        if isinstance(self.fpdtype, str):
            self.fpdtype = getattr(np, self.fpdtype)
        if bc is None:
            # general data.
            #: Serial number (for certain block).
            self.sern = None
            #: Identifying name.
            self.name = None
            #: The :py:class:`~solvcon.block.Block` object associated with this
            #: :py:class:`BC` object.
            self.blk  = None
            #: Serial number of self :py:attr:`blk`.
            self.blkn = None
            #: Associated :py:class :py:class:`~solvcon.solver.MeshSolver`
            #: object.
            self.svr  = None
            #: Associated :py:class:`Glue` object gluing two collocated
            #: :py:class:`BC` objects.
            self.glue = None
            #: An :py:class:`numpy.ndarray` as a list of faces.  First column
            #: is the face index in block.  The second column is the face index
            #: in bndfcs.  The third column is the face index of the related
            #: block (if exists).
            self.facn = empty((0,self.BFREL), dtype='int32')
            #: An :py:class:`numpy.ndarray` for attached (specified) value for
            #: each boundary face.
            self.value = empty((0,self.nvalue), dtype=self.fpdtype)
        else:
            bc.cloneTo(self)
        super(BC, self).__init__()

    @property
    def fpdtypestr(self):
        from .dependency import str_of
        return str_of(self.fpdtype)

    @property
    def nvalue(self):
        """
        Return the length of :py:attr:`vnames` as number of values per boundary
        face.  It should be equivalent to the second shape element of
        :py:class:`value`.

        FIXME: provide doctests.
        """
        return len(self.vnames)

    def __len__(self):
        """
        Return the first shape element of :py:class:`facn`.

        FIXME: provide doctests.
        """
        return self.facn.shape[0]

    def __str__(self):
        return "[%s#%s \"%s\": %d faces with %d values]" % (
            self.__class__.__name__, self.sern, self.name,
            len(self), self.nvalue)

    def create_bcd(self):
        """
        :return: An object contains the :c:type:`sc_bound_t` variable for C
            interfacing.
        :rtype: :py:class:`solvcon.mesh.Bound`

        The following code shows how and when to use this method:

        >>> import numpy as np
        >>> # craft some face numbers for testing.
        >>> bndfcs = [0,1,2]
        >>> # craft the BC object for testing.
        >>> bc = BC()
        >>> bc.name = 'some_name'
        >>> bc.facn = np.empty((len(bndfcs), BC.BFREL), dtype='int32')
        >>> bc.facn.fill(-1)
        >>> bc.facn[:,0] = bndfcs
        >>> bc.sern = 0
        >>> bc.blk = None # should be set to a block.
        >>> # test for this method.
        >>> bcd = bc.create_bcd()
        """
        bcd = mesh.Bound()
        bcd.setup_bound(self)
        return bcd

    def cloneTo(self, another):
        """
        :param another: Another BC object.
        :type another: solvcon.boundcond.BC
        :return: Nothing.

        Clone self to *another* :py:class:`BC` object.
        """
        assert issubclass(type(another), type(self))
        assert another.fpdtype == self.fpdtype
        another.sern = self.sern
        another.name = self.name
        another.blk  = self.blk
        another.blkn = self.blkn
        another.svr  = self.svr
        another.glue = None
        another.facn = self.facn.copy()
        another.value = self.value.copy()

    def feedValue(self, vdict):
        """
        Get feed values to self boundary condition.

        @param vdict: name and value pairs.
        @type vdict: dict
        @return: nothing.
        """
        from numpy import empty
        # get name-value pairs for each value.
        vpairs = self.vdefaults.copy()
        for vn in vdict:
            if vn in self.vnames:
                vpairs[vn] = vdict[vn]
        # set values to array.
        nbfcs = len(self)
        nvalue = len(self.vnames)
        values = empty((nbfcs, nvalue), dtype=self.fpdtype)
        for iv in range(nvalue):
            vn = self.vnames[iv]
            values[:,iv].fill(vpairs[vn])
        # save set array.
        self.value = values

    def gluetake(self, key):
        """
        Use the attached Glue object to update the array specified by key.

        @param key: array name.
        @type key: str
        @return: nothing
        """
        svr = self.svr
        if svr.scu: svr.cumgr.arr_from_gpu(key)
        self.glue.take(key)
        if svr.scu: svr.cumgr.arr_to_gpu(key)

    def init(self, **kw):
        """
        Initializer.
        """
        pass

    def final(self, **kw):
        """
        Finalizer.
        """
        pass

class unspecified(BC):
    """
    Abstract BC type for unspecified boundary conditions.
    """

class interface(BC):
    """
    Abstract BC type for interface between blocks.
    
    @ivar rblkn: index number of related block.
    @itype rblkn: int
    @ivar rclp: list of related cell pairs.  The first column is for the
        indices of the ghost cells belong to self block.  The second column is
        for the indices of the cells belong to the related block.  The third
        column is for the indices of the cells belong to self block.
    @itype rclp: numpy.ndarray
    """

    def __init__(self, **kw):
        from numpy import empty
        super(interface, self).__init__(**kw)
        self.rblkn = getattr(self, 'rblkn', -1)
        self.rblkinfo = empty(6, dtype='int32')
        self.rclp = empty((0,3), dtype='int32')

    def cloneTo(self, another):
        super(interface, self).cloneTo(another)
        another.rblkn = self.rblkn
        another.rblkinfo = self.rblkinfo.copy()
        another.rclp = self.rclp.copy()

    def relateCells(self, dom):
        """
        Calculate self.rclp[:,:] form the information about related block
        provided by dom parameter.

        @param dom: Collective domain for information about related block.
        @type dom: solvcon.domain.Collective
        @return: nothing.
        """
        from numpy import empty
        facn = self.facn
        blk = self.blk
        rblk = dom[self.rblkn]
        # fill informations from related block.
        self.rblkinfo[:] = (rblk.nnode, rblk.ngstnode,
            rblk.nface, rblk.ngstface, rblk.ncell, rblk.ngstcell)
        # calculate indices of related cells.
        self.rclp = empty((len(self),3), dtype='int32')
        self.rclp[:,0] = blk.fccls[facn[:,0],1]
        self.rclp[:,1] = rblk.fccls[facn[:,2],0]
        self.rclp[:,2] = blk.fccls[facn[:,0],0]
        # assertion.
        assert (self.rclp[:,0]<0).all()
        assert (self.rclp[:,1]>=0).all()
        assert (self.rclp[:,2]>=0).all()

class periodic(BC):
    """
    BC type for periodic boundary condition.
    """

    def __init__(self, **kw):
        from numpy import empty
        super(periodic, self).__init__(**kw)
        self.rblkn = getattr(self, 'rblkn', -1)
        self.rblkinfo = empty(6, dtype='int32')
        self.rclp = empty((0,3), dtype='int32')

    def cloneTo(self, another):
        super(periodic, self).cloneTo(another)
        another.rblkn = self.rblkn
        another.rblkinfo = self.rblkinfo.copy()
        another.rclp = self.rclp.copy()

    def sort(self, ref):
        if ref is None:
            return
        from numpy import sqrt
        dist = sqrt(((self.blk.fccnd[self.facn[:,0],:] - ref)**2).sum(axis=1))
        slct = dist.argsort()
        self.facn = self.facn[slct,:]

    def couple(self, rbc):
        """
        Calculate self.rclp[:,:] form the information about related BC object
        provided by rbc parameter.

        @param rbc: Related BC object.
        @type rbc: solvcon.boundcond.periodic
        @return: nothing.
        """
        from numpy import empty
        blk = self.blk
        facn = self.facn
        try:
            facn[:,2] = rbc.facn[:,0]
            # fill informations from related block.
            self.rblkinfo[:] = (blk.nnode, blk.ngstnode,
                blk.nface, blk.ngstface, blk.ncell, blk.ngstcell)
            # calculate indices of related cells.
            self.rclp = empty((len(self),3), dtype='int32')
            self.rclp[:,0] = blk.fccls[facn[:,0],1]
            self.rclp[:,1] = blk.fccls[facn[:,2],0]
            self.rclp[:,2] = blk.fccls[facn[:,0],0]
            # assertion.
            assert (self.rclp[:,0]<0).all()
            assert (self.rclp[:,1]>=0).all()
            assert (self.rclp[:,2]>=0).all()
            # copy metrics.
            slctm = self.rclp[:,0] + blk.ngstcell
            slctr = self.rclp[:,1] + blk.ngstcell
            blk.shcltpn[slctm] = blk.shcltpn[slctr]
            blk.shclgrp[slctm] = blk.shclgrp[slctr]
            blk.shclvol[slctm] = blk.shclvol[slctr]
            # move coordinates.
            shf = blk.shclcnd[slctr,:] - blk.fccnd[facn[:,2],:]
            blk.shclcnd[slctm,:] = self.blk.fccnd[facn[:,0],:] + shf
        except Exception as e:
            e.args = tuple(list(e.args) + [
                'self bc \'%s\' to rbc \'%s\'' % (self.name, rbc.name)])
            raise

    @staticmethod
    def couple_all(blk, bcmap):
        """
        Couple all periodic boundary conditions.

        @param blk: the block having periodic BCs to be coupled.
        @type blk: solvcon.block.Block
        @param bcmap: mapper for periodic BCs.
        @type bcmap: dict
        @return: nothing
        """
        from solvcon.boundcond import periodic
        nmidx = dict([(blk.bclist[idx].name, idx) for idx in
            range(len(blk.bclist))])
        npidx = list()
        for key in bcmap:
            bct, vdict = bcmap[key]
            if not issubclass(bct, periodic):
                try:
                    if key in nmidx:
                        npidx.append(nmidx[key])
                except Exception as e:
                    args = list(e.args)
                    args.append(str(nmidx))
                    e.args = tuple(args)
                    raise
                continue
            val = vdict['link']
            ibc0 = nmidx[key]
            ibc1 = nmidx[val]
            pbc0 = blk.bclist[ibc0] = bct(bc=blk.bclist[ibc0])
            pbc1 = blk.bclist[ibc1] = bct(bc=blk.bclist[ibc1])
            ref = vdict.get('ref', None)
            pbc0.sort(ref)
            pbc1.sort(ref)
            pbc0.couple(pbc1)
            pbc1.couple(pbc0)
