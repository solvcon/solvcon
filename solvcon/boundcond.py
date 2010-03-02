# -*- coding: UTF-8 -*-
# Copyright (C) 2008-2009 by Yung-Yu Chen.  See LICENSE.txt for terms of usage.

"""Primitive boundary condition definition."""

from .gendata import SingleAssignDict, AttributeDict, TypeWithBinder

class BcTypeRegistry(SingleAssignDict, AttributeDict):
    """
    BC type registry class, and its instance holds BC type classes, which can
    be indexed by BC type name and BC type number. 
    
    In current design, there should exist only one registry singleton in 
    package.

    BC classes in registry should not be altered, in any circumstances.
    """
    def register(self, bctype):
        name = bctype.__name__
        self[name] = bctype
        return bctype
bctregy = BcTypeRegistry()  # registry singleton.

class BCMeta(TypeWithBinder):
    """
    Meta class for boundary condition class.
    """
    def __new__(cls, name, bases, namespace):
        newcls = super(BCMeta, cls).__new__(cls, name, bases, namespace)
        # check class value.
        if not isinstance(newcls.typn, int):
            raise ValueError, \
                "typn has to be an integer, but found to be %s." % newcls.typn
        # register.
        bctregy.register(newcls)
        return newcls

# Base/abstract BC type.
class BC(object):
    """
    Generic boundary condition abstract class.  It's the base class that all
    boundary condition class should subclass.

    @cvar typn: type number of this boundary condition.  Negative value means 
        abstract class (boundary condition).
    @type typn: int
    @cvar vnames: settable value names.
    @type vnames: list
    @cvar vdefaults: default values.
    @type vdefaults: dict

    @ivar sern: serial number (for certain block).
    @type sern: int
    @ivar name: name.
    @type name: str
    @ivar blk: the block associated with this BC object.
    @type blk: solvcon.block.Block
    @ivar blkn: serial number of self block.
    @type blkn: int
    @ivar solver: solver object.
    @type solver: solvcon.solver.core.Solver
    @ivar facn: list of faces.  First column is the face index in block.  The
        second column is the face index in bndfcs.  The third column is the
        face index of the related block (if exists).
    @type facn: numpy.ndarray
    @ivar value: attached value for each boundary face.
    @type value: numpy.ndarray
    """

    __metaclass__ = BCMeta

    _pointers_ = [] # for binder.

    typn = -1   # BC type number, must be an integer.
    vnames = [] # settable value names.
    vdefaults = {}  # defaults to values.

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
            self.sern = None
            self.name = None
            self.blk  = None
            self.blkn = None
            self.solver = None
            # face list.
            self.facn = empty((0,3), dtype='int32')
            # attached (specified) value.
            self.value = empty((0,self.nvalue), dtype=self.fpdtype)
        else:
            bc.cloneTo(self)
        super(BC, self).__init__()

    @property
    def fpdtypestr(self):
        from .dependency import str_of
        return str_of(self.fpdtype)
    @property
    def fpptr(self):
        from .dependency import pointer_of
        return pointer_of(self.fpdtype)

    @property
    def nvalue(self):
        return len(self.vnames)

    def __len__(self):
        return self.facn.shape[0]

    def __str__(self):
        return "[%s(%d)#%d \"%s\": %d faces with %d values]" % (
            self.__class__.__name__, self.typn, self.sern, self.name,
            len(self), self.nvalue)

    def cloneTo(self, another):
        """
        Clone self to passed-in another BC object.

        @param another: Another BC object.
        @type another: solvcon.boundcond.BC
        """
        assert issubclass(type(another), type(self))
        assert another.fpdtype == self.fpdtype
        another.sern = self.sern
        another.name = self.name
        another.blk  = self.blk
        another.blkn = self.blkn
        another.solver = self.solver
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

    def sol(self):
        """
        Update ghost cells after marchsol.
        """
        pass

    def dsol(self):
        """
        Update ghost cells after marchdsol.
        """
        pass

class unspecified(BC):
    """
    Abstract BC type for unspecified boundary conditions.
    """
    typn = -2**15   # give the minimal value for a 2 bytes integer.

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
    typn = -2**15+1

    def __init__(self, bc=None):
        from numpy import empty
        super(interface, self).__init__(bc)
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
